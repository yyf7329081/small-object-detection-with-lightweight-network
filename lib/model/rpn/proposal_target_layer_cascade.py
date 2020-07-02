
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn

from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
from ..utils.config import cfg


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(
            cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(
            cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(
            cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)# (bs,post_nms_topN + num_gt,5)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)# 256
        fg_rois_per_image = int(
            np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))# 0.25*256
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image



        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image, rois_per_image,
            self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data,
                                            labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch# (bs,rois_per_image)
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image,
                                            4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)# 是前景的roi索引
            for i in range(inds.numel()):

                ind = inds[i]
                
                #class_weight = cfg.TRAIN.CLASS_WEIGHT[int(clss[b][ind].item())-1]
                
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS # (bs, rois_per_image, 4)

        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = (
                (targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets)) /
                self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image,
                             rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)# (bs, num_rois = post_nms_topN+num_gt, num_gt)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)# (bs, num_rois = post_nms_topN+num_gt)，对每个roi求与之iou最大的gt

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)# 每张图片的proposal个数，post_nms_topN + max_num_gt
        num_boxes_per_img = overlaps.size(2)# 每张图片最多gt_box的个数

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        # changed indexing way for pytorch 1.0
        labels = gt_boxes[:, :, 4].contiguous().view(-1)[(
            offset.view(-1), )].view(batch_size, -1)# (bs,num_rois = post_nms_topN+num_gt)，每个batch中roi对应的分类标签

        labels_batch = labels.new(batch_size, rois_per_image).zero_()# (bs, rois_per_image)，每个batch中选出的roi对应的分类标签
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()# (bs, rois_per_image, 5)，每个batch中选出的roi的坐标和batch_ind
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()# (bs, rois_per_image, 5)，roi对应的gt_box，(bs,rois_per_image,5)，[x1, y1, x2, y2, label]
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):# 对batch中每张图片

            fg_inds = torch.nonzero(# roi与gt的最大iou > cfg.TRAIN.FG_THRESH=0.5，标定为正样本
                max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)# (正样本数,)，正样本在rois中的索引
            fg_num_rois = fg_inds.numel()# 正样本roi数量

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            # print(cfg.TRAIN.BG_THRESH_HI)
            # print(cfg.TRAIN.BG_THRESH_LO)

            bg_inds = torch.nonzero(# 0.0=cfg.TRAIN.BG_THRESH_LO <= 与gt最大iou < cfg.TRAIN.BG_THRESH_HI=0.3，标定为负样本
                (max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI)
                & (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()# 负样本roi数量

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg，前景随机选择256*0.25个
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                
                # -------------------------------------------------------------------------------------------------------
                ## 平衡采样
                #fg_inds_inds = torch.empty(size=[1,],dtype=torch.long)# 记录选择的fg_inds的索引
                #fg_inds_rest_inds = torch.empty(size=[1,],dtype=torch.long)# 记录平衡采样之后剩下的fg_inds的索引，用于最后补充采样
                #labels_numpy = labels[i][fg_inds].cpu().numpy()# 当前正样本rois对应的标签
                #label_ind, label_num = np.unique(labels_numpy,return_counts=True)# 标签种类，和每一类的rois个数 
                #labels_tensor = torch.from_numpy(labels_numpy)
                #fg_rois_per_class = fg_rois_per_this_image//len(label_ind)# 每个类别需要采样的roi个数
                #for ind,class_ind in enumerate(label_ind):# 对每一类分别采样
                    #mask = torch.nonzero(labels_tensor == class_ind).view(-1)# fg_inds中对应是某一类的索引
                    #num = label_num[ind]# fg_inds中该类别的rois个数
                    #assert num == len(mask)
                    #if num <= fg_rois_per_class:# 如果该类别rois个数比该类需要采样的roi个数小
                        #fg_inds_inds = torch.cat([fg_inds_inds, mask], 0)# 则全部保留
                    #else:
                        #rand_num = torch.from_numpy(np.random.permutation(num)).type_as(gt_boxes).long()
                        #rand_inds = rand_num[:fg_rois_per_class]# 从该类rois中随机选择该类需要采样的mask的索引
                        #rest_inds = rand_num[fg_rois_per_class:]# 随机选择后剩下的mask的索引
                        #fg_inds_inds = torch.cat([fg_inds_inds, mask[rand_inds]], 0)
                        #fg_inds_rest_inds = torch.cat([fg_inds_rest_inds, mask[rest_inds]], 0)# 保留剩下的rois在fg_inds中的索引
                #fg_inds_inds = fg_inds_inds[1:].view(-1)
                #fg_inds_rest_inds = fg_inds_rest_inds[1:].view(-1)
                #fg_rois_rest_num = fg_rois_per_this_image - len(fg_inds_inds)# 剩下需要补充采样的rois个数
                #assert fg_rois_rest_num>=0
                #rand_num = torch.from_numpy(np.random.permutation(len(fg_inds_rest_inds))).type_as(gt_boxes).long()
                #fg_inds_inds = torch.cat([fg_inds_inds, fg_inds_rest_inds[rand_num[:fg_rois_rest_num]]], 0)
                #fg_inds = fg_inds[fg_inds_inds]
                # -------------------------------------------------------------------------------------------------------

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_num_rois).long().cuda()
                
                rand_num = torch.from_numpy(np.random.permutation(
                    fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg，随机选择剩下的背景roi
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                # rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(
                    np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                # rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(
                    np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(
                    np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError(
                    "bg_num_rois = 0 and fg_num_rois = 0, this should not happen!"
                )

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)# (选择样本数=256,)，选择的roi索引

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])# 每个batch从proposal筛选出来的(post_nms_topN+num_gt)个roi中选择的roi的分类标签(bs,rois_per_image=256)

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]# 选出的roi，(bs,rois_per_image=256,5)，(batch_index, x1, y1, x2, y2)
            rois_batch[i, :, 0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]# roi对应的gt_box，(bs,rois_per_image,5)，[x1, y1, x2, y2, label]

        bbox_target_data = self._compute_targets_pytorch(
            rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])# (bs, num_rois, 4)，真实的偏移参数
        
        # 保留是前景的roi对应的偏移参数；bbox_inside_weights前景为1，背景都设为0
        # bbox_targets(bs, rois_per_image, 4), bbox_inside_weights(bs, rois_per_image, 4)
        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
