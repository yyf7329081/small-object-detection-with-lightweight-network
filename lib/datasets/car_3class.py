from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .car_3class_eval import car_3class_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete


class car_3class(imdb):
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'CAR_3CLASS_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
            else devkit_path
        self._data_path = os.path.join(self._devkit_path, '3class')
        self._classes = (
            '__background__',  # always index 0
            'car',
            'bus',
            'truck')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {
            'cleanup': True,
            'use_salt': True,
            'use_diff': False,
            'matlab_eval': False,
            'rpn_file': None,
            'min_size': 2
        }

        assert os.path.exists(self._devkit_path), \
            'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, index + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + VisDrone2018/ImageSets/val.txt
        image_set_file = os.path.join(self._devkit_path, '3class_' + self._image_set + '_0209' + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file, 'r', encoding='utf-8') as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where PASCAL VOC is expected to be installed.
        """
        # return os.path.join(cfg.DATA_DIR, 'VOCdevkit' + self._year)
        return os.path.join(cfg.DATA_DIR, 'VOCdevkit')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [
            self._load_pascal_annotation(index) for index in self.image_index
        ]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'testset':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        assert os.path.exists(filename), \
            'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = pickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(
            os.path.join(cfg.DATA_DIR, 'selective_search_data',
                         self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, index + '.txt')
        f = open(filename, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        lines = lines[1:]
        #num_objs = len(lines)
        boxes = np.empty((0, 4), dtype=np.uint16)
        gt_classes = np.empty((0), dtype=np.int32)
        overlaps = np.empty((0, self.num_classes), dtype=np.float32)
        seg_areas = np.empty((0), dtype=np.float32)
        ishards = np.empty((0), dtype=np.int32)
        for line in lines:
            line = line.strip()
            data = line.split(' ')
            cls_name = data[0]
            if cls_name == 'c':
                cls = 1
            elif cls_name == 'b':
                cls = 2
            else:
                cls = 3
            x1 = float(data[1])-1
            y1 = float(data[2])-1
            w = float(data[3])
            h = float(data[4])
            x2 = x1 + w
            y2 = y1 + h
            trunc = 0
            occ = 0
            ishards = np.append(ishards, [int(trunc and occ)], axis=0)
            boxes = np.append(boxes, [[x1, y1, x2, y2]], axis=0)
            gt_classes = np.append(gt_classes, [cls], axis=0)
            overlap = [0.0 for i in range(self.num_classes)]
            overlap[cls] = 1.0
            overlaps = np.append(overlaps, [overlap], axis=0)
            seg_area = (x2 - x1 + 1) * (y2 - y1 + 1)
            seg_areas = np.append(seg_areas, [seg_area], axis=0)
        
        overlaps = scipy.sparse.csr_matrix(overlaps)
        
        return {
            'boxes': boxes,# (num_box, 4),(x1,y1,x2,y2),绝对坐标
            'gt_classes': gt_classes,# (num_box,),种类编号
            'gt_ishard': ishards,# (num_box,)
            'gt_overlaps': overlaps,# (num_box, num_class)，每个box对应类为1.0，因为是gt_box
            'flipped': False,
            'ver_flipped': False,
            'seg_areas': seg_areas # (num_box,)gt_box的绝对面积
        }
        

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' +
                   self._salt if self.config['use_salt'] else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self, output_dir):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id(
        ) + '_det_' + self._image_set + '_{:s}.txt'
        filedir = os.path.join(output_dir, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, output_dir, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write(
                            '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(
                                index, dets[k, -1], dets[k, 0] + 1,
                                dets[k, 1] + 1, dets[k, 2] + 1,
                                dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(self._data_path, '{:s}.txt')
        imagesetfile = os.path.join(self._devkit_path, '3class_' + self._image_set + '_0209' + '.txt')
        cachedir = os.path.join(output_dir, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = True
        #use_07_metric = True if int(self._year) < 2010 else False
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template(output_dir).format(cls)
            rec, prec, ap = car_3class_eval(filename,# 每一类的测试结果，图片名，得分，四个坐标
                                     annopath,
                                     imagesetfile,
                                     i,
                                     cls,
                                     cachedir,
                                     ovthresh=0.5,
                                     use_07_metric=use_07_metric)
            aps += [ap]
            if len(prec)==0:
                prec=[0]
            if len(rec)==0:
                rec=[0]
            print('{}: rec={:.4f}, prec={:.4f}, ap={:.4f}'.format(cls, rec[-1], prec[-1], ap))
            #print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

        return np.mean(aps)

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'visdrone_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._get_comp_id(),
                    self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)


    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(output_dir, all_boxes)
        ap_50 = self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template(output_dir).format(cls)
                os.remove(filename)
        return ap_50

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True


if __name__ == '__main__':
    d = visdrone('trainset', '2018')
    res = d.roidb
    from IPython import embed

    embed()
