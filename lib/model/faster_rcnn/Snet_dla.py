from .modules import *

from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.utils.config import cfg

from .dla_up import dla34up


class snet(_fasterRCNN):
    def __init__(self,
                 classes,
                 layer ,
                 pretrained_path=None,
                 class_agnostic=False,
                ):
        self.pretrained_path = pretrained_path

        self.class_agnostic = class_agnostic

        self.dout_base_model = 256
        self.layer = layer

        self.dout_lh_base_model = 245

        _fasterRCNN.__init__(self,
                             classes,
                             class_agnostic,
                             compact_mode=True)
        
    def _initialize_weights(self):

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        if  self.pretrained_path is not None:# 如果有预训练权重，就固定conv1,stage1,所有bn层参数。          
            
            print("Loading pretrained weights from %s" % (self.pretrained_path))
            if torch.cuda.is_available():
                pretrained_dict = torch.load(self.pretrained_path)['model']
            model_dict = self.state_dict()

            exclude_dict = ['RCNN_cls_score.weight', 'RCNN_cls_score.bias', 'RCNN_bbox_pred.weight', 'RCNN_bbox_pred.bias']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in exclude_dict}
            #include_dict = ['RCNN_base','rpn']
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.split('.')[0] in include_dict}
            #for k, v in pretrained_dict.items():
                #print(k)
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict,strict=False)
            
            for para in self.RCNN_base.base.base_layer.parameters():
                para.requires_grad = False
            for para in self.RCNN_base.base.level0.parameters():
                para.requires_grad = False
            for para in self.RCNN_base.base.level1.parameters():
                para.requires_grad = False            
            #for para in self.RCNN_cls_score.parameters():
                #para.requires_grad = False
            #for para in self.RCNN_bbox_pred.parameters():
                #para.requires_grad = False
            #for para in self.rpn.parameters():
                #para.requires_grad = False   
            #for para in self.sam.parameters():
                #para.requires_grad = False 
            #for para in self.RCNN_rpn.parameters():
                #para.requires_grad = False      
            #for para in self.RCNN_proposal_target.parameters():
                #para.requires_grad = False 
            
            set_bn_fix(self.RCNN_base.base.base_layer)
            set_bn_fix(self.RCNN_base.base.level0)
            set_bn_fix(self.RCNN_base.base.level1)
            #set_bn_fix(self.RCNN_top)
            #set_bn_fix(self.RCNN_cls_score)
            #set_bn_fix(self.RCNN_bbox_pred)
            #set_bn_fix(self.rpn)
            #set_bn_fix(self.sam)
            #set_bn_fix(self.RCNN_rpn)
            #set_bn_fix(self.RCNN_proposal_target) 
            print('freeze parameters.')

        else:# 如果没有预训练权重，就全部随机初始化。
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d):
                    if 'first' in name:
                        nn.init.normal_(m.weight, 0, 0.01)
                    else:
                        nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0001)
                    nn.init.constant_(m.running_mean, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0001)
                    nn.init.constant_(m.running_mean, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def _init_modules(self):
        snet = dla34up(pretrained_base=None, down_ratio=cfg.FEAT_STRIDE)




        # Build snet.
        self.RCNN_base = snet

        # Fix Layers
        # if self.pretrained:
        #     for layer  in self.RCNN_base:
        #         print(layer)
        #         for p in self.RCNN_base[layer].parameters():
        #             p.requires_grad = False


        self.RCNN_top = nn.Sequential(nn.Linear(5 * 7 * 7, 1024),
                                          nn.ReLU(inplace=True),

                                         )


        c_in = 1024

        self.RCNN_cls_score = nn.Linear(c_in, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(c_in, 4 * self.n_classes)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            #self.RCNN_base.conv1.eval()
            #self.RCNN_base.stage1.eval()
            #self.RCNN_base.conv1.train()
            #self.RCNN_base.stage1.train()
            #self.RCNN_base.stage2.train()
            #self.RCNN_base.stage3.train()


            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            #set_bn_eval(self.RCNN_base.conv1)
            #set_bn_eval(self.RCNN_base.stage1)
            #set_bn_eval(self.RCNN_base.stage2)
            #set_bn_eval(self.RCNN_base.stage3)


    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)  # or two large fully-connected layers

        return fc7

