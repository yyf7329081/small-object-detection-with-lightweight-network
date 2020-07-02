from .modules import *

from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.utils.config import cfg

import math
from os.path import join

import torch
from torch import nn

#BatchNorm = nn.BatchNorm2d

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]
        if stride == 2:
            self.conv1 = ShuffleV2Block(inplanes, planes, mid_channels=planes // 2, ksize=5, stride=stride)
        elif stride == 1:
            self.conv1 = ShuffleV2Block(inplanes//2, planes, mid_channels=planes // 2, ksize=5, stride=stride)
        self.conv2 = ShuffleV2Block(planes // 2, planes, mid_channels=planes // 2, ksize=5, stride=1)            
        #self.conv1 = ShuffleV2Block(inplanes, planes, mid_channels=planes // 2, ksize=5, stride=stride)
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               #stride=stride, padding=dilation,
                               #bias=False, dilation=dilation)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = ShuffleV2Block(planes // 2, out_channels, mid_channels=out_channels // 2, ksize=5, stride=1)
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               #stride=1, padding=dilation,
                               #bias=False, dilation=dilation)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        #out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x

class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        #self.base_layer = nn.Sequential(
            #nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      #padding=3, bias=False),
            #BatchNorm(channels[0]),
            #nn.ReLU(inplace=True))        
        self.base_layer = nn.Sequential(nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, groups=3),
                                         nn.BatchNorm2d(3),
                                         nn.ReLU(inplace=True))
        self.level0 = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=1), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))     
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        #self.level0 = self._make_conv_level(
            #channels[0], channels[0], levels[0])
        #self.level1 = self._make_conv_level(
            #channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        #self.avgpool = nn.AvgPool2d(pool_size)
        #self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            #stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    #def _initialize_weights(self):

        #def set_bn_fix(m):
            #classname = m.__class__.__name__
            #if classname.find('BatchNorm') != -1:
                #for p in m.parameters(): p.requires_grad = False

        #if  self.model_path is not None:# 如果有预训练权重，就固定conv1,stage1,所有bn层参数。

            #print("Loading pretrained weights from %s" % (self.model_path))
            #if torch.cuda.is_available():
                #state_dict = torch.load(self.model_path)["state_dict"]
            #else:
                #state_dict = torch.load(
                    #self.model_path, map_location=lambda storage, loc: storage)["state_dict"]
            #keys = []
            #for k, v in state_dict.items():
                #keys.append(k)
            #for k in keys:
                #state_dict[k.replace("module.", "")] = state_dict.pop(k)

            #self.load_state_dict(state_dict,strict = False)

        #else:# 如果没有预训练权重，就全部随机初始化。
            #for name, m in self.named_modules():
                #if isinstance(m, nn.Conv2d):
                    #if 'first' in name:
                        #nn.init.normal_(m.weight, 0, 0.01)
                    #else:
                        #nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                    #if m.bias is not None:
                        #nn.init.constant_(m.bias, 0)
                #elif isinstance(m, nn.BatchNorm2d):
                    #nn.init.constant_(m.weight, 1)
                    #if m.bias is not None:
                        #nn.init.constant_(m.bias, 0.0001)
                    #nn.init.constant_(m.running_mean, 0)
                #elif isinstance(m, nn.BatchNorm1d):
                    #nn.init.constant_(m.weight, 1)
                    #if m.bias is not None:
                        #nn.init.constant_(m.bias, 0.0001)
                    #nn.init.constant_(m.running_mean, 0)
                #elif isinstance(m, nn.Linear):
                    #nn.init.normal_(m.weight, 0, 0.01)
                    #if m.bias is not None:
                        #nn.init.constant_(m.bias, 0)

def dla34(pretrained=None, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    #if pretrained is not None:
        #model.load_pretrained_model(pretrained, 'dla34')
    return model
