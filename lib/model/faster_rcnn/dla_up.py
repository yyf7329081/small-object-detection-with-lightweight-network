import math

import numpy as np
import torch
from torch import nn

from .dla import dla34

#BatchNorm = nn.BatchNorm2d


#def set_bn(bn):
    #global BatchNorm
    #BatchNorm = bn
    #dla.BatchNorm = bn

dla_dict = {'dla34':dla34}

idaup_node_kernel_size = 1

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():# 权重随机初始化
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(idaup_node_kernel_size, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
        
        

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        out = [layers[-1]]
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
            out.insert(0, layers[-1])
            
        return out


class DLASeg(nn.Module):
    def __init__(self, base_name, pretrained_base=None, down_ratio=2):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.out_channel = 245
        self.first_level = int(np.log2(down_ratio))
        self.base = dla_dict[base_name](pretrained=pretrained_base,
                                            return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)
        #self.fc = nn.Sequential(
            #nn.Conv2d(channels[self.first_level], classes, kernel_size=1,
                      #stride=1, padding=0, bias=True)
        #)
        #up_factor = 2 ** self.first_level
        #if up_factor > 1:
            #up = nn.ConvTranspose2d(classes, classes, up_factor * 2,
                                    #stride=up_factor, padding=up_factor // 2,
                                    #output_padding=0, groups=classes,
                                    #bias=False)
            #fill_up_weights(up)
            #up.weight.requires_grad = False
        #else:
            #up = Identity()
        #self.up = up
        #self.softmax = nn.LogSoftmax(dim=1)
        
        self.glbavg = nn.AdaptiveAvgPool2d(1)
        self.last_level = len(channels)-1
        self.ida_up = IDAUp(idaup_node_kernel_size, channels[self.first_level], channels[self.first_level:self.last_level], 
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.conv_avg= nn.Conv2d(channels[-1], channels[self.first_level], kernel_size=1, bias=True)
        self.conv_last = nn.Conv2d(channels[self.first_level] , self.out_channel, kernel_size=1, bias=True)

        # fc权重随机化
        #for m in self.fc.modules():
            #if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
            #elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        ida_feat = x[-self.last_level-1+self.first_level:-1]
        if len(ida_feat)>1:
            ida_feat = self.ida_up(ida_feat)[0]
        output = self.conv_avg(self.glbavg(x[-1])) + ida_feat
        output = self.conv_last(output)
        #x = self.fc(x)
        #y = self.softmax(self.up(x))
        return output

    def optim_parameters(self, memo=None):
        for param in self.base.parameters():
            yield param
        for param in self.dla_up.parameters():
            yield param
        for param in self.glbavg.parameters():
            yield param
        for param in self.conv_avg.parameters():
            yield param        
        for param in self.conv_last.parameters():
            yield param        


def dla34up(pretrained_base=None, down_ratio=8):
    model = DLASeg('dla34', pretrained_base=pretrained_base, down_ratio=8)
    return model



