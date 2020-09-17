# small object detection with lightweight network
## Introduction and Reference
This project uses ThunderNet as detection framework, DlaNet as backbone network, ShuffleNetV2 block as lightweight module, papers and codes i use for reference are shown below.

ThunderNet [paper](https://arxiv.org/pdf/1903.11752.pdf)   [github](https://github.com/ouyanghuiyu/Thundernet_Pytorch)

DlaNet [paper](https://arxiv.org/abs/1707.06484)   [github](https://github.com/ucbdrive/dla)

ShuffleNetV2 [paper](https://arxiv.org/abs/1807.11164)

This project gets better performance when facing small objects because DlaNet fuses features better and gets larger feature map. At the same time, its computation complexity and run time decrease greatly as lightweight framework and convolution module.

## Environment
Ubuntu16.04
GTX 1080 Ti
Pytorch 1.0
cuda 9.0


## How to use
This project is mainly based on ThunderNet-ouyanghuiyu [github](https://github.com/ouyanghuiyu/Thundernet_Pytorch), so you can view the webpage above to use this project as a whole.
By the way, A related blog will show you how to use this project to train and test in detail. It will be finished soon...

## chinese blog
[blog](https://blog.csdn.net/u014796085/article/details/108653377) here introduce the project in detail and show you the result pictures.
