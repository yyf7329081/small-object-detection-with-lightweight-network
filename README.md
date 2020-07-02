# small object detection with lightweight network
## Introduction and Reference
This project uses ThunderNet as detection framework, DlaNet as backbone network, ShuffleNetV2 block as lightweight module, papers and codes i use for reference are shown below.

ThunderNet [paper](https://arxiv.org/pdf/1903.11752.pdf)   [github](https://github.com/ouyanghuiyu/Thundernet_Pytorch)

DlaNet [paper](https://arxiv.org/abs/1707.06484)   [github](https://github.com/ucbdrive/dla)

ShuffleNetV2 [paper](https://arxiv.org/abs/1807.11164)

This project gets better performance when facing small objects because DlaNet fuses features better and gets larger feature map. At the same time, its computation complexity and run time decrease greatly as lightweight framework and convolution module.
![avatar](F:\硕士\毕业论文\毕业论文图片材料\visdrone_show\9999947_00000_d_0000012_det.jpg)

精度对比
![avatar](F:\github_repositories\upload\精度对比.png)

速度对比
![avatar](F:\github_repositories\upload\速度对比.png)
