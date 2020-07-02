#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset pascal_voc_0712 --net snet_146 \
       --cfg ./cfgs/snet.yml --checkepoch 8  --cuda \
        --image_dir pictures/voc_images/
