#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset visdrone --net snet_146 \
       --cfg ./cfgs/snet_480.yml --checkepoch 6  --cuda \
        --image_dir pictures/visdrone_images/
