#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset car_3class --net dla_34 \
       --cfg ./cfgs/snet_dla.yml --checkepoch 149  --cuda \
        --image_dir pictures/car_3class/
