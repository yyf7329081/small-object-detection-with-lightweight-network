#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset car_3class --net snet_146 \
       --cfg ./cfgs/snet_ls.yml --checkepoch 299  --cuda \
        --image_dir pictures/car_3class/
