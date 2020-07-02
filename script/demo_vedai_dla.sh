#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=1  python demo.py --dataset vedai --net dla_34 \
       --cfg ./cfgs/snet_dla_vedai.yml --checkepoch 299  --cuda \
        --image_dir pictures/VEDAI/  --output_dir pictures_det/VEDAI/
