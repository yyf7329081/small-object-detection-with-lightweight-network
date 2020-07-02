#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=1  python demo.py --dataset nwpu --net dla_34 \
       --cfg ./cfgs/snet_dla_nwpu.yml --checkepoch 349  --cuda \
        --image_dir pictures/NWPU/  --output_dir pictures_det/NWPU/
