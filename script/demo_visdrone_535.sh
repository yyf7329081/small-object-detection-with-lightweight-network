#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset visdrone --net snet_535 \
       --cfg ./cfgs/snet_535.yml --checkepoch 50  --cuda \
        --image_dir pictures/visdrone_images/
