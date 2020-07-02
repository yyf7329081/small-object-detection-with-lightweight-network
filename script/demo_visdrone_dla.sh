#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python demo.py --dataset visdrone --net dla_34 \
       --cfg ./cfgs/snet_dla_visdrone.yml --checkepoch 199  --cuda \
        --image_dir pictures/Visdrone_nms/  --output_dir pictures_det/Visdrone_nms/
