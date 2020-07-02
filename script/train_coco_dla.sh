#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset coco --cfg ./cfgs/snet_dla.yml \
     --net dla_34 --nw 8 --lr 1e-2   --epochs 100 --cuda  --bs 16  --lr_decay_step 80,90,95  \
     --use_tfboard  True --eval_interval 5  --r True  --checkepoch 0