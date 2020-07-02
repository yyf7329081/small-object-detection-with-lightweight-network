#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset car_3class --cfg ./cfgs/snet_535.yml \
    --net snet_535 --nw 8 --lr 1e-2  --epochs 100 --cuda  --lr_decay_step 80,90,95  \
    --use_tfboard  True --eval_interval 5   \
