#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset car_3class --cfg ./cfgs/snet_dla.yml \
     --net dla_34 --nw 8 --lr 1e-2   --epochs 300 --cuda  --bs 8  --lr_decay_step 280,290,295  \
     --use_tfboard  True --eval_interval 10  --r True --checkepoch 149