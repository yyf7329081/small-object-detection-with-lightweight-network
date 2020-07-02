#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset pascal_voc_0712 --cfg ./cfgs/snet.yml \
     --net snet_146 --nw 8 --lr 1e-2   --epochs 50 --cuda  --lr_decay_step 35,40,45  \
     --use_tfboard  True --eval_interval 5   \
