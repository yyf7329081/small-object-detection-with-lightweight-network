#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=0  python trainval_net.py --dataset car_3class --cfg ./cfgs/snet_ls.yml \
     --net snet_146 --nw 8 --lr 1e-2   --epochs 500 --cuda  --bs 32  --lr_decay_step 480,490,495  \
     --use_tfboard  True --eval_interval 10   --pre ./weights/pretrained/snet_146.tar  \
     --r True --checkepoch 299