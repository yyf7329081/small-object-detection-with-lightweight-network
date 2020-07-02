#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=1  python trainval_net.py --dataset vedai --cfg ./cfgs/snet_dla_vedai.yml \
     --net dla_34 --nw 8 --lr 1e-3   --epochs 300 --cuda  --bs 16  --lr_decay_step 250,270,280  \
     --use_tfboard  True --eval_interval 10  --pre ./weights/snetdla_coco_kernel_1/dla_34/coco/thundernet_epoch_69.pth \
     --r True  --checkepoch 149
