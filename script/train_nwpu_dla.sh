#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=1  python trainval_net.py --dataset nwpu --cfg ./cfgs/snet_dla_nwpu.yml \
     --net dla_34 --nw 8 --lr 1e-2   --epochs 500 --cuda  --bs 16  --lr_decay_step 450,470,490  \
     --use_tfboard  True --eval_interval 50  --pre ./weights/snetdla_coco_kernel_1/dla_34/coco/thundernet_epoch_69.pth \
     --r True  --checkepoch 199
