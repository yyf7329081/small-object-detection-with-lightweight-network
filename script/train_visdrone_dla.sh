#!/bin/bash
set -e
cd ..

CUDA_VISIBLE_DEVICES=1  python trainval_net.py --dataset visdrone --cfg ./cfgs/snet_dla_visdrone.yml \
     --net dla_34 --nw 8 --lr 1e-2   --epochs 200 --cuda  --bs 14  --lr_decay_step 150,170,190  \
     --use_tfboard  True --eval_interval 10  --pre ./weights/snetdla_coco_kernel_1/dla_34/coco/thundernet_epoch_69.pth \
     --r True --checkepoch 199
