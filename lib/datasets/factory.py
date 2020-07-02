# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.visdrone import visdrone
from datasets.visdrone_coco import visdrone_coco
from datasets.nwpu_coco import nwpu_coco
from datasets.vedai_coco import vedai_coco
from datasets.car_3class import car_3class


for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year, "/home/data/VOC/VOCdevkit"))

# Set up coco_2017_<split>

for year in ['2017']:
  for split in ['train', 'val']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year, "/home/data"))

for year in ['2018']:
  for split in ['train','val','test']:
    name = 'visdrone_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: visdrone_coco(split, year, "/home/data"))

for year in ['2018']:
  for split in ['train','val','test']:
    name = 'nwpu_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: nwpu_coco(split, year, "/home/data"))

for year in ['2018']:
  for split in ['train','val','test']:
    name = 'vedai_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: vedai_coco(split, year, "/home/data"))

for year in ['2019']:
  for split in ['trainset','valset','testset']:
    name = 'car_3class_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: car_3class(split, year, "/home/omnisky/yyf/dataset"))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
#
# for version in ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']:
#     for split in ['minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
#
# # set up image net.
# for split in ['train', 'val', 'val1', 'val2', 'test']:
#     name = 'imagenet_{}'.format(split)
#     devkit_path = 'data/imagenet/ILSVRC/devkit'
#     data_path = 'data/imagenet/ILSVRC'
#     __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(split,devkit_path,data_path))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  # if name not in __sets:
  #   raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
