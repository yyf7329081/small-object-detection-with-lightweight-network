# 创建coco格式的json标注文件
import json
import os
import cv2
import collections

def create_dataset(dataset_txt, image_id, bnd_id):
    images = []
    annotations = []
    # category_map = {'background':0, 'vehicle':1}
    category_map = collections.OrderedDict([('pedestrian', 1),
                    ('people', 2),
                    ('bicycle', 3),
                    ('car', 4),
                    ('van', 5),
                    ('truck', 6),
                    ('tricycle', 7),
                    ('awning-tricycle', 8),
                    ('bus', 9),
                    ('motor', 10)
                    ])
    dataset_dir = os.path.abspath(os.path.join(dataset_txt, '..'))
    jpg_dir = os.path.join(dataset_dir, 'images')
    txt_dir = os.path.join(dataset_dir, 'annotations')
    with open(dataset_txt, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            filename = line.strip('\n')        
            jpg_path = os.path.join(jpg_dir, filename + '.jpg')
            txt_path = os.path.join(txt_dir, filename + '.txt')
            
            image_info = dict()
            image_id += 1
            if image_id%100==0:
                print(image_id)
            img = cv2.imread(jpg_path)
            img_height = img.shape[0]
            img_width = img.shape[1]
            image_info['file_name'] = filename + '.jpg'
            image_info['height'] = img_height
            image_info['width'] = img_width
            image_info['id'] = image_id            
            
            with open(txt_path, 'r', encoding='utf-8') as new_f:
                new_lines = new_f.readlines()
                for index, new_line in enumerate(new_lines):
                    data = new_line.strip()
                    data = new_line.split(',')
                    assert(len(data)==8), "wrong anno, file{}, line:{}".format(line,index)
                    category_id = int(data[5])
                    assert(category_id>=0 and category_id<12),"unknow cate, file:{}, line:{}".format(line,index)
                    if category_id == 0 or category_id == 11:
                        continue
                    xmin = int(data[0])
                    ymin = int(data[1])
                    width = int(data[2])
                    height = int(data[3])
                    iscrowd = int(data[-1])
                    assert(xmin>=0 and ymin>=0 and width>=0 and height>=0), "<0?, {}, {}".format(line,xmin)
                    if width==0 or height==0:
                        continue
                    bnd_id += 1
                    assert(xmin+width<=img_width and ymin+height<=img_height), "out of boundary, {}, {}".format(line,xmin)
                    area = int(width*height)
                    ann = {'area': area, 'iscrowd': iscrowd, 'image_id':
                           image_id, 'bbox':[xmin, ymin, width, height],
                           'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                           'segmentation': []}
                    annotations.append(ann)
            images.append(image_info)
    categories = []
    for cate, cid in category_map.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        categories.append(cat)
    return images,annotations,categories,image_id,bnd_id

def create_json_file(images, annotations, categories, json_file_name):
    json_data = {"images": [], "type": "instances", "annotations": [], "categories": []}
    json_data['images'] = images
    json_data['annotations'] = annotations
    json_data['categories'] = categories
    with open(json_file_name, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(json_data))

trainset_txt = '/home/data/VisDrone2018/VisDrone2018-DET-train/trainset.txt'
valset_txt = '/home/data/VisDrone2018/VisDrone2018-DET-val/valset.txt'
testset_txt = '/home/data/VisDrone2018/VisDrone2018-DET-test-dev/testset.txt'

img_id_start = 0
bbox_id_start = 0
print('trainset start img_id:%d bbox_id:%d'%(img_id_start,bbox_id_start))
trainset_images, trainset_annotations, trainset_categories, img_id_start, bbox_id_start = create_dataset(trainset_txt, img_id_start, bbox_id_start)
print(trainset_categories)
print('valset start img_id:%d bbox_id:%d'%(img_id_start,bbox_id_start))
valset_images, valset_annotations, valset_categories, img_id_start, bbox_id_start = create_dataset(valset_txt, img_id_start, bbox_id_start)
print(valset_categories)
print('testset start img_id:%d bbox_id:%d'%(img_id_start,bbox_id_start))
testset_images, testset_annotations, testset_categories, img_id_start, bbox_id_start = create_dataset(testset_txt, img_id_start, bbox_id_start)
print(testset_categories)
print('total end img_id:%d bbox_id:%d'%(img_id_start,bbox_id_start))

anno_dir = os.path.abspath(os.path.join(trainset_txt, '..', '..'))
anno_dir = os.path.join(anno_dir, 'annotations')
if not os.path.exists(anno_dir):
    os.makedirs(anno_dir)
train_anno = os.path.join(anno_dir, 'instances_train2018.json')
val_anno = os.path.join(anno_dir, 'instances_val2018.json')
test_anno = os.path.join(anno_dir, 'instances_test2018.json')
create_json_file(trainset_images, trainset_annotations, trainset_categories, train_anno)
create_json_file(valset_images, valset_annotations, valset_categories, val_anno)
create_json_file(testset_images, testset_annotations, testset_categories, test_anno)


print('complete!')