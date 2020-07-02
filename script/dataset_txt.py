# 数据集读取文件生成txt
import os

def read2txt(data_dir, txt_path):
    print(data_dir)
    print(txt_path)
    count = 0
    f = open(txt_path,'w',encoding='utf-8')
    file_list = os.listdir(os.path.join(data_dir,'images'))
    file_num = len(file_list)
    for filename in file_list:
        filename,extname = os.path.splitext(os.path.split(filename)[-1])
        if extname == '.jpg':
            f.write(filename)
            f.write('\n')
            count += 1
        if count%100 == 0:
            print('%d/%d'%(count,file_num))
    print(txt_path + ':' + '%d'%count)
    f.close()

dataset_dir = '/home/data/VisDrone2018'
trainset_dir = os.path.join(dataset_dir, 'VisDrone2018-DET-train')
valset_dir = os.path.join(dataset_dir, 'VisDrone2018-DET-val')
testset_dir = os.path.join(dataset_dir, 'VisDrone2018-DET-test-dev')

#txt_dir = os.path.join(dataset_dir, 'ImageSets')
#if not os.path.exists(txt_dir):
    #os.makedirs(txt_dir)

trainset_path = os.path.join(trainset_dir, 'trainset.txt')
valset_path = os.path.join(valset_dir, 'valset.txt')
testset_path = os.path.join(testset_dir, 'testset.txt')

read2txt(trainset_dir, trainset_path)
read2txt(valset_dir, valset_path)
read2txt(testset_dir, testset_path)


