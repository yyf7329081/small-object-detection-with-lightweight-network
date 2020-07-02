# 3.数据集的整理和划分
import os
import glob
import random

dataset_dir = '/home/data/VEDAI'
img_files = glob.glob(os.path.join(dataset_dir,'image_set','*.jpg'))# 找到匹配的文件列表
img_num = len(img_files)
totalset_txt = os.path.join(dataset_dir,'totalset.txt')# 所有文件列表的txt文件
trainset_txt = os.path.join(dataset_dir,'trainset.txt')# 训练集文件的txt文件
valset_txt = os.path.join(dataset_dir,'valset.txt')# 验证集文件的txt文件
testset_txt = os.path.join(dataset_dir,'testset.txt')# 测试集文件的txt文件

# 新建所有文件列表的txt文件
f_total = open(totalset_txt,'w',encoding='utf-8')
for i, file in enumerate(img_files):
    filename = os.path.splitext(os.path.split(file)[1])[0]
    #strline = os.path.join(dataset_dir,filename)
    f_total.write(filename)
    f_total.write('\n')
    if i%100 == 0:
        print('%d/%d'%(i,img_num))
f_total.close()
print('totalset complete!')

# 拆分成训练集、验证集、测试集
with open(totalset_txt, 'r', encoding='utf-8') as f:
    data_list = f.readlines()
data_list = [_.strip('\n') for _ in data_list]
total_num = len(data_list)
print(total_num)
pick_for_train = list(range(total_num))

# 先选验证集的样本编号
val_num = int(total_num*0.1)
pick_for_val = random.sample(range(0,total_num),val_num)
for ele in pick_for_val:
    pick_for_train.remove(ele)
print(len(pick_for_train))

# 从剩下的里面选择测试集的样本编号，剩下的都作为训练集
test_num = int(total_num*0.1)
pick_for_test_ind = random.sample(range(0,total_num-val_num),test_num)
pick_for_test = []
for ind in pick_for_test_ind:
    pick_for_test.append(pick_for_train[ind])
for ele in pick_for_test:
    pick_for_train.remove(ele)
print(len(pick_for_train))

# 训练集、验证集和测试集的样本名称都写到对应的txt文件中
with open(valset_txt, 'wt', encoding='utf-8') as f:
    for val_ind in pick_for_val:
        f.write(data_list[val_ind] + '\n')

with open(testset_txt, 'wt', encoding='utf-8') as f:
    for test_ind in pick_for_test:
        f.write(data_list[test_ind] + '\n')

with open(trainset_txt, 'wt', encoding='utf-8') as f:
    for train_ind in pick_for_train:
        f.write(data_list[train_ind] + '\n')

print('all complete!')