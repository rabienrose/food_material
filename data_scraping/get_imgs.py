import os
import numpy as np
import shutil
import csv

number_of_raw=20
material_list=[['花菜'],['虾仁'],['木耳'],['番茄','西红柿'],['鸡蛋']]
raw_data_xls='E:/raw_dish_data.csv'
target_addr='D:/material_sort_all'
source_addr='D:/1000_data/train_no_expand'
exclude_dishs=['虎皮鸡蛋','清炒木耳菜']

csv_reader = csv.reader(open(raw_data_xls, encoding='utf-8'))
choosed_img=[]
raw_data={}
for row in csv_reader:
    raw_data[row[0]]=row[1]

def copy_certain_dish(dish_id, dish_name, label_str):
    imgs_dir=source_addr+'/'+dish_id
    count=0
    if os.path.exists(imgs_dir):
        imgs = os.listdir(imgs_dir)
        for i in range(number_of_raw):
            count=count+1
            rand_int = np.random.randint(len(imgs))
            shutil.copyfile(imgs_dir + "/" + imgs[rand_int], target_addr + "/" + dish_name+'_'+label_str+'_'+str(count)+'.jpg')

for key in raw_data:
    excluded=False
    for exclude in exclude_dishs:
        if raw_data[key] ==exclude:
            excluded=True
            break
    if excluded:
        break
    material_flag=[0 for n in range(len(material_list))]
    for i in range(len(material_list)):
        candi_name=material_list[i]
        for name in candi_name:
            if raw_data[key].find(name) != -1:
                material_flag[i]=1
        material_flag_np=np.array(material_flag)
    if np.sum(material_flag_np)!=0:
        label_str=''
        for item in material_flag:
            label_str=label_str+str(item)
        #print('%s %s' % (raw_data[key], label_str))
        copy_certain_dish(key, raw_data[key], label_str)


