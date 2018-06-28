# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import shutil
import filecmp
from data_scraping.materil_name import *
import numpy as np

excel_path = '/home/leo/Downloads/chamo/v2_material/V1.1.0.0525_1156.xlsx'
output_path='/home/leo/Downloads/chamo/v3_material/m_all'
img_root='/home/leo/Downloads/1000_work_set/5_29/all/1156_dish'

total_img_count=0
cur_des_folder=''
folder_count=0

def get_materials(excel_path):
    material_list={}
    xls_data = get_data(excel_path)
    for sheet_n in xls_data.keys():
        for row in xls_data[sheet_n]:
            for i in [2,3,4,5]:
                if len(row)>i:
                    if row[i]in material_list.keys():
                        material_list[row[i]]=material_list[row[i]]+1
                    else:
                        material_list[row[i]]=1

    material_list= sorted(material_list.items(), key=lambda e: e[1], reverse=True)
    # print('totoal count: %d' % len(material_list))
    # count=0
    # for (key, value) in material_list:
    #     count=count+1
    #     if count>73:
    #         break
    #     print('%s: %d' % (key, value))
    return material_list[0:73]


def copy_certain_dish(source_addr, dish_id,dish_name, label_str, target_addr):
    global total_img_count
    global cur_des_folder
    global folder_count
    imgs_dir=source_addr+dish_id
    if os.path.exists(imgs_dir):
        imgs = os.listdir(imgs_dir)
        for i in range(len(imgs)):
            if total_img_count%10000==0:
                cur_des_folder = target_addr + 'chamo_%05d/' % folder_count
                folder_count = folder_count + 1
                os.makedirs(cur_des_folder)
            total_img_count = total_img_count + 1
            shutil.copyfile(imgs_dir + "/" + imgs[i], cur_des_folder + dish_name+'_'+label_str+'_'+str(i)+'.jpg')

def get_folders(material_list, excel_path, output_path, img_root):
    bit_len=len(material_list)
    xls_data = get_data(excel_path)
    for sheet_n in xls_data.keys():
        for row in xls_data[sheet_n]:
            labels=[0 for _ in range(bit_len)]
            for i in [2,3,4,5]:
                if len(row)>i:
                    if row[i] in material_list.keys():
                        labels[material_list[row[i]]]=1
            if (sum(labels)>0):
                name=row[1]
                dish_id=row[0]
                label_dec=0
                temp_count=bit_len
                for item in labels:
                    temp_count = temp_count - 1
                    label_dec=label_dec+item*pow(2,temp_count)
                copy_certain_dish(img_root,dish_id ,name, str(label_dec), output_path)

def get_material_dish(material_list, excel_path, output_path, img_root):
    material_dish_list = {}
    xls_data = get_data(excel_path)
    dish_list=[]
    for sheet_n in xls_data.keys():
        for row in xls_data[sheet_n]:
            for i in [2, 3, 4, 5]:
                if len(row) > i:
                    if row[i] in material_list.keys():
                        imgs_dir=img_root + '/' + row[0]
                        if os.path.exists(imgs_dir):
                            dish_list.append(row[0])
                            if row[i] in material_dish_list.keys():
                                material_dish_list[row[i]].append(row[0])
                            else:
                                material_dish_list[row[i]]=[row[0]]
    for material_name in material_dish_list.keys():
        os.system('mkdir ' + output_path+'/'+material_name)
        os.system('mkdir ' + output_path + '/' + material_name+'/positive')
        os.system('mkdir ' + output_path + '/' + material_name + '/negative')
    materials = list(material_list.keys())
    for material_name in material_dish_list.keys():
        print(material_name)
        for i in range(10000):
            rand_int = np.random.randint(len(material_dish_list[material_name]))
            folder_addr = img_root + '/' + material_dish_list[material_name][rand_int]
            imgs = os.listdir(folder_addr)
            rand_int = np.random.randint(len(imgs))
            img_addr = folder_addr + '/' + imgs[rand_int]
            shutil.copy(img_addr, output_path+'/'+material_name + '/positive/' + imgs[rand_int])
        for i in range(30000):
            rand_dis_id=None
            while True:
                rand_int = np.random.randint(len(dish_list))
                exist=False
                for m_id in material_dish_list[material_name]:
                    if dish_list[rand_int]==m_id:
                        exist=True
                        break
                if exist==False:
                    rand_dis_id=rand_int
                    break
            folder_addr = img_root + '/' + dish_list[rand_dis_id]
            imgs = os.listdir(folder_addr)
            rand_int = np.random.randint(len(imgs))
            img_addr = folder_addr + '/' + imgs[rand_int]
            shutil.copy(img_addr, output_path+'/'+material_name + '/negative/' + imgs[rand_int])

material_list = get_materials(excel_path)
print(material_list)
material_list_dict={}
count=0
for (key, value) in material_list:
    material_list_dict[key]=count
    count=count+1
#material_list_dict={'青椒':0, '木耳':1, '鸡蛋':2}
get_material_dish(material_list_dict,excel_path, output_path, img_root)
#get_folders(material_list_dict, excel_path, output_path, img_root)




