import os
import numpy as np
import shutil
import csv

folder_root='/home/leo/Downloads/chamo/v2_material/train_div'
target_root='/home/leo/Downloads/chamo/v2_material/train_div_shuffle'

batch_size=10000
img_count=0
folder_count=0
cur_des_folder=''

while True:
    if img_count%batch_size==0:
        cur_des_folder=target_root+'/chamo_%05d/' % folder_count
        folder_count=folder_count+1
        os.system('mkdir '+cur_des_folder)
    img_count = img_count + 1
    folders = os.listdir(folder_root)
    if len(folders) == 0:
        break
    rand_int = np.random.randint(len(folders))
    folder_addr=folder_root+'/'+folders[rand_int]
    imgs = os.listdir(folder_addr)
    if len(imgs)==0:
        shutil.rmtree(folder_addr)
        continue
    rand_int = np.random.randint(len(imgs))
    img_addr=folder_addr+'/'+imgs[rand_int]
    shutil.move(img_addr, cur_des_folder+'/'+imgs[rand_int])