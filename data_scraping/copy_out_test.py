import os
import numpy as np
import shutil
import csv

folder_root='D:/1156_dish'
target_root='E:/data/青椒/false'
number_of_test=1000
for i in range(number_of_test):
    folders = os.listdir(folder_root)
    rand_int = np.random.randint(len(folders))
    folder_addr=folder_root+'/'+folders[rand_int]
    imgs = os.listdir(folder_addr)
    rand_int = np.random.randint(len(imgs))
    img_addr=folder_addr+'/'+imgs[rand_int]
    shutil.move(img_addr, target_root+'/'+imgs[rand_int])