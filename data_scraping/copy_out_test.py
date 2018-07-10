import os
import numpy as np
import shutil
import csv
folder_root='E:/work/code/transfer/wawacai'
target_root='E:/work/code/transfer/wawacai_f'
pre_name='chamo'

def rename_files():
    folders = os.listdir(folder_root)
    i = 1000000
    for item in folders:
        folder_addr = folder_root + '/' + item
        dst_folder_addr = folder_root + '/' + pre_name+'_'+str(i)+'.jpg'
        os.rename(folder_addr, dst_folder_addr)
        i=i+1

#rename_files()
number_of_test=1000
for i in range(number_of_test):
    #folders = os.listdir(folder_root)
    #rand_int = np.random.randint(len(folders))
    #folder_addr=folder_root+'/'+folders[rand_int]
    folder_addr=folder_root
    imgs = os.listdir(folder_addr)
    rand_int = np.random.randint(len(imgs))
    img_addr=folder_addr+'/'+imgs[rand_int]
    shutil.move(img_addr, target_root+'/'+imgs[rand_int])