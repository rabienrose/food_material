import os
import numpy as np
import shutil
import csv
folder_root='/home/leo/Documents/chamo/transfer/tool/img'
target_root='/home/leo/Documents/chamo/transfer/tool/img/花菜'
pre_name='chamo'
test_p_count=100
test_n_count=200
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
os.mkdir(target_root+'/test')
os.mkdir(target_root+'/test/positive')
os.mkdir(target_root+'/test/negative')
os.mkdir(target_root+'/train')
#os.mkdir(target_root+'/train/positive')
#os.mkdir(target_root+'/train/negative')

posi_dir=folder_root+'/positive'
nega_dir=folder_root+'/negative'

for i in range(test_p_count):
    imgs = os.listdir(posi_dir)
    rand_int = np.random.randint(len(imgs))
    img_addr=posi_dir+'/'+imgs[rand_int]
    shutil.move(img_addr, target_root+'/test/positive/'+imgs[rand_int])
for i in range(test_n_count):
    imgs = os.listdir(nega_dir)
    rand_int = np.random.randint(len(imgs))
    img_addr = nega_dir + '/' + imgs[rand_int]
    shutil.move(img_addr, target_root + '/test/negative/' + imgs[rand_int])

shutil.move(posi_dir, target_root+'/train')
shutil.move(nega_dir, target_root+'/train')