import os
import numpy as np
import shutil
root_dir='D:/1000_data/train_no_expand'
files= os.listdir(root_dir)
target_dir='D:/material_sort/non_califlower'
for i in range(500):
    rand_int = np.random.randint(len(files))
    imgs_dir = root_dir+"/"+files[rand_int]
    imgs = os.listdir(imgs_dir)
    rand_int = np.random.randint(len(imgs))
    shutil.copyfile(imgs_dir+"/"+imgs[rand_int], target_dir+"/"+imgs[rand_int])

