import os
import shutil
img_root=u"/home/leo/Downloads/chamo/train_merge/"
des_img_root=u"/home/leo/Downloads/chamo/train_div/"
imgs = os.listdir(img_root)
batch_size=10000
img_count=0
folder_count=0
cur_des_folder=''
#os.system('rm '+cur_des_folder)
#os.system('mkdir '+cur_des_folder)
print(len(imgs))
for img in imgs:
    if img_count%batch_size==0:
        cur_des_folder=des_img_root+'chamo_%05d/' % folder_count
        folder_count=folder_count+1
        os.system('mkdir '+cur_des_folder)
    img_count=img_count+1
    os.system('mv '+img_root+img+' '+cur_des_folder+img)
    #shutil.copy(img_root+img, cur_des_folder+img)

