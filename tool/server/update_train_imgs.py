import subprocess
import os
import shutil

def check_sim(query, db):
    if os.path.exists("query_list.txt"):
        os.remove("query_list.txt")
    if os.path.exists("img_list.txt"):
        os.remove("img_list.txt")
    os.system('../img_sim_check/script/check_sim.sh ' + query +' '+ db)

def copy_files(src, db):
    files = os.listdir(src)
    for file in files:
        shutil.move(src+'/'+file, db+'/'+file)

train_temp='/home/leo/Downloads/chamo/mat_service/train_temp'
train_root='/home/leo/Downloads/chamo/mat_service/train'
upload_m = os.listdir(train_temp)
db_m = os.listdir(train_root)
for mat in upload_m:
    has_m=False
    for mat1 in db_m:
        if mat==mat1:
            has_m=True
            break
    if has_m==True:
        check_sim(train_root + '/' + mat + '/positive', train_temp + '/' + mat + '/positive')
        check_sim(train_root + '/' + mat + '/negative', train_temp + '/' + mat + '/negative')
    else:
        os.mkdir(train_root+'/'+mat)
        os.mkdir(train_root + '/' + mat+'/positive')
        os.mkdir(train_root + '/' + mat + '/negative')
    copy_files(train_temp+'/'+mat+'/positive', train_root+'/'+mat+'/positive')
    copy_files(train_temp + '/' + mat + '/negative', train_root + '/' + mat + '/negative')
shutil.rmtree(train_temp)



