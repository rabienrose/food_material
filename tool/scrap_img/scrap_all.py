import os
import data_convertor.process_img
import tool.server.update_train_imgs
import data_scraping.materil_name
import shutil
import tool.scrap_img.scrap_img
def create_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)

def copy_files(src, db):
    files = os.listdir(src)
    for file in files:
        shutil.copy(src+'/'+file, db+'/'+file)

material_candi=data_scraping.materil_name.material_candi
temp_root='/home/leo/Downloads/chamo/scrap_all/temp'
all_root='/home/leo/Downloads/chamo/scrap_all/all'
sub_root='/home/leo/Downloads/chamo/scrap_all/sub'
create_dir(all_root)
create_dir(sub_root)
for i in range(len(material_candi)):
    for j in range(i+1, len(material_candi)):
        create_dir(temp_root)
        mat1=material_candi[i]
        mat2 = material_candi[j]
        key_word_c = mat1 + mat2
        tool.scrap_img.scrap_img.scrap(key_word_c, temp_root, int(200))
        data_convertor.process_img.checkFormat(temp_root, 6)
        data_convertor.process_img.checkChannel(temp_root, 6)
        folder_name = mat1 + '_' + mat2
        os.mkdir(sub_root + '/' + folder_name)
        copy_files(temp_root, sub_root + '/' + folder_name)
        files = os.listdir(all_root)
        if len(files) > 0:
            tool.server.update_train_imgs.check_sim(temp_root, all_root)
        tool.server.update_train_imgs.copy_files(temp_root, all_root)


