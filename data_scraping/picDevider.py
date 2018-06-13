# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import shutil
import filecmp
import utils.logger as logger
from data_scraping.materil_name import *


TAG = __name__


excel_path = 'E:/data/v1.0.0_0507.xlsx'

pic_src_path = 'E:/1000_data/train_no_expand/'

pic_des_path = 'E:/material_merge/'

pic_merge_path_train = 'E:/material_merge/'

pic_merge_path_test = 'E:\\only200_test'




materials = [[u'青椒'], [u'番茄', u'西红柿'], [u'鸡蛋'], [u'木耳']]

SUF_LABEL = 'labelmat'

PIC_NUM = 0

# get a set of materials's number
def get_mat_num(materials, excel_path):
    '''
    通过食材名称获取该食材在excel下的所有对应编号
    :param materials:食材数组
    :param excel_path:excel文件位置
    :return:食材名对应文件夹编号的字典
    '''
    ret = {}
    xls_data = get_data(excel_path)
    for sheet_n in xls_data.keys():
        for row in xls_data[sheet_n]:
            for material in materials:
                for sub_mat in material:
                    if sub_mat in row[1]:
                        if not material[0] in ret.keys():
                            ret[material[0]] = []
                        ret[material[0]].append([row[0], row[1].strip()])
        break
    return ret


def create_and_copy(mat_dict, src_path, des_path):
    '''
    根据食材名与它的文件夹编号列表来将某食材的文件全部转存到另一个文件夹下，并将编号替换为菜名
    :param mat_dict:食材对应文件夹编号的字典
    :param src_path:源根路径
    :param des_path:图片存储目标路径
    :return:
    '''
    if not os.path.exists(src_path):
        return
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    for material in mat_dict:
        m_path = des_path + '/' + material
        if not os.path.exists(m_path):
            os.mkdir(m_path)
        for num, name in mat_dict[material]:
            num_src = src_path + '/'  + num
            num_des = des_path + '/' + material + '/' + name
            print('past: ', num_src)
            print('to: ', num_des)
            if not os.path.exists(num_src):
                print(num_src + '路径不存在')
                continue
            if os.path.exists(num_des):
                shutil.rmtree(num_des)
                #os.mkdir(num_des)
            try:
                shutil.copytree(num_src, num_des)
            except:
                continue


# TODO implement this func
def merge_all(src_path, des_path, mat_array):
    '''
    把通过食材分类的菜品进行标记并整合到目标文件夹
    :param src_path:
    :param des_path:
    :param mat_array: 食材的列表
    :return:
    '''
    if not os.path.exists(src_path):
        return
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    mat_list = os.listdir(src_path)

    array_len = len(mat_array)

    for mat_path in mat_list:
        # 得到食材的名称
        mat_name = mat_path

        mat_num = get_matarray_num(mat_array, mat_name)

        # 转换为完整路径
        mat_path = os.path.join(src_path, mat_path)
        num_list = os.listdir(mat_path)
        for veg_path in num_list:
            # 得到具体的菜品
            veg_name = veg_path
            # 得到具体的食材的菜品的文件夹路径
            veg_path = os.path.join(mat_path, veg_path)
            print('处理路径：', veg_path)
            # for root, dirs, files in os.walk(veg_path):
            #     for file in files:
            #         file = os.path.join(root, file)
            #         shutil.copy(file, des_path)
            for root, dirs, files in os.walk(veg_path):
                for file in files:
                    src_file_path = os.path.join(root, file)
                    des_file_path = os.path.join(des_path, file)
                    shutil.copyfile(src_file_path, des_file_path)
                    if merge_name(des_file_path, veg_name, mat_num, array_len) is None:
                        print('标记失败', des_file_path)


def merge_num(src_path, des_path, mat_array, pic_num):
    '''
    把通过食材分类的菜品进行文件名格式的调整并整合到同一个文件夹
    :param src_path:
    :param des_path:
    :param mat_array: 食材的列表
    :param pic_num: 每个食材需要的图片数量
    :return:
    '''
    if not os.path.exists(src_path):
        return
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    mat_list = os.listdir(src_path)
    for mat_path in mat_list:
        # 得到食材的名称
        mat_name = mat_path
        # 转换为完整路径
        mat_path = os.path.join(src_path, mat_path)
        num_list = os.listdir(mat_path)
        # 计算每个菜品需要取的数量
        num_avg = pic_num/len(num_list)
        # 记录剩余的菜品数量
        num_left = pic_num
        for veg_path in num_list:
            # 如果是最后一个文件夹，就把最后剩下的图都取了
            if num_list.index(veg_path) == len(num_list):
                num_avg = num_left
            # 得到具体的菜品
            veg_name = veg_path
            # 得到具体的食材的菜品的文件夹路径
            veg_path = os.path.join(mat_path, veg_path)
            # 每个文件夹当前的图片获取数
            num_cur = 0
            for root, dirs, files in os.walk(veg_path):
                for file in files:
                    src_file_path = os.path.join(root, file)
                    des_file_path = os.path.join(des_path, file)
                    file_existed, pic_des_path = is_file_exist(src_file_path, des_path)
                    if not file_existed:
                        shutil.copyfile(src_file_path, des_file_path)
                        merge_name(des_file_path, veg_name, get_matarray_num(mat_array, mat_name), len(mat_array))
                    else:
                        continue
                    num_cur += 1
                    num_left -= 1
                    # 如果本文件夹取的数量大于了本文件夹的设定数量或者本文件夹的最大图片数，则退出本文件夹
                    if num_cur >= num_avg or num_cur >= len(files):
                        break


def merge_num_test(src_path, des_path, train_path, mat_array, pic_num):
    '''
    把通过食材分类的菜品进行文件名格式的调整并整合到同一个文件夹，此代码为获取测试集，多一步判断训练集文件夹是否包含该
    图片
    :param src_path:
    :param des_path:
    :param des_path: 训练集文件夹位置
    :param mat_array: 食材的列表
    :param pic_num: 每个食材需要的图片数量
    :return:
    '''
    if not os.path.exists(src_path):
        return
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    mat_list = os.listdir(src_path)
    for mat_path in mat_list:
        # 得到食材的名称
        mat_name = mat_path
        # 转换为完整路径
        mat_path = os.path.join(src_path, mat_path)
        num_list = os.listdir(mat_path)
        # 计算每个菜品需要取的数量
        num_avg = pic_num/len(num_list)
        # 记录剩余的菜品数量
        num_left = pic_num
        for veg_path in num_list:
            # 如果是最后一个文件夹，就把最后剩下的图都取了
            if num_list.index(veg_path) == len(num_list):
                num_avg = num_left
            # 得到具体的菜品
            veg_name = veg_path
            # 得到具体的食材的菜品的文件夹路径
            veg_path = os.path.join(mat_path, veg_path)
            print('处理路径：', veg_path)
            # 每个文件夹当前的图片获取数
            num_cur = 0
            for root, dirs, files in os.walk(veg_path):
                for file in files:
                    src_file_path = os.path.join(root, file)
                    des_file_path = os.path.join(des_path, file)
                    file_existed, pic_des_path = is_file_exist(src_file_path, des_path)
                    train_existed, t = is_file_exist(src_file_path, train_path)
                    if file_existed or train_existed:
                        continue
                    shutil.copyfile(src_file_path, des_file_path)
                    merge_name(des_file_path, veg_name, get_matarray_num(mat_array, mat_name), len(mat_array))
                    num_cur += 1
                    num_left -= 1
                    # 如果本文件夹取的数量大于了本文件夹的设定数量或者本文件夹的最大图片数，则退出本文件夹
                    if num_cur >= num_avg or num_cur >= len(files):
                        break


def merge_name(file_path, name, num, length):
    '''
    对图片的名称进行修改，以符合图片标签的格式
    格式：'菜名_二进制食材编码_唯一码_是否已经标记'
    :param file_path: 图片文件全路径
    :param name: 菜品的名称
    :param num: 将第几位置位为1,方向为按位从高到低
    :param length: 编码的总长度
    :return:
    '''
    global PIC_NUM
    if num > length:
        return
    if not os.path.exists(file_path):
        return
    file_path_name = os.path.basename(file_path)
    file_path_basename = os.path.splitext(file_path_name)[0]
    if not file_path_basename.endswith(SUF_LABEL):
        des_name = name + '_' + str(pow(2, length - num - 1)) + '_' + str(PIC_NUM) + '_' + SUF_LABEL + '.jpg'
        try:
            ret_name = os.path.join(os.path.dirname(file_path), des_name)
            os.rename(file_path, ret_name)
        except:
            return None

        PIC_NUM += 1
        return ret_name
    else:
        num_start = file_path_basename.find('_')
        num_end = file_path_basename.find('_', num_start + 1)
        past_num = file_path_basename[num_start + 1:num_end]
        past_num_i = int(past_num)
        if int(past_num_i & pow(2, length - num - 1)) == 0:
            past_num_i = past_num_i | pow(2, length - num - 1)
            print('past_num_i changed', past_num_i, file_path)
        else:
            return file_path
        des_name = file_path_basename[0:num_start + 1] \
                   + str(past_num_i) \
                   + file_path_basename[num_end:len(file_path_basename)] + '.jpg'
        ret_name = os.path.join(os.path.dirname(file_path), des_name)
        os.rename(file_path, ret_name)
        return ret_name


def tag_path(pic_path, mat_array):
    '''
    遍历某文件夹并对其中的每一个图片进行标记
    :param pic_path: 图片所在的文件夹
    :param mat_array: 食材的类别
    :return:
    '''
    for root, dirs, files in os.walk(pic_path):
        for file in files:
            file_path = os.path.join(root, file)
            for mat in mat_array:
                for sub_mat in mat:
                    if sub_mat in file:
                        file_path = merge_name(file_path, 'test', get_matarray_num(mat_array, sub_mat), len(mat_array))


# 判断该文件夹下是否包含与file相同的文件
def is_file_exist(file, des_path):
    for root, dirs, files in os.walk(des_path):
        for f in files:
            f_path = os.path.join(root, f)
            if is_same_file(f_path, file):
                return True, f_path
    return False, None


def get_matarray_num(mat_array, mat_name):
    '''
    根据食材的名称获取食材的编号
    :param mat_array: 食材数组
    :param mat_name: 食材的名称
    :return: 食材的编号
    '''
    for i in mat_array:
        if mat_name in i:
            return mat_array.index(i)
    return None


# 比较两个文件是否相同
def is_same_file(file_l, file_r):
    return filecmp.cmp(file_l, file_r)


def split_img(img_root, des_img_root, each_folder_num):
    imgs = os.listdir(img_root)
    img_count = 0
    folder_count = 0
    cur_des_folder = ''
    for img in imgs:
        if img_count % each_folder_num == 0:
            cur_des_folder = des_img_root + 'chamo_%05d/' % folder_count
            folder_count = folder_count + 1
            os.makedirs(cur_des_folder)
            print('%d images splited' % img_count)
        img_count = img_count + 1
        shutil.move(img_root + img, cur_des_folder + img)


def devide_pic(pic_src_path, pic_des_path, excel_path, materials, each_folder_num):
    if not os.path.exists(pic_src_path) or not os.path.exists(excel_path):
        logger.I(TAG, 'devide_pic: ' + pic_src_path + 'not exist')
        print('devide_pic: ' + pic_src_path + 'not exist')
        return
    temp_path_mat = os.path.join(pic_src_path, 'temp_mat' + str(time.clock()))
    if not os.path.exists(temp_path_mat):
        os.mkdir(temp_path_mat)
    mat_dict = get_mat_num(materials, excel_path)
    create_and_copy(mat_dict, pic_src_path, temp_path_mat)
    if not os.path.exists(pic_des_path):
        os.makedirs(pic_des_path)
    temp_path_merge = os.path.join(pic_src_path, 'temp_merge' + str(time.clock()))
    merge_all(temp_path_mat, temp_path_merge, materials)
    tag_path(temp_path_merge, materials)
    split_img(temp_path_merge, pic_des_path, each_folder_num)

if __name__ == '__main__':
    devide_pic(pic_src_path, pic_des_path, excel_path, materials, 1000)
    #mat_dict = get_mat_num(material_list, excel_path)
    # # for i in mat_dict:
    # #     print(i, ':', mat_dict[i])
    #create_and_copy(mat_dict, pic_src_path, pic_des_path)
    # # merge_num(pic_des_path, pic_merge_path_train, data_scraping.materil_name.material_list, 4000)
    #merge_all(pic_des_path, pic_merge_path_train, material_list)
    # merge_num_test(pic_des_path, pic_merge_path_test, pic_merge_path_train, materials, 5)
    #tag_path(pic_merge_path_train, material_list)




