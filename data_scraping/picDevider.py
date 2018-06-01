# -*- coding: utf-8 -*-
import os
from collections import OrderedDict
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import shutil
import filecmp

excel_path = 'E:\\data\\v1.0.0_0507.xlsx'

pic_src_path = 'D:\\1000_data\\train_no_expand'

pic_des_path = 'D:\\fine_filtered\\test\\'

pic_merge_path_train = 'D:\\only200_train'

pic_merge_path_test = 'D:\\only200_test'

materials = [u'青椒', u'炒肉', u'鸡蛋', u'木耳']
#materials = ['木耳肉片']

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
                if material in row[1]:
                    if not material in ret.keys():
                        ret[material] = []
                    ret[material].append([row[0], row[1].strip()])
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
        m_path = des_path + '\\' + material
        if not os.path.exists(m_path):
            os.mkdir(m_path)
        for num, name in mat_dict[material]:
            num_src = src_path + '\\'  + num
            num_des = des_path + '\\' + material + '\\' + name
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


def merge(src_path, des_path, mat_array):
    '''
    把通过食材分类的菜品进行标记并整合到同一个文件夹
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
    for mat_path in mat_list:
        # 得到食材的名称
        mat_name = mat_path
        # 转换为完整路径
        mat_path = os.path.join(src_path, mat_path)
        num_list = os.listdir(mat_path)
        for veg_path in num_list:
            # 得到具体的菜品
            veg_name = veg_path
            # 得到具体的食材的菜品的文件夹路径
            veg_path = os.path.join(mat_path, veg_path)
            print('处理路径：', veg_path)
            for root, dirs, files in os.walk(veg_path):
                for file in files:
                    src_file_path = os.path.join(root, file)
                    des_file_path = os.path.join(des_path, file)
                    file_existed, pic_des_path = is_file_exist(src_file_path, des_path)
                    if not file_existed:
                        shutil.copyfile(src_file_path, des_file_path)
                    else:
                        des_file_path = pic_des_path
                    merge_name(des_file_path, veg_name, mat_array.index(mat_name), len(mat_array))


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
            print('处理路径：', veg_path)
            # 每个文件夹当前的图片获取数
            num_cur = 0
            for root, dirs, files in os.walk(veg_path):
                for file in files:
                    src_file_path = os.path.join(root, file)
                    des_file_path = os.path.join(des_path, file)
                    file_existed, pic_des_path = is_file_exist(src_file_path, des_path)
                    if not file_existed:
                        shutil.copyfile(src_file_path, des_file_path)
                        merge_name(des_file_path, veg_name, mat_array.index(mat_name), len(mat_array))
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
                    merge_name(des_file_path, veg_name, mat_array.index(mat_name), len(mat_array))
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
        des_name = name + '_' + str(pow(10, length-num-1)).zfill(length) + '_' + str(PIC_NUM) + '_' + SUF_LABEL + '.jpg'
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
        temp = pow(10, length - num)
        if int(past_num_i / pow(10, length - num - 1)) % 2 == 0:
            past_num_i += pow(10, length - num - 1)
            print('past_num_i changed', past_num_i, file_path)
        else:
            return file_path
        des_name = file_path_basename[0:num_start + 1] \
                   + str(past_num_i).zfill(length) \
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
                if mat in file:
                    file_path = merge_name(file_path, 'test', mat_array.index(mat), len(mat_array))


# 判断该文件夹下是否包含与file相同的文件
def is_file_exist(file, des_path):
    for root, dirs, files in os.walk(des_path):
        for f in files:
            f_path = os.path.join(root, f)
            if is_same_file(f_path, file):
                return True, f_path
    return False, None


# 比较两个文件是否相同
def is_same_file(file_l, file_r):
    return filecmp.cmp(file_l, file_r)

if __name__ == '__main__':
    #mat_dict=get_mat_num(materials, excel_path)
    #create_and_copy(mat_dict, pic_src_path, pic_des_path)
    merge_num(pic_des_path, pic_merge_path_test, materials, 4000)
    #merge_num_test(pic_des_path, pic_merge_path_test, pic_merge_path_train, materials, 5)
    tag_path(pic_merge_path_test, materials)



