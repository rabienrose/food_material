import tensorflow as tf
import os
from PIL import Image
from multiprocessing import Pool
import sys
import time

def Denary2Binary(n):
    '''convert denary integer n to binary string bStr'''
    bStr = ''
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr

def convert_a_folder(folder_name,file_root,tfrecord_root,lable_bit):
    print('start convert %s' % folder_name)
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_root + folder_name + '.tfrecord')
    img_list = os.listdir(file_root + folder_name)
    img_count = 0
    for img_name in img_list:
        if img_name.find('.jpg'):
            try:
                image_data = tf.gfile.FastGFile(file_root + folder_name + '/' + img_name, 'rb').read()
                img = Image.open(file_root + folder_name + '/' + img_name, 'r')
                size = img.size
                splited = img_name.split('_')
                code = splited[1]
                code = Denary2Binary(int(code))
                bit_array = [0 for i in range(lable_bit)]
                bit_count = lable_bit
                for i in range(len(code)):
                    bit_count = bit_count - 1
                    bit_array[bit_count] = (int(code[len(code) - i - 1]))
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=bit_array)),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                        'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                        'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                    }))
                tfrecord_writer.write(example.SerializeToString())
                img_count = img_count + 1
            except:
                print("filenames:%s,i:%d" % (file_name, img_count))
                continue
    tfrecord_writer.close()

def convert_folders(folder_list,file_root,tfrecord_root,lable_bit):
    for item in folder_list:
        convert_a_folder(item,file_root,tfrecord_root,lable_bit)

def main(file_root, tfrecord_root, lable_bit, thread_count):
    file_list = os.listdir(file_root)
    folder_lists = []
    for i in range(thread_count):
        folder_lists.append([])
    file_count = 0
    max_file_count = len(file_list)
    while True:
        is_done = False
        for i in range(thread_count):
            if file_count < max_file_count:
                folder_lists[i].append(file_list[file_count])
                file_count = file_count + 1
            else:
                is_done = True
                break
        if is_done:
            break
    p = Pool()
    for i in range(thread_count):
        print(folder_lists[i])
        p.apply_async(convert_folders, args=(folder_lists[i],file_root,tfrecord_root,lable_bit,))
    p.close()
    p.join()
