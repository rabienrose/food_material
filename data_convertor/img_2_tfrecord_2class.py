import tensorflow as tf
import os
from PIL import Image
from multiprocessing import Pool
import sys
import time
def convert_a_folder(folder_name,file_root,tfrecord_root, material_dict):
    print('start convert %s' % folder_name)
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_root +'/'+ folder_name + '.tfrecord')
    for isTrue in ['true','false']:
        folder_addr=file_root +'/'+ folder_name+'/'+isTrue
        img_list = os.listdir(folder_addr)
        img_count = 0
        for img_name in img_list:
            if img_name.find('.jpg'):
                try:
                    img_addr=folder_addr + '/' + img_name
                    image_data = tf.gfile.FastGFile(img_addr, 'rb').read()
                    img = Image.open(img_addr, 'r')
                    size = img.size
                    bit_array= [0 for i in range(len(material_dict))]
                    bit_pos=material_dict[folder_name]
                    bit_array[bit_pos]=1*2
                    #the first decimal indecate the true or false of a material, the second decimal inecates mask
                    if isTrue=='true':
                        bit_array[bit_pos] = bit_array[bit_pos]+1
                    else:
                        bit_array[bit_pos] = bit_array[bit_pos]+0
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

if __name__ == '__main__':
    folder_name='青椒'
    file_root='E:/data'
    tfrecord_root = file_root+'/'+folder_name+'/tfrecord'
    material_dict={'青椒':0, '鸡蛋':1, '炒肉':2}
    convert_a_folder(folder_name, file_root, tfrecord_root, material_dict)