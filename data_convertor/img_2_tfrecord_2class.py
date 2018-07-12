import tensorflow as tf
import os
from PIL import Image
import random
import data_scraping.materil_name
from multiprocessing import Pool
import sys
import time
def convert_a_folder(file_root,tfrecord_root, material_dict):

    img_name_list=[]
    label_list=[]
    material_list=[]
    for material_name in material_dict:
        for isTrue in ['positive','negative']:
            folder_addr=file_root+'/' + material_name+'/'+isTrue
            #folder_addr = file_root + '/' + isTrue
            img_list = os.listdir(folder_addr)
            for img_name in img_list:
                if img_name.find('.jpg'):
                    img_addr=folder_addr + '/' + img_name
                    img_name_list.append(img_addr)
                    if isTrue=='positive':
                        label_list.append(1)
                    else:
                        label_list.append(0)
                    material_list.append(material_dict[material_name])
    count=len(img_name_list)
    id_list=list(range(1, count))
    random.shuffle(id_list)
    img_count=0
    print(count)
    tfrecord_writer = None
    count=0
    tfrecord_count=0
    for id in id_list:
        if count%10000==0:
            tfrecord_name='/chamo_%05d.tfrecord' % tfrecord_count
            if tfrecord_writer!=None:
                tfrecord_writer.close()
            tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_root + tfrecord_name)
            tfrecord_count=tfrecord_count+1
        count=count+1
        image_data = tf.gfile.FastGFile(img_name_list[id], 'rb').read()
        img = Image.open(img_name_list[id], 'r')
        size = img.size
        bit_array= [0 for i in range(len(material_dict))]
        bit_pos=material_list[id]
        bit_array[bit_pos]=1*2
            #the first decimal indecate the true or false of a material, the second decimal inecates mask
        try:
            if label_list[id]==1:
                bit_array[bit_pos] = bit_array[bit_pos]+1
            else:
                bit_array[bit_pos] = bit_array[bit_pos]+0
            #print(bit_array)
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
            print("filenames:%s,i:%d" % (material_name, img_count))
            continue
    tfrecord_writer.close()

if __name__ == '__main__':
    file_root = '/home/leo/Downloads/chamo/v3_material/m_all/all/test'
    tfrecord_root = '/home/leo/Downloads/chamo/v3_material/m_all/all/test/tfrecord'

    material_list = data_scraping.materil_name.material_list
    material_dict = {}
    count = 0
    for item in material_list:
        material_dict[item[0]] = count
        count = count + 1
    convert_a_folder(file_root, tfrecord_root, material_dict)
