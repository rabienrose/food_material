import tensorflow as tf
import os
from PIL import Image

img_root="D:/material_try_all/"
sess=tf.Session()
with tf.python_io.TFRecordWriter('chamo.tfrecord') as tfrecord_writer:
    img_list = os.listdir(img_root)
    for file_name in img_list:
        if file_name.find('.jpg'):
            img = Image.open(img_root + '/' + file_name, 'r')
            size = img.size
            img_raw = img.tobytes()
            splited=file_name.split('_')
            code=splited[1]
            bit_array=[]
            for bit in code:
                bit_array.append(int(bit))
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=bit_array)),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                }))
            tfrecord_writer.write(example.SerializeToString())