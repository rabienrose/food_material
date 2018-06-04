import tensorflow as tf
import os
from PIL import Image

def Denary2Binary(n):
    '''convert denary integer n to binary string bStr'''
    bStr = ''
    if n == 0: return '0'
    while n > 0:
        bStr = str(n % 2) + bStr
        n = n >> 1
    return bStr

lable_bit=20
img_root=u"D:\\only200_train\\"
tfrecord_batch_size=20
sess=tf.Session()
img_list = os.listdir(img_root)
img_count=0
rfrecod_file_count=0
tfrecord_writer=None
for file_name in img_list:
    if file_name.find('.jpg'):
        if img_count%tfrecord_batch_size ==0:
            if tfrecord_writer!= None:
                tfrecord_writer.close()
            tfrecord_writer=tf.python_io.TFRecordWriter('../tfrecord/chamo_%05d.tfrecord' % rfrecod_file_count)
            rfrecod_file_count=rfrecod_file_count+1
        img_count = img_count + 1
        img = Image.open(img_root + '/' + file_name, 'r')
        size = img.size
        img_raw = img.tobytes()
        splited=file_name.split('_')
        code=splited[1]
        code = Denary2Binary(int(code))
        bit_array=[ 0 for i in range(lable_bit)]
        bit_count=lable_bit
        for i in range(len(code)):
            bit_count = bit_count - 1
            bit_array[bit_count]= (int(code[len(code)-i-1]))
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=bit_array)),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
            }))
        tfrecord_writer.write(example.SerializeToString())
