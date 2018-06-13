import data_convertor.imgutils
import data_convertor.jpegcheck
import data_convertor.img_2_tfrecord
from multiprocessing import Pool
import os

#if the code is executed in module, there will be no warning for the error. If you find the code exit
#abnomally, use the log to find where the error is.

#the number of bit in label
label_dim=102
#root address of all the img folders
file_root="/home/leo/Downloads/chamo/train_div/"
#root address to save the tfrecord
tfrecord_root='/home/leo/Downloads/chamo/tfrecord/'
#number of thread to execute the code
thread_count=6


def checkFormat(file_root, thread_count):
    p = Pool(processes=thread_count)
    file_list = os.listdir(file_root)
    for file_name in file_list:
        p.apply_async(data_convertor.jpegcheck.jpeg_check, args=(file_root+file_name,))
    p.close()
    p.join()


def checkChannel(file_root, thread_count):
    file_list = os.listdir(file_root)
    for file_name in file_list:
        data_convertor.imgutils.main(file_root+file_name, '*.jpg', '1-2-3', thread_count)


def convertTFRecord(file_root, tfrecord_root, label_dim, thread_count):
    if not os.path.exists(tfrecord_root):
        os.makedirs(tfrecord_root)
    data_convertor.img_2_tfrecord.main(file_root, tfrecord_root, label_dim, thread_count)


def check_and_convert(file_root, tfrecord_root, label_dim, thread_count):
    checkFormat(file_root, thread_count)
    checkChannel(file_root, thread_count)
    convertTFRecord(file_root, tfrecord_root, label_dim, thread_count)


if __name__ == '__main__':
    checkFormat(file_root, thread_count)
    checkChannel(file_root, thread_count)
    convertTFRecord(file_root, tfrecord_root, label_dim, thread_count)

