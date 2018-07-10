import data_convertor.imgutils
import data_convertor.jpegcheck
import data_convertor.img_2_tfrecord
from multiprocessing import Pool
import data_scraping.materil_name
import os

#if the code is executed in module, there will be no warning for the error. If you find the code exit
#abnomally, use the log to find where the error is.

#the number of bit in label
label_dim=73
#root address of all the img folders
file_root="/home/leo/Documents/chamo/transfer/tool/img/re2"
#root address to save the tfrecord
tfrecord_root='/home/leo/Downloads/chamo/v2_material/test_tfrecord/'
#number of thread to execute the code
thread_count=6


def checkFormat(file_root, thread_count):
    p = Pool(processes=thread_count)
    #file_list = os.listdir(file_root)
    #for file_name in file_root:
    p.apply_async(data_convertor.jpegcheck.jpeg_check, args=(file_root,))
    p.close()
    p.join()


def checkChannel(file_root, thread_count):
    #file_list = os.listdir(file_root)
    #for file_name in file_root:
    data_convertor.imgutils.main(file_root, '*.jpg', '1-2-3', thread_count)


def convertTFRecord(file_root, tfrecord_root, label_dim, thread_count):
    if not os.path.exists(tfrecord_root):
        os.makedirs(tfrecord_root)
    data_convertor.img_2_tfrecord.main(file_root, tfrecord_root, label_dim, thread_count)


def check_and_convert(file_root, tfrecord_root, label_dim, thread_count):
    checkFormat(file_root, thread_count)
    checkChannel(file_root, thread_count)
    convertTFRecord(file_root, tfrecord_root, label_dim, thread_count)


if __name__ == '__main__':
    material_list = data_scraping.materil_name.material_list
    #material_list=[["青椒"],
    #["鸡蛋"],
    #["炒肉"]]
    #file_list = os.listdir(file_root)
    #count=0
    #for item in file_list:
    #    print(item)
    checkFormat(file_root, thread_count)
    checkChannel(file_root, thread_count)

    #convertTFRecord(file_root, tfrecord_root, label_dim, thread_count)

