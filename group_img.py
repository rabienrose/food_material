import sys
import os
import tensorflow as tf
import config.chamo
import config.chamo_full_run
import config.chamo_class2
import net.vgg16
import net.mobilenet_v2
import shutil
import utils.data_helper
import utils.global_var
import data_scraping.materil_name


config_name=sys.argv[1]
print('choose config: '+config_name)
config_obj=None
if config_name=='chamo':
    config_obj=config.chamo.get_config()
elif config_name=='chamo_full_run':
    config_obj = config.chamo_full_run.get_config()
elif config_name == 'chamo_class2':
    config_obj = config.chamo_class2.get_config()

group_root='/home/leo/Downloads/chamo/v3_material/result'
material_list = data_scraping.materil_name.material_list
material_dict = {}
count=0
for item in material_list:
    material_dict[item[0]]=count
    count=count+1
material_dict={'青椒': 0, '木耳': 1, '鸡蛋': 2}
for material_name in material_dict.keys():
    os.system('mkdir ' + group_root + '/' + material_name)
    os.system('mkdir ' + group_root + '/' + material_name + '/positive')
    os.system('mkdir ' + group_root + '/' + material_name + '/negative')
materials = list(material_dict.keys())

net_obj=None
net_obj = net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', config_obj.class_num)

#image_root='/home/leo/Downloads/chamo/v3_material/auto/青椒/positive'
image_root='/home/leo/Downloads/chamo/v3_material/chamo_00001'

image_raw_data = tf.placeholder(tf.string, None)

img_data_jpg = tf.image.decode_jpeg(image_raw_data)
image = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)

image = utils.data_helper._aspect_preserving_resize(image, utils.global_var._RESIZE_SIDE_MIN)
train_image_size = utils.global_var._RESIZE_SIDE_MIN
image = utils.data_helper._central_crop([image], train_image_size, train_image_size)[0]
image.set_shape([train_image_size, train_image_size, 3])
image = tf.to_float(image)
image=utils.data_helper._mean_image_subtraction(image)
image=tf.expand_dims(image,[0])
net_test = net_obj.def_net(image)
inputs=tf.sigmoid(net_test)
predict=tf.cast(inputs> 0.9, tf.float32)
saver = tf.train.Saver()
count=0

with tf.Session() as sess:
    saver.restore(sess, config_obj.result_addr+config_obj.ckpt_name+'/chamo.ckpt')
    folders = os.listdir(image_root)
    for file in folders:
        count=count+1
        img_name=image_root+'/'+file
        print(img_name)
        image_raw_data_jpg = tf.gfile.FastGFile(img_name, 'rb').read()
        predict_v = sess.run(predict,feed_dict={image_raw_data: image_raw_data_jpg})
        for m in range(len(predict_v[0])):
            if predict_v[0][m]==1:
                folder_true = group_root + '/' +materials[m]+'/positive'
                shutil.copy(img_name, folder_true)
            else:
                folder_false = group_root + '/' + materials[m] + '/negative'
                shutil.copy(img_name, folder_false)
        if count>1000:
            break
