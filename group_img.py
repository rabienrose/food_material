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


config_name=sys.argv[1]
print('choose config: '+config_name)
config_obj=None
if config_name=='chamo':
    config_obj=config.chamo.get_config()
elif config_name=='chamo_full_run':
    config_obj = config.chamo_full_run.get_config()
elif config_name == 'chamo_class2':
    config_obj = config.chamo_class2.get_config()

group_root=config_obj.result_addr
folder_true = group_root+'/true'
folder_false = group_root+'/false'

if not os.path.exists(folder_true):
    os.makedirs(folder_true)
if not os.path.exists(folder_false):
    os.makedirs(folder_false)

net_name=config_obj.net_type
net_obj=None
test_net_obj=None
if net_name=='False':
    net_obj=net.vgg16.vgg16(True, 'vgg16', config_obj.class_num)
elif net_name=='mobilenet_v2':
    net_obj = net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', config_obj.class_num)

image_root='E:/data/green_pepper/chamo_00049'

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
predict=tf.cast(inputs> 0.5, tf.float32)
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
        print(predict_v[0][0])
        if predict_v[0][0]==1:
            shutil.move(img_name, folder_true)
        else:
            shutil.move(img_name, folder_false)
        if count>2500:
            break
