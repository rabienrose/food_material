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

def process_img(img):
    material_list = data_scraping.materil_name.material_list
    material_dict = {}
    count=0
    for item in material_list:
        material_dict[item[0]]=count
        count=count+1


    net_obj=None
    net_obj = net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', len(material_list))

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
