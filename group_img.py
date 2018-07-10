import sys
import os
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import net.vgg16
import net.mobilenet_v2
import shutil
import utils.data_helper
import utils.global_var
import data_scraping.materil_name

checkpt=''

def process_img(imgs):
    material_list = data_scraping.materil_name.material_list
    net_obj = net.mobilenet_v2.mobilenet_v2(False, 'mobilenet_v2', len(material_list))
    image_raw_data = tf.placeholder(tf.string, None)
    img_data_jpg = tf.image.decode_jpeg(image_raw_data)
    image = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
    image = utils.data_helper._aspect_preserving_resize(image, utils.global_var._RESIZE_SIDE_MIN)
    train_image_size = utils.global_var._RESIZE_SIDE_MIN
    image = utils.data_helper._central_crop([image], train_image_size, train_image_size)[0]
    image.set_shape([train_image_size, train_image_size, 3])
    image = tf.to_float(image)
    img_show=image
    image=utils.data_helper._mean_image_subtraction(image)
    image=tf.expand_dims(image,[0])
    net_test = net_obj.def_net(image)
    inputs=tf.sigmoid(net_test)
    predict=tf.cast(inputs> 0.9, tf.float32)
    saver = tf.train.Saver()
    count=0

    with tf.Session() as sess:
        saver.restore(sess, checkpt)
        for image_raw_data_jpg in imgs:
            count=count+1
            predict_v, img_data = sess.run([predict,img_show],feed_dict={image_raw_data: image_raw_data_jpg})
            mat_show = []
            for i in range(len(predict_v)):
                if predict_v[i] == 1:
                    mat_show.append(data_scraping.materil_name.material_list[i])
            print(mat_show)
            # show_img = img_data
            show_img = abs(img_data) / 256.0
            # plt.imshow(show_img)
            zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
            for i in range(len(mat_show)):
                plt.text(150, 25 * (i + 1), str(mat_show[i]), fontproperties=zhfont, fontsize=15, color='red')
            plt.show()

img_root='D:\\self_collect\\try'
imgs = os.listdir(img_root)
img_data_list=[]
for img_name in imgs:
    image_raw_data_jpg = tf.gfile.FastGFile(img_name, 'rb').read()
    img_data_list.append(image_raw_data_jpg)
process_img(img_data_list)
