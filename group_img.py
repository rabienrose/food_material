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

checkpt='/home/leo/Documents/chamo/transfer/output/chamo_8000.000000_0.001457/chamo.ckpt'

def process_img(imgs, re_dir):
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
    predict=tf.cast(inputs> 0.4, tf.float32)
    saver = tf.train.Saver()
    count=0

    with tf.Session() as sess:
        saver.restore(sess, checkpt)
        for image_name in imgs:
            image_raw_data_jpg = tf.gfile.FastGFile(image_name, 'rb').read()
            count=count+1
            predict_v, img_data = sess.run([predict,img_show],feed_dict={image_raw_data: image_raw_data_jpg})
            mat_show = []
            print(predict_v)
            for i in range(len(predict_v[0])):
                if predict_v[0][i] == 1:
                    mat_show.append(data_scraping.materil_name.material_list[i])
            print(mat_show)
            # show_img = img_data
            show_img = abs(img_data) / 256.0
            plt.imshow(show_img)
            zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
            for i in range(len(mat_show)):
                plt.text(150, 25 * (i + 1), str(mat_show[i]), fontproperties=zhfont, fontsize=40, color='red')
            #plt.show()
            file_name=image_name.split('/')[-1]
            plt.savefig(re_dir+'/'+file_name)
            plt.close('all')


if __name__ == '__main__':
    img_root='/home/leo/Downloads/chamo/self_collect_all'
    imgs = os.listdir(img_root)
    img_data_list=[]
    for img_name in imgs:
        img_data_list.append(img_root+'/'+img_name)
    process_img(img_data_list)
