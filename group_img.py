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


def process_img(imgs, re_dir,result_group, checkpt, material_list):
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
    #predict=tf.cast(inputs> 0.95, tf.float32)
    saver = tf.train.Saver()
    count=0
    for material_name in material_list:
        os.system('mkdir ' + result_group + '/' + material_name[0])
        os.system('mkdir ' + result_group + '/' + material_name[0] + '/positive')
        os.system('mkdir ' + result_group + '/' + material_name[0] + '/negative')

    with tf.Session() as sess:
        saver.restore(sess, checkpt)
        for image_name in imgs:
            print(image_name)
            image_raw_data_jpg = tf.gfile.FastGFile(image_name, 'rb').read()
            count=count+1
            predict_v, img_data = sess.run([inputs,img_show],feed_dict={image_raw_data: image_raw_data_jpg})
            mat_show = []
            #print(predict_v)
            mat_thres=[0.8, 0.9, 0.9, 0.6]
                     # 芹菜, 香菇, 木耳, 花菜
            for i in range(len(predict_v[0])):
                if predict_v[0][i] > mat_thres[i]:
                    predict_v[0][i] = 1
                    mat_show.append(material_list[i])
                else:
                    predict_v[0][i] = 0

            # show_img = img_data
            show_img = abs(img_data) / 256.0
            plt.imshow(show_img)
            zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
            for i in range(len(mat_show)):
                plt.text(150, 25 * (i + 1), str(mat_show[i]), fontproperties=zhfont, fontsize=20, color='red')
            #plt.show()
            file_name=image_name.split('/')[-1]
            plt.savefig(re_dir+'/'+file_name)
            plt.close('all')
            for m in range(len(predict_v[0])):
                if predict_v[0][m] == 1:
                    folder_true = result_group + '/' + material_list[m][0] + '/positive'
                    shutil.copy(image_name, folder_true)
                else:
                    folder_false = result_group + '/' + material_list[m][0] + '/negative'
                    shutil.copy(image_name, folder_false)


if __name__ == '__main__':
    img_root='/home/leo/Downloads/chamo/self_collect_all'
    imgs = os.listdir(img_root)
    img_data_list=[]
    for img_name in imgs:
        img_data_list.append(img_root+'/'+img_name)
    process_img(img_data_list)
