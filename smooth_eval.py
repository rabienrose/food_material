import tensorflow as tf
import config
import data_preprocessing
import net
import accuracy.default_accuracy as default_accuracy
import accuracy.multi_accuracy as multi_accuracy
import data_scraping.picDevider as picDevider
import data_convertor.process_img as process_img
import utils
import sys
import config.chamo
import config.chamo_full_run
import data_preprocessing.test_preprocess
import net.vgg16
import net.mobilenet_v2
import matplotlib
import matplotlib.pyplot as plt
import data_scraping.materil_name
import numpy as np

excel_path = '/home/leo/Documents/chamo/food_material/V1.1.0.0525.xlsx'

pic_src_path = '/home/leo/Downloads/chamo/alltest/alltest_devider/chamo_00000/'

pic_des_path_devider = '/home/leo/Downloads/chamo/alltest/alltest_devider/chamo_00000/'

pic_des_path_tfrecord = '/home/leo/Downloads/chamo/alltest/tfrecord/'

pic_merge_path_train = 'E:/material_merge/'

label_dim = len(picDevider.materials)


def get_config(config_name):
    print('choose config: '+config_name)
    config_obj=None
    if config_name=='chamo':
        config_obj=config.chamo.get_config()
    elif config_name=='chamo_full_run':
        config_obj = config.chamo_full_run.get_config()
    return config_obj


def eval_smooth_show(config_obj):
    test_preprocess_obj=data_preprocessing.test_preprocess.test_preprocess(config_obj.tfrecord_test_addr, config_obj.class_num)
    net_name=config_obj.net_type
    test_net_obj=None
    if net_name=='vgg16':
        test_net_obj=net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
    elif net_name=='mobilenet_v2':
        test_net_obj=net.mobilenet_v2.mobilenet_v2(True, 'mobilenet_v2', config_obj.class_num)

    accu_name=config_obj.accuracy_type
    accu_obj=None
    if accu_name=='default':
        accu_obj=default_accuracy.default_accuracy()
    elif accu_name=='multi':
        accu_obj=multi_accuracy.multi_accuracy()

    images_test, labels_test = test_preprocess_obj.def_preposess()
    net_test = test_net_obj.def_net(images_test)
    inputs=tf.sigmoid(net_test)
    predict=tf.cast(inputs> 0.5, tf.float32)
    accuracy_perfect, accuracy, precision, recall, acc_list, pre_list, rec_list = accu_obj.def_accuracy(net_test, labels_test)

    saver = tf.train.Saver()
    img_mean = utils.global_var.means
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, config_obj.result_addr)

        accuracy_prefect_v, accuracy_v, precision_v, recall_v, acc_list_v, pre_list_v, rec_list_v, predict_v, labels_test_v, images_test_v = sess.run([accuracy_perfect, accuracy, precision, recall, acc_list, pre_list, rec_list, predict, labels_test, images_test])
        print('accuracy_prefect_v: %s' % accuracy_prefect_v)
        print('accuracy_v: ', accuracy_v)
        print('precision:' , precision_v)
        print('recall:' , recall_v)
        print('acc_list_v:' , acc_list_v)
        print('pre_list_v:' , pre_list_v)
        print('rec_list_v:' , rec_list_v)
        print('len(predict_V):', len(predict_v))
        for k in range(len(predict_v)):
            #print(labels_test_v[k])
            mat_show=[]
            for i in range(len(predict_v[k])):
                if predict_v[k][i] == 1:
                    mat_show.append(data_scraping.materil_name.material_list[i])
                    print(str(data_scraping.materil_name.material_list[i]))
            print(predict_v[k])
            show_img=images_test_v[k]
            show_img=show_img+img_mean
            show_img = abs(show_img) / 256.0
            plt.imshow(show_img)
            zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
            for i in range(len(mat_show)):
                plt.text(150, 25*(i+1), str(mat_show[i]), fontproperties = zhfont, fontsize=15, color='red')
            plt.show()

        coord.request_stop()
        coord.join(threads)


def eval_smooth(config_name, repeat_time):
    config_name = config_name
    print('choose config: ' + config_name)
    config_obj = None
    if config_name == 'chamo':
        config_obj = config.chamo.get_config()
    elif config_name == 'chamo_full_run':
        config_obj = config.chamo_full_run.get_config()

    test_preprocess_obj = data_preprocessing.test_preprocess.test_preprocess(config_obj.tfrecord_test_addr,
                                                                             config_obj.class_num)
    net_name = config_obj.net_type
    test_net_obj = None
    if net_name == 'vgg16':
        test_net_obj = net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)
    elif net_name == 'mobilenet_v2':
        test_net_obj = net.mobilenet_v2.mobilenet_v2(True, 'mobilenet_v2', config_obj.class_num)

    accu_name = config_obj.accuracy_type
    accu_obj = None
    if accu_name == 'default':
        accu_obj = default_accuracy.default_accuracy()
    elif accu_name == 'multi':
        accu_obj = multi_accuracy.multi_accuracy()

    images_test, labels_test = test_preprocess_obj.def_preposess()
    net_test = test_net_obj.def_net(images_test)
    inputs = tf.sigmoid(net_test)
    predict = tf.cast(inputs > 0.5, tf.float32)
    accuracy_perfect, accuracy, precision, recall, f1, acc_list, \
        pre_list, pre_list_nume, pre_list_deno, rec_list, \
        rec_list_nume, rec_list_deno = accu_obj.def_accuracy(net_test, labels_test)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, config_obj.result_addr)

        # len_all = len(labels_test)
        each_size = labels_test.get_shape().as_list()[0]
        len_all = labels_test.get_shape().as_list()[1]

        acc_perfect_all = 0.0
        acc_all = 0.0
        precision_all = 0.0
        recall_all = 0.0

        acc_list_all = np.zeros(shape=[len_all], dtype=np.float32)
        precision_all_nume = np.zeros(shape=[len_all], dtype=np.float32)
        precision_all_deno = np.zeros(shape=[len_all], dtype=np.float32)
        precision_list_all = np.zeros(shape=[len_all], dtype=np.float32)
        recall_all_nume = np.zeros(shape=[len_all], dtype=np.float32)
        recall_all_deno = np.zeros(shape=[len_all], dtype=np.float32)
        recall_list_all = np.zeros(shape=[len_all], dtype=np.float32)

        for repeat_i in range(1, repeat_time+1):
            accuracy_perfect_v, accuracy_v, precision_v, recall_v, acc_list_v, pre_list_nume_v, pre_list_deno_v, \
            rec_list_nume_v, rec_list_deno_v, predict_v, labels_test_v, images_test_v = sess.run(
                [accuracy_perfect, accuracy, precision, recall, acc_list,
                 pre_list_nume, pre_list_deno, rec_list_nume, rec_list_deno, predict, labels_test,
                 images_test])

            acc_perfect_all = acc_perfect_all + accuracy_perfect_v
            acc_all = acc_all + accuracy_v
            precision_all = precision_all + precision_v
            recall_all = recall_all + recall_v

            acc_list_all = np.nan_to_num(acc_list_all) + acc_list_v
            precision_all_nume = precision_all_nume + pre_list_nume_v
            precision_all_deno = precision_all_deno + pre_list_deno_v
            recall_all_nume = recall_all_nume + rec_list_nume_v
            recall_all_deno = recall_all_deno + rec_list_deno_v

            repeat_i = float(repeat_i)
            print('step: %d total pictures: %d' % (repeat_i, each_size*repeat_i))
            print('accuracy_prefect_v:', acc_perfect_all/repeat_i)
            print('accuracy_v: ', acc_all/repeat_i)
            print('precision:', precision_all/repeat_i)
            print('recall:', recall_all/repeat_i)
            print('acc_list_v:', acc_list_all/repeat_i)
            print('pre_list_v:', precision_all_nume/precision_all_deno)
            print('rec_list_v:', recall_all_nume/recall_all_deno)

        coord.request_stop()
        coord.join(threads)
    repeat_time = float(repeat_time)
    acc_perfect_all = acc_perfect_all/repeat_time
    acc_all = acc_all/repeat_time
    precision_all = precision_all/repeat_time
    recall_all = recall_all/repeat_time
    acc_list_all = acc_list_all/repeat_time
    precision_list_all = precision_all_nume/precision_all_deno
    recall_list_all = recall_all_nume/recall_all_deno
    return acc_perfect_all, acc_all, precision_all, recall_all, acc_list_all, precision_list_all, recall_list_all


if __name__ == '__main__':
    config_obj = get_config('chamo')
    #picDevider.devide_pic(pic_src_path, pic_des_path_devider, excel_path, picDevider.materials, 1000)
    #process_img.check_and_convert(pic_des_path_devider, pic_des_path_tfrecord, label_dim, 6)
    config.tfrecord_test_addr = pic_des_path_tfrecord
    config.tfrecord_addr = pic_des_path_tfrecord
    eval_smooth('chamo', 10)
