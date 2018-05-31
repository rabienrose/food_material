import sys
import config.chamo
import config.chamo_full_run
import data_preprocessing.test_preprocess
import net.vgg16
import tensorflow as tf
import accuracy.default_accuracy
import matplotlib.pyplot as plt
import utils.global_var

config_name=sys.argv[1]
print('choose config: '+config_name)
config_obj=None
if config_name=='chamo':
    config_obj=config.chamo.get_config()
elif config_name=='chamo_full_run':
    config_obj = config.chamo_full_run.get_config()

test_preprocess_obj=data_preprocessing.test_preprocess.test_preprocess(config_obj.tfrecord_test_addr, config_obj.class_num)
net_name=config_obj.net_type
test_net_obj=None
if net_name=='vgg16':
    test_net_obj=net.vgg16.vgg16(False, 'vgg16', config_obj.class_num)

accu_name=config_obj.accuracy_type
accu_obj=None
if accu_name=='default':
    accu_obj=accuracy.default_accuracy.default_accuracy()

images_test, labels_test = test_preprocess_obj.def_preposess()
net_test = test_net_obj.def_net(images_test)
inputs=tf.sigmoid(net_test)
predict=tf.cast(inputs> 0.5, tf.float32)
inputs=tf.cast((tf.reduce_sum(tf.abs(labels_test - predict),axis=1)<0.1), tf.float32)
accuracy = tf.reduce_mean(inputs)
saver = tf.train.Saver()
img_mean = utils.global_var.means
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, config_obj.result_addr)

    accuracy_v, predict_v, labels_test_v, images_test_v = sess.run([accuracy, predict, labels_test, images_test])
    print('accuracy: %s' % accuracy_v)
    for k in range(len(predict_v)):
        #print(labels_test_v[k])
        print(predict_v[k])
        show_img=images_test_v[k]
        show_img=show_img+img_mean
        show_img = abs(show_img) / 256.0
        plt.imshow(show_img)
        plt.show()

    coord.request_stop()
    coord.join(threads)

