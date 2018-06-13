import tensorflow as tf
import numpy as np
import net.mobilenet_v2_impl
import random
import accuracy.multi_accuracy as multi_accuracy
import accuracy.default_accuracy as default_accuracy
import utils.data_helper as data_helper


is_training = True
sess_path = './test/test.ckpt'


def get_rand_np():
    ret_y = []
    temp = []
    for i in range(10):
        num = random.randint(0, 10)
        ret_y.append([1.0, 0.0] if num > 4 else [0.0, 1.0])
        for j in range(5):
            raw_temp = []
            for k in range(5):
                raw_temp.append(num)
            temp.append(raw_temp)
    ret = np.array(temp).reshape([10, 5, 5, 1])
    ret = ret.astype(np.float32)
    return ret, ret_y


x_data = tf.placeholder(shape=[10, 5, 5, 1], dtype=np.float32, name='x_data')
y_data = tf.placeholder(shape=[10, 2], dtype=np.float32, name='y_data')

# no dropout
net_obj = net.mobilenet_v2_impl.mobilenet_v2_impl(is_training, 'mobilenet_v2', 2)

y_temp = net_obj.def_net(x_data)

# y = tf.nn.sigmoid(net_obj.def_net(x_data))

y = tf.nn.sigmoid(y_temp)

# 通过预测的y值与真实的y_data值进行对比得出误差
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_data))
# 定义训练时采用的算法
optmizer = tf.train.GradientDescentOptimizer(0.1)
train = optmizer.minimize(loss)
acc = tf.reduce_mean(tf.cast(tf.abs(y - y_data) < 0.0001, tf.float32))

obj_acc = multi_accuracy.multi_accuracy()
acc_total, accu, precision, recall, acc_list, precision_list, recall_list = obj_acc.def_accuracy(y_temp, y_data)
obj_de_acc = default_accuracy.default_accuracy()
acc_default = obj_de_acc.def_accuracy(y_temp, y_data)
saver = tf.train.Saver()

with tf.Session() as sess:
    if not is_training:
        saver.restore(sess, sess_path)
    else:
        # 初始化相关变量
        init = tf.global_variables_initializer()
        sess.run(init)
    # 训练无数次，每训练1000次，输出一次训练的结果
    i = 0
    while True:
        xdata, ydata = get_rand_np()
        # if is_training:
        #     _, loss_, acc_, y_, acc_total_, acc_list_, precision_, recall_, acc_de_ = sess.run(
        #         [train, loss, acc, y, acc_total, acc_list, precision, recall, acc_default],
        #         feed_dict={x_data: xdata, y_data: ydata})
        # else:
        loss_, acc_, y_, acc_total_, accu_, precision_, recall_, acc_list_, prelist_, recall_list_, acc_de_ = sess.run(
            [loss, acc, y, acc_total, accu, precision, recall, acc_list, precision_list, recall_list, acc_default],
            feed_dict={x_data: xdata, y_data: ydata})
        i = i + 1
        if i % 100 == 0:
            print("[step: %f][loss: %f][acc: %f]" % (i, loss_, acc_))

            print('[precision: %f][recall: %f][acc_default: %f]'
                  % (precision_, recall_, acc_de_))

            print('acc_total_:', acc_total_)
            print('accu:', accu_)
            # print('acc_listm:', acc_list_)
            # print('precision_list:', prelist_)
            # print('recall_list:', recall_list_)
            #
            # print('y_prediction:', y_, ' y_data:', ydata)

            # for k in range(10):
            #     print('x:', xdata[k][0][0][0], 'y:', y_[k][0])

            if acc_ == 1.0 and is_training:
                print('准确率100%, 保存模型， 结束训练')
                saver.save(sess, sess_path)
                break


