import tensorflow as tf
import time
import os
class default_eval:
    max_step=None
    result_addr=None
    def __init__(self,max_step, result_addr ):
        self.max_step=max_step
        self.result_addr = result_addr
        print('choose default_opt')
    def run(self, loss, test_acc):
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.result_addr)
            sess.run(init_op)
            step = 0
            while True:
                before_time = time.perf_counter()
                step += 1
                # acc_total_, acc_list_, precision_, recall_ = sess.run(
                #     test_acc)
                acc_total_ = sess.run(
                    test_acc)
                after_time = time.perf_counter()
                step_time = after_time - before_time
                train_loss_val = sess.run(loss)
                # print("[step: %f][train loss: %f][acc_total: %f][precision: %f][recall: %f][step time: %f]"
                #       % (step, train_loss_val, acc_total_, precision_, recall_, step_time))
                print("[step: %f][train loss: %f][acc_total: %f][step time: %f]"
                      % (step, train_loss_val, acc_total_, step_time))
