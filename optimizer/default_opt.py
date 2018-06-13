import tensorflow as tf
import time
import os
class default_opt:
    max_step=None
    debug_step_len=None
    result_addr=None
    stop_accu=None
    loading_his=None
    ckpt_name=None
    def __init__(self,max_step, debug_step_len, result_addr, stop_accu, loading_his, ckpt_name):
        self.max_step=max_step
        self.debug_step_len=debug_step_len
        self.result_addr = result_addr
        self.stop_accu = stop_accu
        self.loading_his = loading_his
        self.ckpt_name = ckpt_name
        print('choose default_opt')
    def run(self, loss, test_acc):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            if self.loading_his:
                saver.restore(sess, self.result_addr+self.ckpt_name+'/chamo.ckpt')
            writer = tf.summary.FileWriter("logs/", sess.graph)
            i =-1
            while True:
                before_time = time.perf_counter()
                i=i+1
                sess.run(train_step)
                after_time = time.perf_counter()
                step_time = after_time - before_time
                if i % self.debug_step_len == 0:
                    test_acc_val=sess.run(test_acc)
                    train_loss_val = sess.run(loss)
                    print("[step: %f][train loss: %f][perf accu: %f][accu: %f][prec: %f][recall: %f][step time: %f]" % (i, train_loss_val, test_acc_val[0], test_acc_val[1], test_acc_val[2], test_acc_val[3], step_time))
                    if i % (self.debug_step_len*20) == 0:
                        output_name=self.result_addr+'chamo_%f_%f_%f' % (i, train_loss_val, test_acc_val[0])
                        os.system('mkdir '+output_name)
                        saver.save(sess, output_name+'/chamo.ckpt')
            coord.request_stop()
            coord.join(threads)
