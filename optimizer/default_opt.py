import tensorflow as tf
import time
class default_opt:
    max_step=None
    debug_step_len=None
    result_addr=None
    stop_accu=None
    def __init__(self,max_step, debug_step_len, result_addr, stop_accu):
        self.max_step=max_step
        self.debug_step_len=debug_step_len
        self.result_addr = result_addr
        self.stop_accu = stop_accu
        print('choose default_opt')
    def run(self, loss, test_acc, labels, inputs):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            #writer = tf.summary.FileWriter("logs/", sess.graph)
            for i in range(self.max_step):
                before_time = time.perf_counter()
                sess.run(train_step)
                after_time = time.perf_counter()
                step_time = after_time - before_time
                if i % self.debug_step_len == 0:
                    test_acc_val, labels_val, inputs_val=sess.run([test_acc,labels, inputs])
                    train_loss_val = sess.run(loss)
                    print("[step: %f][train loss: %f][test accu: %f][step time: %f]" % (i, train_loss_val, test_acc_val, step_time))
                    print(labels_val)
                    print(inputs_val)
                    if(test_acc_val>self.stop_accu):
                        saver.save(sess, self.result_addr)
            coord.request_stop()
            coord.join(threads)