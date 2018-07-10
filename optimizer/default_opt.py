import tensorflow as tf
import time
import os
slim = tf.contrib.slim
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
        tvars = tf.trainable_variables()
        g_vars = [var for var in tvars if 'MobilenetV2/Logits' in var.name]
        print(g_vars)
        optimizer = tf.train.AdamOptimizer(1e-4)
        #train_step = slim.learning.create_train_op(loss,optimizer,update_ops=g_vars)
        train_step = slim.learning.create_train_op(loss, optimizer)
        init_op = tf.global_variables_initializer()
        exclude = ['test','MobilenetV2/Logits']

        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            if self.loading_his:
                saver.restore(sess, self.result_addr+self.ckpt_name+'/chamo.ckpt')
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("logs/", sess.graph)
            i =-1
            while True:
                before_time = time.perf_counter()
                i=i+1
                [_, train_loss_val]=sess.run([train_step,loss])
                after_time = time.perf_counter()
                step_time = after_time - before_time
                if i % self.debug_step_len == 0:

                    summary=sess.run(merge)
                    writer.add_summary(summary, i)
                    print("[step: %f][loss: %f][step time: %f]" % (i,train_loss_val,step_time,))
                    if i % (self.debug_step_len*20) == 0:
                        output_name=self.result_addr+'chamo_%f_%f' % (i, train_loss_val)
                        os.system('mkdir '+output_name)
                        saver.save(sess, output_name+'/chamo.ckpt')
            coord.request_stop()
            coord.join(threads)
