import tensorflow as tf
slim = tf.contrib.slim
class default_opt:
    name=None
    def __init__(self):
        self.name ='default'
    def run(self, loss):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            #writer = tf.summary.FileWriter("logs/", sess.graph)
            for i in range(1000):
                sess.run(train_step)
                if i % 1 == 0:
                    print(sess.run(loss))
            coord.request_stop()
            coord.join(threads)