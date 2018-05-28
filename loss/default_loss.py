import tensorflow as tf
slim = tf.contrib.slim
class default_loss:
    def __init__(self):
        name='test_loss'
    def def_loss(self,inputs, labels):
        loss = tf.reduce_sum(tf.square(labels - inputs))
        return loss
