import tensorflow as tf
class default_loss:
    def __init__(self):
        print('choose default_loss')
    def def_loss(self,inputs, labels):
        inputs = tf.tanh(inputs)
        loss = tf.reduce_sum(tf.square(labels - inputs))
        return loss
