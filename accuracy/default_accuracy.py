import tensorflow as tf
import utils.data_helper

class default_accuracy:
    def __init__(self):
        print('choose default_accuracy')
    def def_accuracy(self,inputs, labels):
        inputs=tf.sigmoid(inputs)
        inputs=tf.cast(inputs> 0.5, tf.float32)
        inputs=tf.cast((tf.reduce_sum(tf.abs(labels - inputs),axis=1)<0.1), tf.float32)
        accuracy = tf.reduce_mean(inputs)
        return accuracy
