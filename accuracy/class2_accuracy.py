import tensorflow as tf
import utils.data_helper

class class2_accuracy:
    def __init__(self):
        print('choose class2_accuracy')
    def def_accuracy(self,inputs, labels):
        labels_m = labels - tf.floor(labels / 2) * 2
        mask = tf.floor(labels / 2)
        mask = tf.gather(mask, tf.constant([0]))
        mask = tf.squeeze(mask)
        idx = tf.where(mask > 0.5)
        idx = tf.squeeze(idx)
        labels_m = tf.gather(labels_m, idx, axis=1)
        inputs = tf.gather(inputs, idx, axis=1)
        inputs=tf.cast(inputs> 0.5, tf.float32)
        inputs=tf.cast((tf.reduce_sum(tf.abs(labels_m - inputs),axis=0)<0.1), tf.float32)
        accuracy = tf.reduce_mean(inputs)
        accuracy = tf.Print(accuracy, [accuracy], 'accuracy: ')
        return accuracy
