import tensorflow as tf
import utils.data_helper

class class2_accuracy:
    def __init__(self):
        print('choose class2_accuracy')
    def def_accuracy(self,inputs, labels):
        labels_m = labels - tf.floor(labels / 2) * 2
        mask = tf.floor(labels / 2)
        # mask = tf.Print(mask, [mask], "mask: ", summarize=32)
        # mask = tf.gather(mask, tf.constant([0]))
        # mask = tf.Print(mask, [mask], "mask1: ", summarize=32)
        # mask = tf.squeeze(mask)
        # idx = tf.where(mask > 0.5)
        # idx = tf.squeeze(idx)
        # labels_m = tf.gather(labels_m, idx, axis=1)
        # inputs = tf.gather(inputs, idx, axis=1)

        inputs=tf.cast(inputs> 0.5, tf.float32)

        #inputs = tf.Print(inputs, [mask], "masks: ", summarize=60)
        #inputs = tf.Print(inputs, [labels_m], "label: ", summarize=60)
        #inputs = tf.Print(inputs, [inputs], "predi: ", summarize=60)

        inputs = inputs * mask
        #because of the impact of mask. we consider the negative case, because all masked bit is positive
        #it means we calculate the error rate install of right rate
        accuracy = 1-tf.reduce_mean(tf.reduce_sum(tf.cast(tf.abs(labels_m - inputs)> 0.9, tf.float32), axis=1))
        precision = tf.divide(tf.reduce_sum(inputs) - tf.reduce_sum(tf.cast(inputs - labels_m >= 1, tf.float32)), tf.reduce_sum(inputs)+0.000001)
        recall = tf.divide(tf.reduce_sum(labels_m) - tf.reduce_sum(tf.cast(labels_m - inputs >= 1, tf.float32)), tf.reduce_sum(labels_m)+0.000001)
        f1 = tf.divide(2 * precision * recall, precision + recall)
        accuracy = tf.Print(accuracy, [accuracy, precision, recall, f1], '[acc][prec][recall][f1]')
        return accuracy
