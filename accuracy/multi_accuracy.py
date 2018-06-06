import tensorflow as tf
import utils.data_helper as data_helper


class multi_accuracy:
    def __init__(self):
        print('choose multi_accuracy')

    def def_accuracy(self, inputs, labels):
        '''
        get the accuracy_total, acc_list, precision and recall of the inputs
        :param inputs:
        :param labels:
        :return:
        acc_total: the perfect matching accuracy
        acc_list: the accuracy of each material
        precision: precision
        recall: recall
        '''
        inputs = tf.sigmoid(inputs)
        in_shape = inputs.get_shape().as_list()
        inputs = tf.cast(inputs > 0.5, tf.float32)
        acc_total = tf.reduce_mean(tf.cast(tf.reduce_sum(tf.abs(labels - inputs), axis=1) < 0.1, tf.float32))
        acc_list = tf.divide(tf.reduce_sum(tf.cast(tf.abs(labels - inputs) < 0.1, tf.float32), axis=0), in_shape[1])
        precision = tf.divide(
            tf.reduce_sum(inputs) - tf.reduce_sum(tf.cast(inputs - labels >= 1, tf.float32)), tf.reduce_sum(inputs))
        recall = tf.divide(
            tf.reduce_sum(labels) - tf.reduce_sum(tf.cast(labels - inputs >= 1, tf.float32)), tf.reduce_sum(labels))
        return acc_total, acc_list, precision, recall


if __name__ == '__main__':
    print('testing multi_accuracy')