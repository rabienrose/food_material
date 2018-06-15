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
        acc_perfect = tf.reduce_mean(tf.cast(tf.reduce_sum(tf.abs(labels - inputs), axis=1) < 0.1, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, inputs), tf.float32))
        precision = tf.divide(
            tf.reduce_sum(inputs) - tf.reduce_sum(tf.cast(inputs - labels >= 1, tf.float32)), tf.reduce_sum(inputs))
        recall = tf.divide(
            tf.reduce_sum(labels) - tf.reduce_sum(tf.cast(labels - inputs >= 1, tf.float32)), tf.reduce_sum(labels))
        f1 = tf.divide(2 * precision * recall, precision + recall)
        acc_list = tf.divide(tf.reduce_sum(tf.cast(tf.equal(inputs, labels), tf.float32), axis=0), in_shape[0])
        pre_list = tf.divide(
            tf.reduce_sum(inputs, axis=0) - tf.reduce_sum(tf.cast(inputs - labels >= 1, tf.float32), axis=0),
            tf.reduce_sum(inputs, axis=0))
        pre_list_nume = tf.reduce_sum(inputs, axis=0) \
                        - tf.reduce_sum(tf.cast(inputs - labels >= 1, tf.float32), axis=0)
        pre_list_deno = tf.reduce_sum(inputs, axis=0)
        recall_list = tf.divide(
            tf.reduce_sum(labels, axis=0) - tf.reduce_sum(tf.cast(labels - inputs >= 1, tf.float32), axis=0),
            tf.reduce_sum(labels, axis=0))
        recall_list_nume = tf.reduce_sum(labels, axis=0)\
                           - tf.reduce_sum(tf.cast(labels - inputs >= 1, tf.float32), axis=0)
        reacall_list_deno = tf.reduce_sum(labels, axis=0)
        return [acc_perfect, accuracy, precision, recall, f1, acc_list, pre_list, pre_list_nume, pre_list_deno,
                recall_list, recall_list_nume, reacall_list_deno]


if __name__ == '__main__':
    print('testing multi_accuracy')
