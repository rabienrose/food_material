import tensorflow as tf

class default_accuracy:
    def __init__(self):
        print('choose default_accuracy')
    def def_accuracy(self,inputs, labels):
        inputs=tf.sigmoid(inputs)
        accuracy = tf.reduce_mean(tf.cast(tf.abs(labels - inputs) < 0.4, tf.float32))
        return accuracy
