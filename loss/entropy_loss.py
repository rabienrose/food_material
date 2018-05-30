import tensorflow as tf
class entropy_loss:
    def __init__(self):
        print('choose entropy_loss')
    def def_loss(self,inputs, labels):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels =labels))
        return loss
