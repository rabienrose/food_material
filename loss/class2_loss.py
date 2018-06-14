import tensorflow as tf
import utils.data_helper
class class2_loss:
    def __init__(self):
        print('choose class2_loss')
    def def_loss(self,inputs, labels):
        labels_m=labels-tf.floor(labels/2)*2
        mask=tf.floor(labels/2)
        mask=tf.gather(mask,tf.constant([0]))
        mask=tf.squeeze(mask)
        idx= tf.where(mask > 0.5)
        idx = tf.squeeze(idx)
        labels_m = tf.gather(labels_m, idx, axis=1)
        inputs = tf.gather(inputs, idx, axis=1)
        loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels_m)
        loss = tf.reduce_mean(loss_vec)
        #utils.data_helper.check_vars([labels_m,inputs,loss_vec])
        return loss
