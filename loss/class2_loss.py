import tensorflow as tf
import utils.data_helper
class class2_loss:
    def __init__(self):
        print('choose class2_loss')
    def def_loss(self,inputs, labels):
        with tf.name_scope("train"):
            #inputs = tf.Print(inputs, [inputs], "inputs: ", summarize=32)
            tf.summary.histogram("predictions", inputs)
            labels_m=labels-tf.floor(labels/2)*2
            mask=tf.floor(labels/2)
            #mask=tf.gather(mask,tf.constant([0]))
            #mask=tf.squeeze(mask)
            #idx= tf.where(mask > 0.5)
            #idx = tf.squeeze(idx)
            #labels_m = tf.gather(labels_m, idx, axis=1)
            #inputs = tf.gather(inputs, idx, axis=1)
            #labels_m = tf.Print(labels_m, [labels_m], "loss label: ", summarize=32)

            loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels_m)
            #loss_vec = tf.Print(loss_vec, [loss_vec], "loss_vec: ", summarize=32)
            #mask = tf.Print(mask, [mask], "mask: ", summarize=32)
            loss_vec = tf.multiply(loss_vec, mask, name='chamo_mul')
            loss = tf.reduce_mean(loss_vec)
            inputs_m = tf.cast(inputs >  0.5, tf.float32)
            inputs_m = inputs_m * mask
            accuracy = 1 - tf.reduce_mean(tf.reduce_sum(tf.cast(tf.abs(labels_m - inputs_m) > 0.9, tf.float32), axis=1))
            precision = tf.divide(tf.reduce_sum(inputs_m) - tf.reduce_sum(tf.cast(inputs_m - labels_m >= 1, tf.float32)),
                                  tf.reduce_sum(inputs_m) + 0.000001)
            recall = tf.divide(tf.reduce_sum(labels_m) - tf.reduce_sum(tf.cast(labels_m - inputs_m >= 1, tf.float32)),
                               tf.reduce_sum(labels_m) + 0.000001)
            f1 = tf.divide(2 * precision * recall, precision + recall + 0.000001)
            tf.summary.scalar('f1_train', f1)
            tf.summary.scalar('accuracy_train', accuracy)
            tf.summary.scalar('precision_train', precision)
            tf.summary.scalar('recall_train', recall)
            tf.summary.scalar('loss', loss)
        return loss
