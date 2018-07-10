import tensorflow as tf
import utils.data_helper

class class2_accuracy:
    def __init__(self):
        print('choose class2_accuracy')
    def def_accuracy(self,inputs, labels):
        with tf.name_scope("test"):
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

            inputs_m=tf.cast(inputs > 0.5, tf.float32)

            #inputs = tf.Print(inputs, [mask], "masks: ", summarize=60)
            #labels_m = tf.Print(labels_m, [labels_m], "label: ", summarize=60)
            #inputs_m = tf.Print(inputs_m, [inputs_m], "predi: ", summarize=60)

            inputs_m = inputs_m * mask
            #because of the impact of mask. we consider the negative case, because all masked bit is positive
            #it means we calculate the error rate install of right rate
            accuracy = 1-tf.reduce_mean(tf.reduce_sum(tf.cast(tf.abs(labels_m - inputs_m)> 0.9, tf.float32), axis=1))
            precision = tf.divide(tf.reduce_sum(inputs_m) - tf.reduce_sum(tf.cast(inputs_m - labels_m >= 1, tf.float32)), tf.reduce_sum(inputs_m)+0.000001)
            recall = tf.divide(tf.reduce_sum(labels_m) - tf.reduce_sum(tf.cast(labels_m - inputs_m >= 1, tf.float32)), tf.reduce_sum(labels_m)+0.000001)

            loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(logits=inputs, labels=labels_m)
            loss_vec = tf.multiply(loss_vec, mask, name='chamo_mul')
            loss = tf.reduce_mean(loss_vec)

            #ema = tf.train.ExponentialMovingAverage(decay=0.9)

            #avg_op = ema.apply([accuracy, precision, recall])
            #precision_avg=ema.average(precision)
            #recall_avg = ema.average(recall)
            #accuracy_avg = ema.average(accuracy)
            f1 = tf.divide(2 * precision * recall, precision + recall+0.000001)
            #accuracy_avg = tf.Print(accuracy_avg, [accuracy_avg, precision_avg, recall_avg, f1], '[acc][prec][recall][f1]')
            tf.summary.histogram("predictions", inputs)
            tf.summary.scalar('f1_test', f1)
            tf.summary.scalar('accuracy_test', accuracy)
            tf.summary.scalar('precision_test', precision)
            tf.summary.scalar('recall_test', recall)
            tf.summary.scalar('loss', loss)

        return accuracy
