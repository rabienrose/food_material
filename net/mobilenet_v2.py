import net.mobilenet.mobilenet_v2 as mnv2
import tensorflow as tf
import tensorflow.contrib.slim as slim


class mobilenet_v2:
    num_classes=None
    is_training = None
    spatial_squeeze = None
    scope = None
    fc_conv_padding = None
    global_pool = None
    def __init__(self, is_training, scope, num_classes):
        print('choose MobileNet_V2')
        self.num_classes = num_classes
        self.is_training = is_training
        self.spatial_squeeze = True
        self.scope = scope
        self.fc_conv_padding = 'VALID'
        self.global_pool = False

    def def_net(self, inputs):
        #with tf.contrib.slim.arg_scope(mnv2.training_scope(is_training=self.is_training)):
        logits, endpoint = mnv2.mobilenet(inputs, num_classes=self.num_classes, reuse=tf.AUTO_REUSE)
        return logits
            


