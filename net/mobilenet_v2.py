import net.mobilenet.mobilenet_v2 as mnv2
import tensorflow as tf
import tensorflow.contrib.slim as slim


class mobilenet_v2:
    num_classes=None
    is_training = None
    dropout_keep_prob = None
    spatial_squeeze = None
    scope = None
    fc_conv_padding = None
    global_pool = None
    def __init__(self, is_training, scope, num_classes):
        print('choose MobileNet_V2')
        self.num_classes = num_classes
        self.is_training = is_training
        self.dropout_keep_prob = 0.5
        self.spatial_squeeze = True
        self.scope = scope
        self.fc_conv_padding = 'VALID'
        self.global_pool = False

    def def_net(self, inputs):
        with tf.variable_scope(self.scope, 'vgg_16', [inputs], reuse=tf.AUTO_REUSE) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                logits, endpoint = mnv2.mobilenet(inputs, num_classes=self.num_classes, is_training=self.is_training)
                return logits


