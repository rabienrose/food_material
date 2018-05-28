import tensorflow as tf
slim = tf.contrib.slim
class vgg16:
    num_classes=None
    is_training = None
    dropout_keep_prob = None
    spatial_squeeze = None
    scope = None
    fc_conv_padding = None
    global_pool = None
    def __init__(self):
        self.num_classes = 1
        self.is_training = True
        self.dropout_keep_prob = 0.5
        self.spatial_squeeze = True
        self.scope = 'vgg_16'
        self.fc_conv_padding = 'VALID'
        self.global_pool = False

    def def_net(self,inputs):
        with tf.variable_scope(self.scope, 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=self.fc_conv_padding, scope='fc6')
                net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if self.num_classes:
                    net = slim.dropout(net, self.dropout_keep_prob, is_training=self.is_training,scope='dropout7')
                    net = slim.conv2d(net, self.num_classes, [1, 1],activation_fn=None,normalizer_fn=None,scope='fc8')
                    end_points[sc.name + '/fc8'] = net
                net = tf.tanh(inputs)
                net = tf.squeeze(net)
                return net
