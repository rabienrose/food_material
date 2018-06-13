import net.mobilenet.mobilenet_v2 as mnv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import copy
from net.mobilenet import conv_blocks as ops
import net.mobilenet.mobilenet as mn


expand_input = ops.expand_input_by_factor

class mobilenet_v2_impl:
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
        return train(inputs, self.scope, self.num_classes, is_training=self.is_training, reuse=tf.AUTO_REUSE)

    def def_net_1(self, inputs):
        with tf.variable_scope(self.scope, 'mobilenet_v2', [inputs], reuse=tf.AUTO_REUSE) as sc:
            with slim.arg_scope(mnv2.training_scope()):
                net, end_points = mnv2.mobilenet_base(
                    inputs,
                    conv_defs=mnv2.V2_DEF, depth_multiplier=1.0)
                net = tf.identity(net, name='embedding')

                with tf.variable_scope('Logits'):
                    net = global_pool(net)

                    if not self.is_training:
                        self.dropout_keep_prob = 1.0
                    end_points['global_pool'] = net
                    net = slim.dropout(net, scope='Dropout', keep_prob=self.dropout_keep_prob)
                    # 1 x 1 x num_classes
                    # Note: legacy scope name.
                    logits = slim.conv2d(
                         net,
                         self.num_classes, [1, 1],
                         activation_fn=None,
                         normalizer_fn=None,
                         biases_initializer=tf.zeros_initializer(),
                         scope='Conv2d_1c_1x1')

                    logits = tf.squeeze(logits, [1, 2])

                    logits = tf.identity(logits, name='output')
                end_points['Logits'] = logits
                # if prediction_fn:
                #     end_points['Predictions'] = prediction_fn(logits, 'Predictions')
        return logits, end_points


def global_pool(inputs, pool_op=tf.nn.avg_pool):
    '''
    Applies avg pool to produce 1x1 output.
    NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
    baked in average pool which has better support across hardware.
    :param input: input tensor
    :param pool_op: the pool stratage, avg is default
    :return:
    a tensor as [batch_size, 1, 1, depth]
    '''
    input_shape = inputs.get_shape().as_list()

    if input_shape[1] is None or input_shape[2] is None:
        kernel_size = tf.convert_to_tensor(
            [1, tf.shape(inputs)[1], tf.shape(inputs)[2], 1])
    else:
        kernel_size = [1, input_shape[1], input_shape[2], 1]

    output = pool_op(inputs, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
    output.set_shape([None, 1, 1, None])
    return output









def train(inputs, scope, num_classes, is_training=True, reuse=tf.AUTO_REUSE):
    with tf.contrib.slim.arg_scope(mnv2.training_scope(is_training=is_training)):
            input_shape = inputs.get_shape().as_list()
            if len(input_shape) != 4:
                raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

            with tf.variable_scope(scope, 'Mobilenet', reuse=reuse) as scope:
                inputs = tf.identity(inputs, 'input')

                net = slim.conv2d(inputs, stride=2, num_outputs=32, kernel_size=[3, 3])
                net = ops.expanded_conv(net, expansion_size=expand_input(1, divisible_by=1), num_outputs=16)
                net = ops.expanded_conv(net, stride=2, num_outputs=24)
                net = ops.expanded_conv(net, stride=1, num_outputs=24)
                net = ops.expanded_conv(net, stride=2, num_outputs=32)
                net = ops.expanded_conv(net, stride=1, num_outputs=32)
                net = ops.expanded_conv(net, stride=1, num_outputs=32)
                net = ops.expanded_conv(net, stride=2, num_outputs=64)
                net = ops.expanded_conv(net, stride=1, num_outputs=64)
                net = ops.expanded_conv(net, stride=1, num_outputs=64)
                net = ops.expanded_conv(net, stride=1, num_outputs=64)
                net = ops.expanded_conv(net, stride=1, num_outputs=96)
                net = ops.expanded_conv(net, stride=1, num_outputs=96)
                net = ops.expanded_conv(net, stride=1, num_outputs=96)
                net = ops.expanded_conv(net, stride=2, num_outputs=160)
                net = ops.expanded_conv(net, stride=1, num_outputs=160)
                net = ops.expanded_conv(net, stride=1, num_outputs=160)
                net = ops.expanded_conv(net, stride=1, num_outputs=320)
                net = slim.conv2d(net, stride=1, kernel_size=[1, 1], num_outputs=1280)

                net = tf.identity(net, name='embedding')

                with tf.variable_scope('Logits'):
                    net = global_pool(net)
                    #end_points['global_pool'] = net
                    net = slim.dropout(net, scope='Dropout', is_training=is_training)
                    # 1 x 1 x num_classes
                    # Note: legacy scope name.
                    logits = slim.conv2d(
                        net,
                        num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        biases_initializer=tf.zeros_initializer(),
                        scope='Conv2d_1c_1x1')

                    logits = tf.squeeze(logits, [1, 2])

                    logits = tf.identity(logits, name='output')
                #end_points['Logits'] = logits
            return logits


# @slim.add_arg_scope
# def mobilenet_base(  # pylint: disable=invalid-name
#     inputs,
#     conv_defs,
#     multiplier=1.0,
#     final_endpoint=None,
#     output_stride=None,
#     use_explicit_padding=False,
#     scope=None,
#     is_training=False):
#
#   if multiplier <= 0:
#     raise ValueError('multiplier is not greater than zero.')
#
#   # Set conv defs defaults and overrides.
#   conv_defs_defaults = conv_defs.get('defaults', {})
#   conv_defs_overrides = conv_defs.get('overrides', {})
#   if use_explicit_padding:
#     conv_defs_overrides = copy.deepcopy(conv_defs_overrides)
#     conv_defs_overrides[
#         (slim.conv2d, slim.separable_conv2d)] = {'padding': 'VALID'}
#
#   if output_stride is not None:
#     if output_stride == 0 or (output_stride > 1 and output_stride % 2):
#       raise ValueError('Output stride must be None, 1 or a multiple of 2.')
#
#   # a) Set the tensorflow scope
#   # b) set padding to default: note we might consider removing this
#   # since it is also set by mobilenet_scope
#   # c) set all defaults
#   # d) set all extra overrides.
#   with _scope_all(scope, default_scope='Mobilenet'), \
#       safe_arg_scope([slim.batch_norm], is_training=is_training), \
#       _set_arg_scope_defaults(conv_defs_defaults), \
#       _set_arg_scope_defaults(conv_defs_overrides):
#     # The current_stride variable keeps track of the output stride of the
#     # activations, i.e., the running product of convolution strides up to the
#     # current network layer. This allows us to invoke atrous convolution
#     # whenever applying the next convolution would result in the activations
#     # having output stride larger than the target output_stride.
#     current_stride = 1
#
#     # The atrous convolution rate parameter.
#     rate = 1
#
#     net = inputs
#     # Insert default parameters before the base scope which includes
#     # any custom overrides set in mobilenet.
#     end_points = {}
#     scopes = {}
#     for i, opdef in enumerate(conv_defs['spec']):
#       params = dict(opdef.params)
#       opdef.multiplier_func(params, multiplier)
#       stride = params.get('stride', 1)
#       if output_stride is not None and current_stride == output_stride:
#         # If we have reached the target output_stride, then we need to employ
#         # atrous convolution with stride=1 and multiply the atrous rate by the
#         # current unit's stride for use in subsequent layers.
#         layer_stride = 1
#         layer_rate = rate
#         rate *= stride
#       else:
#         layer_stride = stride
#         layer_rate = 1
#         current_stride *= stride
#       # Update params.
#       params['stride'] = layer_stride
#       # Only insert rate to params if rate > 1.
#       if layer_rate > 1:
#         params['rate'] = layer_rate
#       # Set padding
#       if use_explicit_padding:
#         if 'kernel_size' in params:
#           net = _fixed_padding(net, params['kernel_size'], layer_rate)
#         else:
#           params['use_explicit_padding'] = True
#
#       end_point = 'layer_%d' % (i + 1)
#       try:
#         net = opdef.op(net, **params)
#       except Exception:
#         print('Failed to create op %i: %r params: %r' % (i, opdef, params))
#         raise
#       end_points[end_point] = net
#       scope = os.path.dirname(net.name)
#       scopes[scope] = end_point
#       if final_endpoint is not None and end_point == final_endpoint:
#         break
#
#     # Add all tensors that end with 'output' to
#     # endpoints
#     for t in net.graph.get_operations():
#       scope = os.path.dirname(t.name)
#       bn = os.path.basename(t.name)
#       if scope in scopes and t.name.endswith('output'):
#         end_points[scopes[scope] + '/' + bn] = t.outputs[0]
#     return net, end_points
#
#
# spec=[
#         op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
#         op(ops.expanded_conv,
#            expansion_size=expand_input(1, divisible_by=1),
#            num_outputs=16),
#         op(ops.expanded_conv, stride=2, num_outputs=24),
#         op(ops.expanded_conv, stride=1, num_outputs=24),
#         op(ops.expanded_conv, stride=2, num_outputs=32),
#         op(ops.expanded_conv, stride=1, num_outputs=32),
#         op(ops.expanded_conv, stride=1, num_outputs=32),
#         op(ops.expanded_conv, stride=2, num_outputs=64),
#         op(ops.expanded_conv, stride=1, num_outputs=64),
#         op(ops.expanded_conv, stride=1, num_outputs=64),
#         op(ops.expanded_conv, stride=1, num_outputs=64),
#         op(ops.expanded_conv, stride=1, num_outputs=96),
#         op(ops.expanded_conv, stride=1, num_outputs=96),
#         op(ops.expanded_conv, stride=1, num_outputs=96),
#         op(ops.expanded_conv, stride=2, num_outputs=160),
#         op(ops.expanded_conv, stride=1, num_outputs=160),
#         op(ops.expanded_conv, stride=1, num_outputs=160),
#         op(ops.expanded_conv, stride=1, num_outputs=320),
#         op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
#     ],