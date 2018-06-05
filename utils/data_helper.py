import tensorflow as tf
import matplotlib.pyplot as plt
import utils.global_var
import os

def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)
    rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3),['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])
    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs

def _random_crop(image_list, crop_height, crop_width):
    if not image_list:
        raise ValueError('Empty image_list.')
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform([], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]

def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)
    scale = tf.cond(tf.greater(height, width),lambda: smallest_side / width,lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def _mean_image_subtraction(image):
    means = utils.global_var.means
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)

def get_raw_img(tfrecord_addr, class_num):
    file_list = os.listdir(tfrecord_addr)
    tfrecord_list=[]
    for file_name in file_list:
        if file_name.find('.tfrecord'):
            tfrecord_list.append(tfrecord_addr+file_name)
    filename_queue = tf.train.string_input_producer(tfrecord_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([class_num], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64),
                                       })
    height = tf.cast(features['img_height'], tf.int32)
    width = tf.cast(features['img_width'], tf.int32)
    img_data_jpg = tf.image.decode_jpeg(features['img_raw'])
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)
    image = img_data_jpg
    channel = 3
    image = tf.reshape(image, [height, width, channel])
    label = tf.cast(features['label'], tf.float32)
    return image, label

def check_imgs(images, labels):
    img_mean = utils.global_var.means
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        image_batch_v, label_batch_v = sess.run([images, labels])
        for k in range(len(image_batch_v)):
            processed_img = image_batch_v[k]
            print(label_batch_v[k])
            show_img = processed_img + img_mean
            show_img = abs(show_img) / 256.0
            plt.imshow(show_img)
            plt.show()
        coord.request_stop()
        coord.join(threads)

def check_vars(tensor_list):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        re = sess.run(tensor_list)
        for k in re:
            print(k)
        coord.request_stop()
        coord.join(threads)

