import tensorflow as tf
import utils.data_helper
import utils.global_var

def preprocess_for_train(image,
                         output_height,
                         output_width):
    resize_side = tf.random_uniform(
        [],
        minval=utils.global_var._RESIZE_SIDE_MIN,
        maxval=utils.global_var._RESIZE_SIDE_MIN+100,
        dtype=tf.int32
    )
    image = utils.data_helper._aspect_preserving_resize(image, resize_side)
    image = utils.data_helper._random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)
    return utils.data_helper._mean_image_subtraction(image)

class default_preprocess:
    tfrecord_addr=None
    batchsize=None
    class_num=None
    def __init__(self,tfrecord_addr, batchsize, class_num):
        print('choose default_preprocess')
        self.batchsize=batchsize
        self.tfrecord_addr =tfrecord_addr
        self.class_num=class_num
    def def_preposess(self):
        image, label = utils.data_helper.get_raw_img(self.tfrecord_addr, self.class_num)
        train_image_size = utils.global_var._RESIZE_SIDE_MIN
        image = preprocess_for_train(image, train_image_size, train_image_size)
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=self.batchsize,
            num_threads=1,
            capacity=40 * self.batchsize,
            min_after_dequeue=38 * self.batchsize
        )
        return images, labels