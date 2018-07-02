import tensorflow as tf
import utils.data_helper
import utils.global_var

def preprocess_for_train(image,
                         output_height,
                         output_width):
    image = utils.data_helper._aspect_preserving_resize(image, utils.global_var._RESIZE_SIDE_MIN)
    image = utils.data_helper._central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return utils.data_helper._mean_image_subtraction(image)

class test_preprocess:
    tfrecord_addr=None
    class_num = None
    def __init__(self,tfrecord_addr, class_num):
        print('choose test_preprocess')
        self.tfrecord_addr =tfrecord_addr
        self.class_num = class_num
    def def_preposess(self):
        image, label = utils.data_helper.get_raw_img(self.tfrecord_addr, self.class_num)
        train_image_size = utils.global_var._RESIZE_SIDE_MIN
        image = preprocess_for_train(image, train_image_size, train_image_size)
        c=200
        images, labels = tf.train.batch(
            [image, label],
            batch_size=c,
            num_threads=1,
            capacity=2 * c,

        )
        return images, labels
