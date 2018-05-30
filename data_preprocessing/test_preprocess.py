import tensorflow as tf
import utils.data_helper
_RESIZE_SIDE_MIN = 224
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

def preprocess_for_train(image,
                         output_height,
                         output_width):
    image = utils.data_helper._aspect_preserving_resize(image, _RESIZE_SIDE_MIN)
    image = utils.data_helper._central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return utils.data_helper._mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

class test_preprocess:
    tfrecord_addr=None
    class_num = None
    def __init__(self,tfrecord_addr, class_num):
        print('choose test_preprocess')
        self.tfrecord_addr =tfrecord_addr
        self.class_num = class_num
    def def_preposess(self):
        c = 0
        for record in tf.python_io.tf_record_iterator(self.tfrecord_addr):
            c += 1
        image, label = utils.data_helper.get_raw_img(self.tfrecord_addr, self.class_num)
        train_image_size = _RESIZE_SIDE_MIN
        image = preprocess_for_train(image, train_image_size, train_image_size)
        images, labels = tf.train.batch([image, label], batch_size=c, num_threads=1,capacity=c*5)
        return images, labels