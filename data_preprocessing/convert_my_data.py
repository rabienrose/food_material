import tensorflow as tf
import os
from PIL import Image

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

img_root="D:/material_sort_try/"
img_addrs=["califlower", "non_califlower"]
class_names_to_ids={"califlower":1, "non_califlower":-1}
image_reader = ImageReader()
sess=tf.Session()
with tf.python_io.TFRecordWriter('chamo.tfrecord') as tfrecord_writer:
    for img_addr in img_addrs:
        img_list = os.listdir(img_root+img_addr)
        for i in range(len(img_list)):
            if img_list[i].find('.jpg'):
                img = Image.open(img_root+img_addr + '/' + img_list[i], 'r')
                size = img.size
                img_raw = img.tobytes()
                class_id=class_names_to_ids[img_addr]
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[class_id])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                        'img_width':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                        'img_height':tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                    }))
                tfrecord_writer.write(example.SerializeToString())

