import tensorflow as tf
x = tf.Variable([1, 2, 0, 4])
ix = tf.transpose(tf.where(x > 1))
y = tf.gather(x, ix)
op=tf.Print(y,[y,ix],'chamo: ')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(op)