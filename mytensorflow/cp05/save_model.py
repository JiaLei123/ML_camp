from time import sleep

import tensorflow as tf


def saver_pro():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, "./save/my_model.ckpt")


def restore_pro():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "./save/my_model.ckpt")
        print(sess.run(result))


if __name__ == "__main__":
    saver_pro()
    sleep(100)
    restore_pro()
