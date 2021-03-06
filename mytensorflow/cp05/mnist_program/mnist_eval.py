import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERAVEL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        # 获取到滑动平均的变量映射表，用于回复变量和影子变量
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while (True):
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state() 在一个folder中加载保存的最新的model
                ckpt = tf.train.get_checkpoint_state("model")
                if ckpt and ckpt.model_checkpoint_path:
                    # 获取这个备份文件的path
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过解析文件名来获取step的值
                    global_step = ckpt.model_checkpoint_path.split("\\")[-1].split("-")[-1]
                    accuracy_store = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy using average mode is %g" % (
                    global_step, accuracy_store))
                else:
                    print("no checkpoint found")
                    return
            time.sleep(EVAL_INTERAVEL_SECS)


def main():
    mnist = input_data.read_data_sets("../../data", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    main()
