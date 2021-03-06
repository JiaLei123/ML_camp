import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference
import mnist_train

EVAL_INTERAVEL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        num_example = mnist.validation.images.shape[0]

        x = tf.placeholder(tf.float32, [
            num_example,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        xs = mnist.validation.images
        reshape_xs = np.reshape(xs, (
            num_example,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS
        ))

        validate_feed = {x: reshape_xs, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        y = mnist_inference.inference(x, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while (True):
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state("model")
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
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
