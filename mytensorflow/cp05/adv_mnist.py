import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
LAYER2_NODE = 100
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, w1, b1, w2, b2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, w1) + b1)
        return tf.matmul(layer1, w2) + b2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(w1)) + avg_class.average(b1))
        return tf.matmul(layer1, avg_class.average(w2)) + avg_class.average(b2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    w1 = tf.get_variable('w1', initializer=tf.truncated_normal_initializer(stddev=0.1), shape=[INPUT_NODE, LAYER1_NODE])
    b1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    w2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1), name='w2')
    b2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, w1, b1, w2, b2)

    global_step = tf.Variable(0, trainable=False)

    varialbe_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = varialbe_average.apply(tf.trainable_variables())
    average_y = inference(x, varialbe_average, w1, b1, w2, b2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(w1) + regularizer(w2)
    loss = cross_entropy_mean + regularization

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average mode is %g" %(i, validate_acc))
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average mode is %g" % (TRAINING_STEP, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("../data", one_hot=True)
    train(mnist)

if __name__=="__main__":
    main()
    # tf.app.run()