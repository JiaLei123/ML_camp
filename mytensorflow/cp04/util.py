from numpy.random import RandomState
import tensorflow as tf


def get_train_data(dataset_size):
    rdm = RandomState(1)
    X = rdm.rand(dataset_size, 2)
    Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]
    return X, Y


def tainer(STEPS, batch_size, dataset_size, train_step, X, Y, loss, x, y_, w1=None):
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        if w1:
            print(sess.run(w1))

        for i in range(STEPS):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
            if i % 1000 == 0:
                total_cross_entropy = sess.run(loss, feed_dict={x: X, y_: Y})
                print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))

        if w1:
            print(sess.run(w1))

        writer = tf.summary.FileWriter("output1", sess.graph)
        writer.close()
