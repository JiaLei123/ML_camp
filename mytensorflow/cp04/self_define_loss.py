import tensorflow as tf
from mytensorflow.cp04.util import get_train_data, tainer

batch_size = 8
dataset_size = 128

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1), name='w1')

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')


y = tf.matmul(x, w1)

loss_less = 1
loss_more = 10
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

X, Y = get_train_data(dataset_size)

STEPS = 5000


tainer(STEPS, batch_size, dataset_size, train_step, X, Y, loss, x, y_)