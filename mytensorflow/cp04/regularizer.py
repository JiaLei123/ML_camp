import tensorflow as tf

from mytensorflow.cp04.util import get_train_data, tainer


def get_weight(shape, lam):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lam)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
batch_size = 8

layer_dimension = [2, 10, 10, 10, 1]
n_layers = len(layer_dimension)

cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers):
    out_dimension = layer_dimension[i]
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    in_dimension = layer_dimension[i]

mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection("losses", mes_loss)

loss = tf.add_n(tf.get_collection("losses"))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


STEPS = 5000
dataset_size = 128

X, Y = get_train_data(dataset_size)


tainer(STEPS, batch_size, dataset_size, train_step, X, Y, loss, x, y_)