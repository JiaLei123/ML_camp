import tensorflow as tf

w1 = tf.Variable(tf.random_uniform([2, 3], maxval=10, minval=1), name='w1')
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name='w2')

b1 = tf.Variable(tf.zeros([3]), name='b1')
b2 = tf.Variable(tf.ones([1]), name='b2')

x = tf.constant([[0.7, 0.9]])

a = tf.matmul(x, w1) + b1
y = tf.matmul(a, w2) + b2



sess = tf.Session()
sess.run(w1.initializer)
sess.run(w2.initializer)
sess.run(tf.global_variables_initializer())
print(sess.run(y))
writer = tf.summary.FileWriter("output1", sess.graph)
writer.close()
sess.close()