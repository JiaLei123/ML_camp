import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer=tf.zeros_initializer()(shape=[2]))
    tf.add_to_collection(tf.GraphKeys.VARIABLES, v)
    print(v)

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones([2]))
    v1 = tf.constant([2.0, 3.0], name="v1")
    result = v + v1
    tf.add_to_collection(tf.GraphKeys.VARIABLES, v)
    print(result)

with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
        print(tf.get_collection(tf.GraphKeys.VARIABLES))

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
with tf.Session(graph=g2, config=config) as sess:
    tf.initialize_all_variables().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))
        print(sess.run((result)))
        print(tf.get_collection(tf.GraphKeys.VARIABLES))

print(tf.get_collection(tf.GraphKeys.VARIABLES))