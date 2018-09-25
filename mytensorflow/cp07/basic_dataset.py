import tensorflow as tf

input_data = list(range(1000))
data_set = tf.data.Dataset.from_tensor_slices(input_data)
iterator = data_set.make_one_shot_iterator()
x = iterator.get_next()
y = x * x

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))
