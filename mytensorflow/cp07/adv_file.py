import tensorflow as tf

input_files = tf.placeholder(tf.string)
data_set = tf.data.TextLineDataset(input_files)
iterator = data_set.make_initializable_iterator()
x = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={
        input_files: ["E:\\ML_learning\\Daguan\\data\\train_data_word_seg_fastTest.tsv",
                      "E:\\ML_learning\\Daguan\\data\\test_data_all_word_seg_fastTest.tsv"]})
    while True:
        try:
            print(len(sess.run(x)))
        except tf.errors.OutOfRangeError:
            break
