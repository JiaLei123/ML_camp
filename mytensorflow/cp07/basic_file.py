import tensorflow as tf

input_files = ["E:\\ML_learning\\Daguan\\data\\train_data_word_seg_fastTest.tsv", "E:\\ML_learning\\Daguan\\data\\test_data_all_word_seg_fastTest.tsv"]
data_set = tf.data.TextLineDataset(input_files)
iterator = data_set.make_one_shot_iterator()
x = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            print(len(sess.run(x)))
        except tf.errors.OutOfRangeError:
            break