import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


'''
message Example {
    Features features
}

message Features{
    map<String, Feature> feature
}

message Feature{
    oneof kind {
        BytesList
        FloatList
        Int64List
    }
}

'''




# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets("/path/to/mnist/data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = mnist.train.num_exampels

filename = "path/to/output.tfrecords"
# 创建一个writer来写tf文件
writer = tf.python_io.TFRecordWriter(filename)

# 将mnist数据集中所有的训练数据存储到一个TFRecord文件中
for index in range(num_examples):
    # 将图像矩阵转换为一个字符串
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))

    writer.write(example.SerializeToString())
writer.close()
