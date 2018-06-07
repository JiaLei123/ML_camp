import sys
from MXnet import utils
from mxnet import ndarray as nd

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
# data shape (256, 1, 28 ,28)

# 初始化参数
num_inputs = 28 * 28
num_outputs = 10

num_hidden = 256
weight_scale = 0.01

W1 = nd.random.normal(scale=weight_scale, shape=(num_inputs, num_hidden))
b1 = nd.zeros(num_hidden)


def relu(X):
    return nd.maximum(X, 0)