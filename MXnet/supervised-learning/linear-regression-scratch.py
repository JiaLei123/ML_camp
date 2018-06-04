import matplotlib as mpl
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, nd
import numpy as np
import random
import sys

# import utils


number_inputs = 2
number_example = 1000
true_w = nd.array([2, -3.4]).reshape((2, 1))
true_b = 4.2

X = nd.random.normal(scale=1, shape=(number_example, number_inputs))

y = nd.dot(X, true_w) + true_b
y += nd.random.normal(scale=0.01, shape=y.shape)

# plot
# plt.rcParams['figure.figsize'] = (3.5, 2.5)
# plt.scatter(X[:, 1].asnumpy(), y.asnumpy(), c="r", marker="v")
# plt.show()

batch_size = 10


def data_iter():
    idx = list(range(number_example))
    random.shuffle(idx)
    for i in range(0, number_example, batch_size):
        j = nd.array(idx[i: min(i + batch_size, number_example)])
        yield X.take(j), y.take(j)


for data, label in data_iter():
    print(data, label)
    break

w = nd.random.normal(scale=1, shape=(number_inputs, 1))
b = nd.zeros(1)
params = [w, b]

for param in params:
    param.attach_grad()


def hypothesis_function(X, w, b):
    """
    hypothesis function
    :param X:
    :param w:
    :param b:
    :return:
    """
    return nd.dot(X, w) + b
