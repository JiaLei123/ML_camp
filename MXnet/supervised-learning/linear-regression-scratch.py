import matplotlib as mpl
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd, nd
import numpy as np
import random
import sys

# import utils


number_inputs = 2
number_example = 100000
true_w = nd.array([2, -3.4]).reshape((2, 1))
true_b = 4.2

X = nd.random.normal(scale=1, shape=(number_example, number_inputs))

y = nd.dot(X, true_w) + true_b
y += 0.01 * nd.random.normal(scale=1, shape=y.shape)

# plot
# plt.rcParams['figure.figsize'] = (3.5, 2.5)
# plt.scatter(X[:, 1].asnumpy(), y.asnumpy(), c="r", marker="v")
# plt.show()

batch_size = 10


def data_iter():
    """
    get number_example/batch_size data list and each list has batch_size data
    :return:
    """
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


def squared_loss(yhat, y):
    """
    loss function
    :param yhat:
    :param y:
    :return:
    """
    return (yhat - y.reshape(yhat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def trainer():
    lr = 0.05
    num_round = 10

    for epoch in range(1, num_round + 1):
        for X, y in data_iter():
            with autograd.record():
                output = hypothesis_function(X, w, b)
                loss = squared_loss(output, y)
            loss.backward()
            sgd([w, b], lr, batch_size)
        print("epoch %d, loss: %f" % (epoch, squared_loss(hypothesis_function(X, w, b), y).mean().asnumpy()))


trainer()
print(true_w, w)
print(true_b, b)