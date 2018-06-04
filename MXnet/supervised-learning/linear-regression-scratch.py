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

features = nd.random.normal(scale=1, shape=(number_example, number_inputs))

labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()