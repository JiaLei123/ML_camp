import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np

from MXnet import utils

number_inputs = 2
number_example = 10000
true_w = nd.array([2, -3.4]).reshape((2, 1))
true_b = 4.2

X = nd.random.normal(scale=1, shape=(number_example, number_inputs))

y = nd.dot(X, true_w) + true_b
y += 0.01 * nd.random.normal(scale=1, shape=y.shape)

batch_size = 5
dataset = gdata.ArrayDataset(X, y)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(1))
net.initialize()
square_loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})

num_epochs = 10
ctx = utils.try_gpu()
utils.train_1(data_iter, data_iter, net, square_loss, trainer, ctx, num_epochs=num_epochs, batch_size=batch_size)

#
# for epoch in range(1, num_epochs + 1):
#     train_loss = 0
#     for X, y in data_iter:
#         with autograd.record():
#             output = net(X)
#             loss = square_loss(output, y)
#         loss.backward()
#         trainer.step(batch_size)
#         train_loss += nd.mean(loss).asscalar()
#
#     train_loss = train_loss / len(data_iter)
#     print("epoch %d, loss: %f" % (epoch, square_loss(net(X), y).mean().asnumpy()))
#     # print("epoch %d, loss: %f" % (epoch, train_loss))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())