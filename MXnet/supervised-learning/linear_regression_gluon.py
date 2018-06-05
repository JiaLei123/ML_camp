import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np

number_inputs = 2
number_example = 100000
true_w = nd.array([2, -3.4]).reshape((2, 1))
true_b = 4.2

X = nd.random.normal(scale=1, shape=(number_example, number_inputs))

y = nd.dot(X, true_w) + true_b
y += 0.01 * nd.random.normal(scale=1, shape=y.shape)

batch_size = 10
dataset = gdata.ArrayDataset(X, y)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize()
square_loss = gloss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            output = net(X)
            loss = square_loss(output, y)
        loss.backward()
        trainer.step(batch_size)
    print("epoch %d, loss: %f" % (epoch, square_loss(net(X), y).mean().asnumpy()))

dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())