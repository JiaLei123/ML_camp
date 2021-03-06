from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from MXnet import utils
from mxnet import init

net = nn.HybridSequential()
drop_prob1 = 0.2
drop_prob2 = 0.5

with net.name_scope():
    net.add(nn.Flatten())
    net.add(gluon.nn.Dense(56*56, activation="relu"))
    # net.add(nn.Dropout(drop_prob1))
    net.add(gluon.nn.Dense(28*28, activation="relu"))
    # net.add(nn.Dropout(drop_prob1))
    net.add(gluon.nn.Dense(512, activation="relu"))
    # net.add(nn.Dropout(drop_prob1))
    net.add(gluon.nn.Dense(256, activation="relu"))
    # net.add(nn.Dropout(drop_prob1))
    net.add(gluon.nn.Dense(10))


ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
# net.initialize()
ctx = utils.try_gpu()
batch_size = 256
num_epochs = 30
weight_decay = 0.001
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.2, 'wd': weight_decay})
utils.train_wiht_V(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs, logpath="./mpl_deep_weight_decay")
