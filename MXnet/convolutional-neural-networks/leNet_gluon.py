from mxnet.gluon import nn
from mxnet import gluon
from MXnet import utils
from mxnet import init

num_epochs = 20
net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Conv2D(channels=6, kernel_size=5, activation="sigmoid")),
    net.add(nn.MaxPool2D(pool_size=2, strides=2)),
    net.add(nn.Conv2D(channels=16, kernel_size=5, activation="sigmoid")),
    net.add(nn.MaxPool2D(pool_size=2, strides=2)),
    nn.Flatten(),
    net.add(nn.Dense(120, activation="sigmoid")),
    net.add(nn.Dense(84, activation="sigmoid")),
    net.add(nn.Dense(10))

batch_size = 256
ctx = utils.try_gpu()
net.initialize()
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 1})
utils.train_wiht_V(train_data, test_data, net, loss, trainer, ctx, num_epochs, logpath="LeNet")
