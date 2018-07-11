from mxnet.gluon import nn
from mxnet import gluon
from MXnet import utils
from mxnet import init

num_epochs = 10

net = nn.HybridSequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=96, kernel_size=11, strides=4, activation="relu"),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(channels=256, kernel_size=5, padding=2, activation="relu"),
        nn.MaxPool2D(pool_size=3, strides=2),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation="relu"),
        nn.Conv2D(channels=384, kernel_size=3, padding=1, activation="relu"),
        nn.Conv2D(channels=256, kernel_size=3, padding=1, activation="relu"),
        nn.MaxPool2D(pool_size=3, strides=2),

        nn.Flatten(),
        nn.Dense(4096, activation="relu"),
        # nn.Dropout(0.5),
        nn.Dense(4096, activation="relu"),
        # nn.Dropout(0.5),
        nn.Dense(10)
    )

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
batch_size = 64
train_data, test_data = utils.load_data_fashion_mnist(batch_size, resize=224)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
utils.train_wiht_V(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs, logpath="./alex_net_no_dropout")


