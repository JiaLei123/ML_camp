from mxnet.gluon import nn
from mxnet import gluon
import matplotlib.pyplot as plt
from MXnet import utils
from MXnet.display_utils import show_loss_acc_picture


def cnn(num_epochs):
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=20, kernel_size=5, activation="relu"),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=50, kernel_size=3, activation="relu"),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(128, activation="relu"),
            nn.Dense(10)
        )
    net.initialize()
    ctx = utils.try_gpu()
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    return utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs)


num_epochs = 50
train_loss_list, test_loss_list, train_acc_list, test_acc_list = cnn(num_epochs)
show_loss_acc_picture(128, num_epochs, train_loss_list, test_loss_list, train_acc_list, test_acc_list)

