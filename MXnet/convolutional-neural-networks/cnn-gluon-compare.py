from mxnet.gluon import nn
from mxnet import gluon
import matplotlib.pyplot as plt
from MXnet import utils
from MXnet.display_utils import show_loss_acc_picture, show_loss_acc_for_two_model


def cnn(num_epochs, unit_count):
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=20, kernel_size=5, activation="relu"),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=50, kernel_size=3, activation="relu"),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Flatten(),
            nn.Dense(unit_count, activation="relu"),
            nn.Dense(10)
        )
    net.initialize()
    ctx = utils.try_gpu()
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
    return utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs)


def mlp(num_epochs, unit_count):
    net = nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(unit_count, activation="relu"))
        # net.add(gluon.nn.Dense(28 * 28, activation="relu"))
        # net.add(gluon.nn.Dense(28 * 28, activation="relu"))
        # net.add(gluon.nn.Dense(28 * 28, activation="relu"))
        net.add(gluon.nn.Dense(10))
    net.initialize()
    ctx = utils.try_gpu()
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.5})
    return utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs)


num_epochs = 50
unit_count = 56 * 56
train_loss_list, test_loss_list, train_acc_list, test_acc_list = cnn(num_epochs, 128)
train_loss_2_list, test_loss_2_list, train_acc_2_list, test_acc_2_list = mlp(num_epochs, unit_count)

show_loss_acc_for_two_model(unit_count, num_epochs,
                            train_loss_list, train_loss_2_list,
                            test_loss_list, test_loss_2_list,
                            train_acc_list, train_acc_2_list,
                            test_acc_list, test_acc_2_list,
                            "cnn", "mpl")

# train_loss_2_list, test_loss_2_list, train_acc_2_list, test_acc_2_list = cnn(num_epochs, unit_count)
# show_loss_acc_for_two_model(unit_count, num_epochs,
#                             train_loss_list, train_loss_2_list,
#                             test_loss_list, test_loss_2_list,
#                             train_acc_list, train_acc_2_list,
#                             test_acc_list, test_acc_2_list,
#                             "cnn 128 unit", "mpl 3136 unit")