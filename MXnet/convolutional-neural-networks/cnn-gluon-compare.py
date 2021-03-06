from mxnet.gluon import nn
from mxnet import gluon
from MXnet import utils
from MXnet.display_utils import show_loss_acc_for_two_model


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


def cnn_big(num_epochs, unit_count):
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Conv2D(channels=50, kernel_size=3, activation="relu"),
            nn.MaxPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=80, kernel_size=3, activation="relu"),
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


if __name__ == '__main__':
    round_num = 5
    hidden_count = 256
    train_loss_list, test_loss_list, train_acc_list, test_acc_list = cnn(round_num, 256)
    train_loss_2_list, test_loss_2_list, train_acc_2_list, test_acc_2_list = cnn_big(round_num, hidden_count)

    show_loss_acc_for_two_model(hidden_count, round_num,
                                train_loss_list, train_loss_2_list,
                                test_loss_list, test_loss_2_list,
                                train_acc_list, train_acc_2_list,
                                test_acc_list, test_acc_2_list,
                                "cnn 256 unit", "cnn 784 unit")
