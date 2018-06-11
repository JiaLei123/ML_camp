from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn
from MXnet import utils
import matplotlib as mpl

from MXnet.display_utils import show_loss_acc_for_two_model

mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt


def mlp(num_epochs, unit_count, hidden_layer_num=1):
    net = nn.Sequential()
    with net.name_scope():
        for _ in range(hidden_layer_num):
            net.add(gluon.nn.Dense(unit_count, activation="relu"))
        net.add(gluon.nn.Dense(10))
    net.initialize()
    ctx = utils.try_gpu()
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.5})
    return utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs)


num_epochs = 50
unit_count = 28 * 28
unit_count_2 = 56 * 56

train_loss_list, test_loss_list, train_acc_list, test_acc_list = mlp(num_epochs, unit_count)
train_loss_2_list, test_loss_2_list, train_acc_2_list, test_acc_2_list = mlp(num_epochs, unit_count_2)

show_loss_acc_for_two_model(unit_count, num_epochs,
                            train_loss_list, train_loss_2_list,
                            test_loss_list, test_loss_2_list,
                            train_acc_list, train_acc_2_list,
                            test_acc_list, test_acc_2_list,
                            "cnn 784 unit", "mpl 3136 unit")
