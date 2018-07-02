from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet.gluon import nn
from MXnet import utils
import matplotlib as mpl

from MXnet.display_utils import show_loss_acc_for_two_model

mpl.rcParams['figure.dpi'] = 120


def mlp(num_epochs, unit_count, hidden_layer_num=1):
    net = nn.HybridSequential()
    with net.name_scope():
        for _ in range(hidden_layer_num):
            net.add(gluon.nn.Dense(unit_count, activation="relu"))
        net.add(gluon.nn.Dense(10))
    net.initialize()
    ctx = utils.try_gpu()
    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.2})
    utils.train_wiht_V(train_data, test_data, net, loss, trainer, ctx, num_epochs=num_epochs, logpath="./mpl")


num_epochs = 30
unit_count = 28*28
unit_count_2 = 28*28

mlp(num_epochs, unit_count_2, hidden_layer_num=2)

