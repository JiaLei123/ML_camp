import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np

from MXnet import utils
from MXnet.display_utils import show_loss_acc_for_two_model

number_inputs = 2
number_example = 100000
true_w = nd.array([2, -3.4]).reshape((2, 1))
true_b = 4.2

X = nd.random.normal(scale=1, shape=(number_example, number_inputs))
y = nd.dot(X, true_w) + true_b
y += 0.01 * nd.random.normal(scale=1, shape=y.shape)


def train_linear(num_epochs, batch_size=5):
    dataset = gdata.ArrayDataset(X, y)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(1))
    net.initialize()
    square_loss = gloss.L2Loss()
    ctx = utils.try_gpu()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
    return utils.train(data_iter, data_iter, net, square_loss, trainer, ctx, num_epochs=num_epochs)


num_epochs = 10
unit_count = 1
train_loss_list, test_loss_list, train_acc_list, test_acc_list = train_linear(num_epochs, batch_size=5)
train_loss_2_list, test_loss_2_list, train_acc_2_list, test_acc_2_list = train_linear(num_epochs, batch_size=10)
show_loss_acc_for_two_model(unit_count, num_epochs,
                            train_loss_list, train_loss_2_list,
                            test_loss_list, test_loss_2_list,
                            train_acc_list, train_acc_2_list,
                            test_acc_list, test_acc_2_list,
                            "batch size 5", "batch sieze 10")
