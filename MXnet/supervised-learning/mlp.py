from mxnet.gluon import nn
from mxnet import gluon
import matplotlib.pyplot as plt
from MXnet import utils
from MXnet.display_utils import show_loss_acc_picture, show_loss_acc_for_two_model


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
