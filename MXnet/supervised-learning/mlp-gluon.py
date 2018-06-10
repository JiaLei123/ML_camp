from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from MXnet import utils
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120
import matplotlib.pyplot as plt

round_of_run = 100
unit_count = 56*56

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(unit_count, activation="relu"))
    # net.add(gluon.nn.Dense(28 * 28, activation="relu"))
    # net.add(gluon.nn.Dense(28 * 28, activation="relu"))
    # net.add(gluon.nn.Dense(28 * 28, activation="relu"))
    net.add(gluon.nn.Dense(10))
net.initialize()

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.5})

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(round_of_run):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    test_lost = utils.calculate_test_lost(test_data, softmax_cross_entropy, net)

    train_loss_list.append(train_loss / len(train_data))
    test_loss_list.append(test_lost)

    train_acc_list.append(train_acc / len(train_data))
    test_acc_list.append(test_acc)

    print("Epoch %d. train Loss: %f, test Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss / len(train_data),  test_lost, train_acc / len(train_data), test_acc))


plt.figure()
title = "Loss 1 layer, %d unit" % unit_count
plt.plot(range(round_of_run), train_loss_list)
plt.plot(range(round_of_run), test_loss_list)
plt.title(title, color='blue', wrap=True)
plt.legend(['train', 'test'])
plt.figure()
title = "Acc 1 layer, %d unit" % unit_count
plt.plot(range(round_of_run), train_acc_list, label='train')
plt.plot(range(round_of_run), test_acc_list, label='test')
plt.title(title, color='blue', wrap=True)

plt.show()
