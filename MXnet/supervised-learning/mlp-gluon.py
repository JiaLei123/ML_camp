from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
from MXnet import utils
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 120

from mxboard import SummaryWriter

with SummaryWriter(logdir='./logs', flush_secs=5) as sw:

    round_of_run = 30
    unit_count = 28*28

    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(unit_count, activation="relu"))
        net.add(gluon.nn.Dense(unit_count, activation="relu"))
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(256, activation="relu"))
        net.add(gluon.nn.Dense(10))
    net.initialize()
    net.hybridize()

    batch_size = 256
    train_data, test_data = utils.load_data_fashion_mnist(batch_size)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.1})

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    params = net.collect_params()
    param_names = params.keys()

    global_step = 0
    for epoch in range(round_of_run):
        train_loss = 0
        train_acc = 0
        for i, (data, label) in enumerate(train_data):
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
            global_step += 1
            sw.add_scalar(tag='cross_entropy', value=loss.mean().asscalar(), global_step=global_step)

            if i == 0:
                sw.add_image('minist_first_minibatch', data.reshape((batch_size, 1, 28, 28)), epoch)

        grads = [i.grad() for i in net.collect_params().values()]
        if epoch == 0:
            sw.add_graph(net)

        for i, name in enumerate(param_names):
            try:
                sw.add_histogram(tag=name, values=grads[i], global_step=epoch, bins=1000)
            except Exception as ex:
                print(grads[i])
        test_acc = utils.evaluate_accuracy(test_data, net)
        test_lost = utils.calculate_test_lost(test_data, softmax_cross_entropy, net)

        train_loss_list.append(train_loss / len(train_data))
        test_loss_list.append(test_lost)

        train_acc_list.append(train_acc / len(train_data))
        test_acc_list.append(test_acc)

        sw.add_scalar(tag='accuracy_curves', value=('train_acc', train_acc / len(train_data)), global_step=epoch)
        sw.add_scalar(tag='accuracy_curves', value=('valid_acc', test_acc), global_step=epoch)

        print("Epoch %d. train Loss: %f, test Loss: %f, Train acc %f, Test acc %f" % (
            epoch, train_loss / len(train_data),  test_lost, train_acc / len(train_data), test_acc))

    # sw.export_scalars('scalar_dict.json')
    sw.close()

# plt.figure()
# title = "Loss 1 layer, %d unit" % unit_count
# plt.plot(range(round_of_run), train_loss_list)
# plt.plot(range(round_of_run), test_loss_list)
# plt.title(title, color='blue', wrap=True)
# plt.legend(['train', 'test'])
# plt.figure()
# title = "Acc 1 layer, %d unit" % unit_count
# plt.plot(range(round_of_run), train_acc_list, label='train')
# plt.plot(range(round_of_run), test_acc_list, label='test')
# plt.title(title, color='blue', wrap=True)
#
# plt.show()
