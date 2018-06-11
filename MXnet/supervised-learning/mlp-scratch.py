import sys
from MXnet import utils
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet import autograd as autograd

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)
# data shape (256, 1, 28 ,28)

# 初始化参数
num_inputs = 28 * 28
num_outputs = 10

num_hidden1 = 56 * 56
num_hidden2 = 56 * 56
weight_scale = 0.01

W1 = nd.random.normal(scale=weight_scale, shape=(num_inputs, num_hidden1))
b1 = nd.zeros(num_hidden1)

W2 = nd.random.normal(scale=weight_scale, shape=(num_hidden1, num_hidden2))
b2 = nd.zeros(num_hidden2)

W3 = nd.random.normal(scale=weight_scale, shape=(num_hidden2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = relu(nd.dot(X, W1) + b1)
    h2 = relu(nd.dot(h1, W2) + b2)
    output = nd.dot(h2, W3) + b3
    return output

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learn_rate = 0.5

for epoch in range(10):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        utils.SGD(params, learn_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss / len(train_data),
        train_acc / len(train_data), test_acc))



