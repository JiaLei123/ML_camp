from mxnet import nd
from MXnet import utils
from mxnet import autograd as autograd
from mxnet import gluon

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

weight_scale = 0.01
w1 = nd.random.normal(shape=(50, 1, 5, 5), scale=weight_scale)
b1 = nd.zeros(w1.shape[0])

w2 = nd.random.normal(shape=(100, 50, 3, 3), scale=weight_scale)
b2 = nd.zeros(w2.shape[0])

w3 = nd.random.normal(shape=(2500, 5000), scale=weight_scale)
b3 = nd.zeros(w3.shape[1])

w4 = nd.random.normal(shape=(w3.shape[1], 10), scale=weight_scale)
b4 = nd.zeros(w4.shape[1])

params = [w1, b1, w2, b2, w3, b3, w4, b4]
[param.attach_grad() for param in params]


def net(X, verbose=False):
    h1_conv = nd.Convolution(data=X, weight=w1, bias=b1, kernel=w1.shape[2:], num_filter=w1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(data=h1_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))

    h2_conv = nd.Convolution(data=h1, weight=w2, bias=b2, kernel=w2.shape[2:], num_filter=w2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
    h2 = nd.flatten(h2)

    h3_linear = nd.dot(h2, w3) + b3
    h3 = nd.relu(h3_linear)

    h4_linear = nd.dot(h3, w4) + b4
    return h4_linear


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = 0.2

for epoch in range(30):
    train_loss = 0.
    train_acc = 0.
    for X, y in train_data:
        with autograd.record():
            output = net(X)
            loss = softmax_cross_entropy(output, y)
        loss.backward()
        utils.SGD(params, learning_rate / batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, y)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss / len(train_data),
        train_acc / len(train_data), test_acc))
