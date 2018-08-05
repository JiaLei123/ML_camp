import math
import os
import random
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, data as gdata
import zipfile
from MXnet import utils
from gensim.models import word2vec
import pandas as pd
import numpy as np


high_frequency_word_list = ['1044285', '7368', '856005', '72195', '195449', '359838', '239755', '427848', '316564']

class RNNModel(gluon.Block):
    def __init__(self, mode, embed_dim, hidden_dim, num_layers, w2v_vec, drop_out=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(drop_out)
            # self.encoder = nn.Embedding(grad_red='null')
            # self.encoder.weight.set_data(w2v_vec)

            if mode == "rnn_relu":
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='relu', dropout=drop_out, input_size=embed_dim)
            elif mode == "rnn_tanh":
                self.rnn = rnn.RNN(hidden_dim, num_layers, activation='tanh', dropout=drop_out, input_size=embed_dim)
            elif mode == "lstm":
                self.rnn = rnn.LSTM(hidden_dim, num_layers, dropout=drop_out, input_size=embed_dim)
            elif mode == "gru":
                self.rnn = rnn.GRU(hidden_dim, num_layers, dropout=drop_out, input_size=embed_dim)
            else:
                raise ValueError("Invalid Mode")

            self.decoder = nn.Dense(19, in_units=hidden_dim)
            self.hidden_dim = hidden_dim
            self.w2v_vec = w2v_vec

    # def get_vec(self,inputs):
    #     input_vec = []
    #     for word in enumerate(inputs):
    #         try:
    #             input_vec.append(self.w2v_vec.wv[word])
    #         except:
    #             input_vec.append(np.random.uniform(-0.25, 0.25, self.w2v_vec.vector_size))
    #     return mx.nd.array(input_vec).reshape((len(inputs), 1, -1))

    def forward(self, inputs, state):
        outputs = []
        for input in inputs:
            input_node = mx.nd.array(input)
            step = input_node.shape[0]
            input_node = input_node.reshape(step, 1, -1)
            output, out_state = self.rnn(input_node, state)
            output = self.drop(output)
            output = output[-1]
            outputs.append(output)
        outputs = mx.nd.concat(*outputs, dim=0)
        decoded = self.decoder(outputs)
        return decoded, out_state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


def get_batch(source, label, i):
    data = source[i]
    target = label[i]
    return data, target


def data_iter(source, target, batch_size):
    """
    get number_example/batch_size data list and each list has batch_size data
    :return:
    """
    number_example = len(target)
    idx = list(range(number_example))
    random.shuffle(idx)

    def _data(pos):
        return source[pos]

    def _lable(pos):
        return target[pos]

    for i in range(0, number_example, batch_size):
        batch_indices = idx[i: min(i + batch_size, number_example)]

        data = [_data(j) for j in batch_indices]
        label = [_lable(j) for j in batch_indices]
        yield data, label


def get_data_iter(path, batch_size, w2v_vec):
    total_data = pd.read_csv(path)
    data = total_data["article"][0:1000]
    f = lambda x: [w2v_vec.wv.get_vector(xi) for xi in [si for si in x.split(" ") if si not in high_frequency_word_list][0:500]]
    # f = lambda x: [xi for xi in x.split(" ")[0:800] ]
    #
    data = data.apply(f)

    label = total_data["class"][0:1000]

    # dataset = gdata.ArrayDataset(data, label)
    # data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    return np.array(data), np.array(label)


def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


def train():
    for epoch in range(epochs):
        total_L = 0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size_clas, ctx=context)

        batch_num = 0
        for X, y in data_iter(train_data, label, batch_size):
            # X, y = get_batch(train_data, label, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(X, hidden)
                L = loss(output, mx.nd.array(y))
                L.backward()
            grads = [i.grad(context) for i in model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, clipping_norm * num_steps * batch_size)
            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()
            batch_num += 1

            if batch_num % eval_period == 0 and batch_num > 0:
                cur_L = total_L / batch_num / batch_size
                # train_acc = evaluate_accuracy(train_data, label, model)
                print('[Epoch %d Batch %d] loss %.2f' % (epoch + 1, batch_num, cur_L))

        cur_L = total_L / len(label)
        train_acc = evaluate_accuracy(train_data, label, model)
        print('[Epoch %d loss %.2f Train acc %f' % (epoch + 1, cur_L, train_acc))


def evaluate_accuracy(train_data, label, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = mx.nd.array([0])
    n = 0.
    for X, y in data_iter(train_data, label, batch_size):
        # X, y = get_batch(train_data, label, i)
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size_clas, ctx=context)
        y = mx.nd.array(y)
        y = y.astype('float32')
        pred, _ = net(X, hidden)
        acc += mx.nd.sum(pred.argmax(axis=1) == y)
        n += y.size
        acc.wait_to_read()  # don't push too many operators into backend
    return acc.asscalar() / len(label)


if __name__ == "__main__":
    model_name = 'rnn_relu'
    embed_dim = 100
    hidden_dim = 100
    num_layers = 2
    lr = 0.5
    clipping_norm = 0.2
    epochs = 10
    batch_size = 20
    batch_size_clas = 1
    num_steps = 1
    dropout_rate = 0.2
    eval_period = 50

    context = utils.try_gpu()

    train_data_path = "E:\\ML_learning\\Daguan\\data\\train_data.csv"
    w2v = word2vec.Word2Vec.load("E:\\ML_learning\\Daguan\\data\\mymodel")
    # test_data_path = ""
    train_data, label = get_data_iter(train_data_path, batch_size, w2v)
    # train_data_iter = get_data_iter(train_data_path, batch_size, w2v)

    model = RNNModel(model_name, embed_dim, hidden_dim, num_layers, w2v, dropout_rate)
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)

    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.1, 'wd': 0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # model_eval(val_data)
    train()
    # test_L = model_eval(test_data_iter)
    # print('Test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
