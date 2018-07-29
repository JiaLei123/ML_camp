import math
import os
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


class Dictionary(object):
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = []

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.idx_to_word.append(word)
            self.word_to_idx[word] = len(self.idx_to_word) - 1  # 就是返回word在idx_to_word中的index值
        return self.word_to_idx[word]

    def __len__(self):
        return len(self.idx_to_word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train, _train = self.tokenize(path + 'train.txt')
        self.valid, _val = self.tokenize(path + 'valid.txt')
        self.test, _test = self.tokenize(path + 'test.txt')
        all_sentences = list()
        all_sentences.extend(_train)
        all_sentences.extend(_val)
        all_sentences.extend(_test)
        self.w2v = word2vec.Word2Vec(all_sentences)

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        with open(path, 'r') as f:
            indices = np.zeros((tokens,), dtype="int32")
            idx = 0
            all_sentences = list()
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    indices[idx] = self.dictionary.word_to_idx[word]
                    idx += 1
                all_sentences.append(words)
        return mx.nd.array(indices, dtype='int32'), all_sentences



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
        # input_node = self.get_vec(inputs)
        # emb = self.drop(inputs)
        outputs = []
        for input in inputs:
            step, vec_size = input.shape
            input = input.reshape(step, 1, -1)
            output, state = self.rnn(input, state)
            output = self.drop(output)
            output = output[-1].reshape((batch_size, -1))
            outputs.append(output)
        outputs = mx.nd.concat(*outputs, dim=0)
        decoded = self.decoder(outputs.reshape((-1, self.hidden_dim)))
        return decoded, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

def get_batch(source, label, i):
    data = source[i]
    target = label[i]
    return data, target

def get_data_iter(path, batch_size, w2v_vec):
    total_data = pd.read_csv(path)
    data = total_data["article"][0:100]
    f = lambda x: [w2v_vec.wv.get_vector(xi) for xi in x.split(" ")]
    #
    data = data.apply(f)

    # data = pd.read_pickle("E:\\ML_learning\\Daguan\\data\\train_data_vec.plk", "gzip")
    label = total_data["class"][0:100]

    # dataset = gdata.ArrayDataset(data, label)
    # data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    return data, label


def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


def model_eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size_clas, ctx=context)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def train():
    for epoch in range(epochs):
        total_L = 0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size_clas, ctx=context)

        batch_num = 0
        for X, y in train_data_iter:
            batch_num += 1
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(X, hidden)
                # output = output[-1].reshape((1, -1))
                L = loss(output, y)
                L.backward()
            grads = [i.grad(context) for i in model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, clipping_norm * num_steps * batch_size)
            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if batch_num % eval_period == 0 and batch_num > 0:
                cur_L = total_L / batch_num
                train_acc = evaluate_accuracy(train_data_iter, model)
                print('[Epoch %d Batch %d] loss %.2f, Train acc %f' % (epoch + 1, batch_num, cur_L, train_acc))

        cur_L = total_L / len(train_data_iter)
        train_acc = evaluate_accuracy(train_data_iter, model)
        print('[Epoch %d loss %.2fm Train acc %f' % (epoch + 1, cur_L, train_acc))


def _get_batch(batch, ctx):
    """return data and label on ctx"""
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = mx.nd.array([0])
    n = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = _get_batch(batch, ctx)
        for X, y in zip(data, label):
            hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size_clas, ctx=context)
            y = y.astype('float32')
            pred, _ = net(X, hidden)
            acc += mx.nd.sum(pred.argmax(axis=1) == y).copyto(mx.cpu())
            n += y.size
        acc.wait_to_read()  # don't push too many operators into backend
    return acc.asscalar() / n


if __name__=="__main__":
    model_name = 'rnn_relu'
    embed_dim = 100
    hidden_dim = 100
    num_layers = 2
    lr = 1
    clipping_norm = 0.2
    epochs = 10
    batch_size = 5
    batch_size_clas = 1
    num_steps = 1
    dropout_rate = 0.2
    eval_period = 20

    context = utils.try_gpu()

    train_data_path = "E:\\ML_learning\\Daguan\\data\\train_data.csv"
    w2v = word2vec.Word2Vec.load("E:\\ML_learning\\Daguan\\data\\mymodel")
    # test_data_path = ""
    # train_data, label = get_data_iter(train_data_path, batch_size, w2v)
    train_data_iter = get_data_iter(train_data_path, batch_size, w2v)


    model = RNNModel(model_name, embed_dim, hidden_dim, num_layers, w2v, dropout_rate)
    model.collect_params().initialize(mx.init.Xavier(), ctx=context)

    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    # model_eval(val_data)
    train()
    # test_L = model_eval(test_data_iter)
    # print('Test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))

