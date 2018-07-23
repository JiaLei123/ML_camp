import math
import os
import time
import numpy as np
from mxnet import nd
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import zipfile
from MXnet import utils

with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data/')


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
        self.train, self.train_words = self.tokenize(path + 'train.txt')
        self.valid, _ = self.tokenize(path + 'valid.txt')
        self.test, _ = self.tokenize(path + 'test.txt')

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
            word_list = list()
            idx = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    indices[idx] = self.dictionary.word_to_idx[word]
                    idx += 1
                    word_list.append(word)
        return mx.nd.array(indices, dtype='int32'), word_list


class RNNModel(gluon.Block):
    def __init__(self, mode, vocab_size, embed_dim, hidden_dim, num_layers, drop_out=0.5, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(drop_out)
            self.encoder = nn.Embedding(vocab_size, embed_dim, weight_initializer=mx.init.Uniform(0.1))

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

            self.decoder = nn.Dense(vocab_size, in_units=hidden_dim)
            self.hidden_dim = hidden_dim

    def forward(self, inputs, state):
        input_node = self.encoder(inputs)
        emb = self.drop(input_node)
        output, state = self.rnn(emb, state)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.hidden_dim)))
        return decoded, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


model_name = 'rnn_tanh'
embed_dim = 100
hidden_dim = 100
num_layers = 2
lr = 1
clipping_norm = 0.2
epochs = 10
batch_size = 32
num_steps = 5
dropout_rate = 0.2
eval_period = 500

context = utils.try_gpu()


def batchify(data, batch_size):
    num_batches = data.shape[0] // batch_size
    data = data[: num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data


data = '../data/ptb/ptb.'
corpus = Corpus(data)
vocab_size = len(corpus.dictionary)
print(vocab_size)

train_data = batchify(corpus.train, batch_size).as_in_context(context)
val_data = batchify(corpus.valid, batch_size).as_in_context(context)
test_data = batchify(corpus.test, batch_size).as_in_context(context)

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim, num_layers, dropout_rate)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)

trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0] - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len]
    return data, target.reshape((-1,))


def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


def model_eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def model_predic(data_source):
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        data, target = get_batch(data_source, i)
        Y, hidden = model(data, hidden)
        next_input = Y.argmax(axis=1)
        if i == 0:
            print(next_input[:15], target[:15])


def train():
    for epoch in range(epochs):
        total_L = 0
        start_time = time.time()
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)

        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, num_steps)):
            data, target = get_batch(train_data, i)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()
            grads = [i.grad(context) for i in model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, clipping_norm * num_steps * batch_size)
            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % eval_period == 0 and ibatch > 0:
                cur_L = total_L / num_steps / batch_size / eval_period
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' % (
                    epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

                prefix = 'talking about'
                num_chars = 5
                predict_words = predict_run(model, prefix, num_chars, context, corpus.dictionary.idx_to_word,
                                            corpus.dictionary.word_to_idx)
                print(predict_words)
                # model_predic(val_data)
        val_L = model_eval(val_data)
        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation '
              'perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L,
                                   math.exp(val_L)))


def predict_run(model, prefix, num_chars, ctx, idx_to_char, char_to_idx):
    prefix = prefix.lower()
    prefix_list = prefix.split()
    output = [char_to_idx[prefix_list[0]]]
    for i in range(num_chars + len(prefix_list)):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=1, ctx=context)
        Y, hidden = model(X, hidden)
        if i < len(prefix_list) - 1:
            next_input = char_to_idx[prefix_list[i + 1]]
        else:
            next_input = int(Y.argmax(axis=1).asscalar())
        output.append(next_input)
    return " ".join([idx_to_char[i] for i in output])


print(corpus.train[:40])
print(corpus.train_words[:40])
prefix = 'talking about'
num_chars = 5
predict_words = predict_run(model, prefix, num_chars, context, corpus.dictionary.idx_to_word, corpus.dictionary.word_to_idx)
print(predict_words)

# model_predic(val_data)

train()

test_L = model_eval(test_data)
print('Test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))

# model_predic(val_data)

predict_words = predict_run(model, prefix, num_chars, context, corpus.dictionary.idx_to_word, corpus.dictionary.word_to_idx)
print(predict_words)
