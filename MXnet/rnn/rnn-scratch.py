import zipfile
import random
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from math import exp
from MXnet import utils

with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/jaychou_lyrics.txt', encoding='utf-8') as f:
    corpus_chars = f.read()
print(corpus_chars[0:49])

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)

corpus_indices = [char_to_idx[char] for char in corpus_chars]


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size

    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        data = nd.array([_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array([_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = - data_len // batch_size

    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in epoch_size:
        i = i * num_steps
        data = indices[:, i: i + num_steps]
        label = indices[:, i + 1: i + num_steps + 1]
        yield data, label


def get_inputs(X):
    return [nd.one_hot(x, vocab_size) for x in X.T]


ctx = utils.try_gpu()
print('will use', ctx)

num_inputs = vocab_size
hidden_dim = 256
num_outputs = vocab_size
std = .01

def get_params():
    W_xh = nd.random.normal(scale=std, shape=(num_inputs, hidden_dim), ctx=ctx)
    W_hh = nd.random.normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    W_y = nd.random.normal(scale=std, shape=(hidden_dim, num_outputs), ctx=ctx)
    b_y = nd.zeros(num_outputs, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_y, b_y]

    for param in params:
        param.attach_grad()
    return params


def rnn(inputs, stats, *params):
    """
    :param inputs:
    :param stats: 隐藏状态
    :param params:
    :return:
    """
    H = stats
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)


def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad ** 2)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm

def train_and_predict_rnn(rnn, is_random_iter, epochs, num_steps, hidden_dim,
                          learning_rate, clipping_theta, batch_size,
                          pred_period, pred_len, seqs, get_params, get_inputs,
                          ctx, corpus_indices, idx_to_char, char_to_idx,
                          is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


    for e in range(1, epochs +1):
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps, ctx):
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                if is_lstm:
                    outputs, state_h, state_c = rnn(get_inputs(data), state_h, state_c, *params)
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                label = label.T.reshape((-1,))
                outputs = nd.concat(*outputs, dim=0)
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, clipping_theta, ctx)
            utils.SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            print("Epoch %d. Preflexity %f" % (e, exp(train_loss/num_examples)))

            for seq in seqs:
                print('-', predict_run(rnn, seq, pred_len, params,
                      hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs,
                      is_lstm))
                print()

def predict_run(rnn, prefix, num_chars, params, hidden_dim, ctx, idx_to_char, char_to_idx, get_inputs, is_lstm=False):
    """
    prefix 就是要进行预测时给出的初始值
    :param rnn:
    :param prefix:
    :param num_chars:
    :param params:
    :param hidden_dim:
    :param ctx:
    :param idx_to_char:
    :param char_to_idx:
    :param get_inputs:
    :param is_lstm:
    :return:
    """
    prefix = prefix.low()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]] #先把预设值写入output
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        if is_lstm:
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)

        if i < len(prefix) -1:
            next_input = char_to_idx[prefix[i+1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return " ".join([idx_to_char[i] for i in output])



epochs = 200
num_steps = 35
learning_rate = 0.1
batch_size = 32
clipping_theta = 5
seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

train_and_predict_rnn(rnn=rnn, is_random_iter=True, epochs=200, num_steps=35,
                      hidden_dim=hidden_dim, learning_rate=learning_rate,
                      clipping_theta=clipping_theta, batch_size=32, pred_period=20,
                      pred_len=100, seqs=seqs, get_params=get_params,
                      get_inputs=get_inputs, ctx=ctx,
                      corpus_indices=corpus_indices, idx_to_char=idx_to_char,
                      char_to_idx=char_to_idx)