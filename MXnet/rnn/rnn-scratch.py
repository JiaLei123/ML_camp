import zipfile
import random
from mxnet import nd

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
        batch_indices = example_indices[i: 1 + batch_size]
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


def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]


ctx = gb.try_gpu()
print('will use', ctx)

num_inputs = vocab_size
num_hiddens = 256
num_outputs = vocab_size


def get_params():
    W_xh = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens), ctx=ctx)
    W_hh = nd.random.normal(scale=0.01, shape=(num_hiddens, num_hiddens), ctx=ctx)
    b_h = nd.zeros(num_hiddens, ctx=ctx)

    W_y = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs), ctx=ctx)
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



