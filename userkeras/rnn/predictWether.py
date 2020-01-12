import matplotlib
from pandas import read_csv
from datetime import datetime


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


# 加载数据并且为数据整形
dataset = read_csv('pollution_online.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

# 删除No列
dataset.drop('No', axis=1, inplace=True)

# 修改剩余列名称
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

# 将所有空值替换为0
dataset['pollution'].fillna(0, inplace=True)

dataset = dataset[24:]  # 第一天的没有污染数据 所以去除
print("整理过后的数据")
print(dataset.head(20))

# 数据特征的探查
from pandas import read_csv
from matplotlib import pyplot

# 方便在浏览器中显示图标

# 去dataframe中的值，组成一个向量，每行是向量中的一个值
values = dataset.values
# print(values)
# print(values[:, 1])

# 将特征值画图
# groups = [0, 1, 2, 3, 5, 6, 7]
# i = 1
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups), 1, i)
# 	pyplot.plot(values[:, group])
# 	pyplot.title(dataset.columns[group], y=0.5, loc='right')
# 	i += 1
# 	pyplot.show()


# 将数据集构建为监督学习问题
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat

# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
print("对离线数据进行处理")
print(encoder.fit_transform(values[:, 4]))

# 确保所有数据是浮点数类型
values = values.astype('float32')

# 对特征标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)
print("特征标准化")
print(scaled_values[: 20])


# 将数据转换成监督学习问题
# n_in  滞后观察次数作为输入 Number of lag observations as input
# n_out: Number of observations as output (y).
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # 输入序列(t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # 预测序列(t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]

    # 纵列的方式连接在一起
    agg = concat(cols, axis=1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace=True)
    return agg


reframed = series_to_supervised(scaled_values, 1, 1)
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

print("转换为监督学习方法后的数据")
print(reframed[: 20])


# 切分训练集和测试机
reframed_values = reframed.values
n_train_hours = 365 * 24
train = reframed_values[:n_train_hours, :]
test = reframed_values[n_train_hours:, :]

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

print("训练集和测试集")
print(train_X[:, :2])
print(train_y)

# 将输入转换为三维格式 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer='adam')

history = model.fit(train_X, train_y, epochs=50, batch_size=36, validation_data=(test_X, test_y), verbose=2, shuffle=False)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

from numpy import concatenate
from keras.layers import LSTM
from math import sqrt

yhat = model.predict(test_X)
# pyplot.plot(yhat, label='yhat')
# pyplot.plot(test_y, label='ytest')
# pyplot.legend()
# pyplot.show()

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# 预测值反转缩放
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 实际值反转缩放
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# pyplot.plot(inv_yhat, label='yhat')
# pyplot.plot(inv_y, label='ytest')
# pyplot.legend()
# pyplot.show()
