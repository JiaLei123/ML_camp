import matplotlib
from pandas import read_csv
from datetime import datetime


def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')


#加载数据并且为数据整形
dataset = read_csv('pollution.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

#删除No列
dataset.drop('No', axis=1, inplace=True)

# 修改剩余列名称
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'


# 将所有空值替换为0
dataset['pollution'].fillna(0, inplace=True)

dataset = dataset[24:] #第一天的没有污染数据 所以去除
print(dataset.head(30))


#数据特征的探查
from pandas import read_csv
from matplotlib import pyplot
#方便在浏览器中显示图标

# 去dataframe中的值，组成一个向量，每行是向量中的一个值
values = dataset.values
# print(values)
print(values[:, 1])


# groups = [0, 1, 2, 3, 5, 6, 7]
# i = 1
# pyplot.figure()
# for group in groups:
# 	pyplot.subplot(len(groups), 1 ,i)
# 	pyplot.plot(values[: , group])
# 	pyplot.title(dataset.columns[group], y=0.5, loc='right')
# 	i += 1
# pyplot.show()


# 将数据集构建为监督学习问题
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat


# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:,4])
print(values[:, 4])
