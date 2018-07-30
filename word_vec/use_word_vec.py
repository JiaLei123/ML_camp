from gensim.models import word2vec
from sklearn.decomposition import PCA
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

mopdelfilePath = 'D:\\ML_learning\\NLP_data\\model.bin'
model = word2vec.Word2Vec.load(mopdelfilePath)
raw_word_vec = model.wv.vectors

cent_word1 = "中国"
cent_word2 = "成都"
cent_word3 = "淘宝"
cent_word4 = "自行车"
cent_word5 = "计算机"


wordList1 = model.wv.most_similar_cosmul(cent_word1)
wordList2 = model.wv.most_similar_cosmul(cent_word2)
wordList3 = model.wv.most_similar_cosmul(cent_word3)
wordList4 = model.wv.most_similar_cosmul(cent_word4)
wordList5 = model.wv.most_similar_cosmul(cent_word5)


wordList1 = np.append([item[0] for item in wordList1], cent_word1)
wordList2 = np.append([item[0] for item in wordList2], cent_word2)
wordList3 = np.append([item[0] for item in wordList3], cent_word3)
wordList4 = np.append([item[0] for item in wordList4], cent_word4)
wordList5 = np.append([item[0] for item in wordList5], cent_word5)

def get_word_index(word):
    index = model.wv.vocab[word].index
    return index

index_list1 = map(get_word_index, wordList1)
index_list2 = map(get_word_index, wordList2)
index_list3 = map(get_word_index, wordList3)
index_list4 = map(get_word_index, wordList4)
index_list5 = map(get_word_index, wordList5)

vec_reduced = PCA(n_components=2).fit_transform(raw_word_vec)
# fig = plt.figure()
# ax = fig.add_subplot(111)
zhfont = matplotlib.font_manager.FontProperties(fname=r'C:\Nuance\python_env\basic_dev\Lib\site-packages\matplotlib\mpl-data\fonts\ttf\msyh.ttf')
x = np.arange(-10, 10, 0.1)
y = x
plt.plot(x, y)

for i in index_list1:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='r', fontproperties=zhfont)

for i in index_list2:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='b', fontproperties=zhfont)

for i in index_list3:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='g', fontproperties=zhfont)

for i in index_list4:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='k', fontproperties=zhfont)

for i in index_list5:
    plt.text(vec_reduced[i][0], vec_reduced[i][1], model.wv.index2word[i], color='c', fontproperties=zhfont)

# plt.axis([40, 160, 0, 0.03])
plt.show()

# indexes = model.wv.most_similar_cosmul('中国')
# for index in indexes:
#     print(index)
