from gensim.models import word2vec
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

mopdelfilePath = 'E:\\ML_learning\\NLP_data\\model.bin'
model = word2vec.Word2Vec.load(mopdelfilePath)
raw_word_vec = model.wv.vectors

vec_reduced = PCA(n_components=2).fit_transform(raw_word_vec)

index1, metrics1 = model.wv.most_similar_cosmul('中国')
index2, metrics2 = model.wv.most_similar_cosmul('清华')
index3, metrics3 = model.wv.most_similar_cosmul('牛顿')
index4, metrics4 = model.wv.most_similar_cosmul('自动化')
index5, metrics5 = model.wv.most_similar_cosmul('刘亦菲')

index01 = np.where(model.wv.vocab == '中国')
index02 = np.where(model.wv.vocab == '清华')
index03 = np.where(model.wv.vocab == '牛顿')
index04 = np.where(model.wv.vocab == '自动化')
index05 = np.where(model.wv.vocab == '刘亦菲')

index1 = np.append(index1, index01)
index2 = np.append(index2, index02)
index3 = np.append(index3, index03)
index4 = np.append(index4, index04)
index5 = np.append(index5, index05)

fig = plt.figure()
ax = fig.add_subplot(111)

for i in index1:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='r')

for i in index2:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='b')

for i in index3:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='g')

for i in index4:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='k')

for i in index5:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.vocab[i], fontproperties=zhfont, color='c')

ax.axis([0, 0.8, -0.5, 0.5])
plt.show()

indexes = model.wv.most_similar_cosmul('中国')
for index in indexes:
    print(index)
