import pandas as pd
import matplotlib.pyplot as plt

data_path = "E:\\ML_learning\\Daguan\\data\\train_set.csv"
train_data = pd.read_csv(data_path)
print(train_data.head())
classer = train_data["class"]
classer_dis = classer.value_counts().sort_index()
classer_dis.plot.bar()
plt.show()

article = train_data["article"]
f = lambda x: len(x.split(" "))
article_len = article.apply(f)
article_len_dis = article_len.value_counts().sort_index()
article_len_dis.plot()
plt.show()

word_seg = train_data["word_seg"]
word_seg_len = word_seg.apply(f)
word_seg_len_dis = word_seg_len.value_counts().sort_index()
word_seg_len_dis.plot()
plt.show()