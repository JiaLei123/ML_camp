from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl



data_path = "D:\\ML_learning\\Daguan\\data\\train_set.csv"
train_data = pd.read_csv(data_path)

article = train_data["article"]
f = lambda x: x.split(" ")
article_list = article.apply(f)

word_counts = Counter()
for line in article_list:
    word_counts.update(line)

counter_list = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

counter_low_freq = pd.DataFrame([item for item in word_counts.items() if item[1] < 10], columns=["word_index", "couter"])
counter_low_freq.to_csv('low_frequency_words.csv')

def top_n(count=10):
    return word_counts.most_common(count)


top10 = top_n(10)
[print("%s\t%d" % (value, count)) for value, count in top10]
plt.bar(range(len(top10)), [x[1] for x in top10], tick_label=[x[0] for x in top10])
plt.show()