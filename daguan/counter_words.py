from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl



data_path = "E:\\ML_learning\\Daguan\\data\\train_set.csv"
train_data = pd.read_csv(data_path)

article = train_data["article"]
f = lambda x: x.split(" ")
article_list = article.apply(f)

word_counts = Counter()
for line in article_list:
    word_counts.update(line)

# i = 1
# with open(data_path) as f:
#     for line in f:
#         # print("add new batch %d" % i)
#         # i += 1
#         words_line = line.strip().split(',')[1]
#         word_counts.update(words_line.strip().split(' '))

# print(word_counts)

counter_list = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

counter_low_freq = pd.DataFrame([item for item in word_counts.items() if item[1] < 10], columns=["word_index", "couter"])
counter_low_freq.to_csv('low_frequency_words.csv')

for word, count in counter_list[:20]:
    print("%s\t%d" % (word, count))


label = list(map(lambda x: x[0], counter_list[:20]))
value = list(map(lambda y: y[1], counter_list[:20]))

plt.bar(range(len(value)), value, tick_label=label)
plt.show()