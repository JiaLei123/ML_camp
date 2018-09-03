from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import os


def top_n(word_counts, count=10):
    return word_counts.most_common(count)


def get_high_low_freq_words(data_path, column_name="article", top=10, low=10, show_plt=False):
    train_data = pd.read_csv(data_path)

    article = train_data[column_name]
    f = lambda x: x.split(" ")
    article_list = article.apply(f)

    word_counts = Counter()
    for line in article_list:
        word_counts.update(line)

    counter_low_freq = pd.DataFrame([item for item in word_counts.items() if item[1] < low],
                                    columns=["word_index", "couter"])

    topN = top_n(word_counts, top)

    top_frame = pd.DataFrame(topN, columns=["Word", "Count"])
    print(top_frame)
    # [print("%s\t%d" % (value, count)) for value, count in topN]
    if show_plt:
        plt.bar(range(len(topN)), [x[1] for x in topN], tick_label=[x[0] for x in topN])
        plt.show()

    return topN, counter_low_freq


if __name__ == "__main__":
    data_path = "E:\\ML_learning\\Daguan\\data\\train_set.csv"
    base_path = os.path.dirname(data_path)

    topN, counter_low_freq = get_high_low_freq_words(data_path, top=20)
    # counter_low_freq.to_csv(os.path.join(base_path, 'low_frequency_words.csv'))
