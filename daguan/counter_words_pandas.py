from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import os


def top_n(word_counts, count=10):
    return word_counts.most_common(count)


def get_high_low_freq_words(data_path, column_name="article", top=10, low=10, show_plt=False):
    train_data_chunker = pd.read_csv(data_path, chunksize=5000)

    tot = pd.Series([])
    f = lambda x: pd.value_counts(x.split(" "))
    for piece in train_data_chunker:
        tot = tot.add(piece[column_name].apply(f))
        print(tot.head())




if __name__ == "__main__":
    data_path = "D:\\ML_learning\\Daguan\\data\\train_set.csv"
    base_path = os.path.dirname(data_path)

    get_high_low_freq_words(data_path)
