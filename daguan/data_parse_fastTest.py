import pandas as pd
import matplotlib.pyplot as plt
import os

from daguan.counter_words import get_high_low_freq_words


def parse_data_fastText(data_path, column_name="article", train_rate=0.6):
    """
    parse data for fastText classifiers
    s a text file containing a training sentence per line along with the labels.
    By default, we assume that labels are words that are prefixed by the string __label__.

    :return:
    """
    base_path = os.path.dirname(data_path)
    train_data = pd.read_csv(data_path)
    train_article_data = train_data.loc[:, [column_name, "class"]]
    f = lambda x: "__label__" + str(x)
    train_article_data["class"] = train_article_data["class"].apply(f)

    topN, _ = get_high_low_freq_words(data_path, top=10)
    f_top = lambda x: " ".join(filter(lambda s: s not in topN, x.split(" ")))
    train_article_data[column_name] = train_article_data[column_name].apply(f_top)

    print(train_article_data.head())
    total_useful_data = train_article_data.sample(frac=1)
    if train_rate < 1:
        total_len = len(total_useful_data.index)
        train_data_len = int(total_len * train_rate)
        train_data = total_useful_data[0:train_data_len]
        test_data = total_useful_data[train_data_len:total_len + 1]
        train_data.to_csv(os.path.join(base_path, "train_data_" + column_name + "_fastTest.tsv"), sep='\t')
        test_data.to_csv(os.path.join(base_path, "test_data_" + column_name + "_fastTest.tsv"), sep='\t')
    else:
        # output all traing_set
        total_useful_data.to_csv(os.path.join(base_path, "train_data_all_" + column_name + "_fastTest.tsv"), sep='\t',
                                 header=False, index=False)


if __name__ == "__main__":
    data_path = "E:\\ML_learning\\Daguan\\data\\train_set.csv"
    parse_data_fastText(data_path, train_rate=1)
