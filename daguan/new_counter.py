import re
from collections import Counter
import pandas as pd


def parse_data(data_path, column_name="article"):
    train_data = pd.read_csv(data_path)

    article = train_data[column_name]
    f = lambda x: x.split(" ")[-1]
    article_list = article.apply(f)

    end_word_counts = Counter()

    end_word_counts.update(article_list)
    print(end_word_counts)

    re_split_str = "|".join(end_word_counts.keys())
    print(re_split_str)
    regex_split = re.compile(re_split_str)
    f_split = lambda x: re.split(regex_split, x)
    article_split_list = article.apply(f_split)
    sentence_counts = Counter()
    for article_line in article_split_list:
        sentence_counts.update(article_line)
    print(sentence_counts)

if __name__ == '__main__':
    data_path = "D:\\ML_learning\\Daguan\\data\\train_set.csv"
    # get_data_article(path1, path2)
    parse_data(data_path)