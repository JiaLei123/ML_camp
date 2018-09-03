import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_data_fastText(data_path, column_name="article"):
    """
    parse data for fastText classifiers predict
    :return:
    """
    base_path = os.path.dirname(data_path)
    test_data = pd.read_csv(data_path)
    test_article_data = test_data.loc[:, [column_name]]
    test_article_data.to_csv(os.path.join(base_path, "test_data_all_" + column_name + "_fastTest.tsv"), sep='\t',
                             header=False, index=False)


def generate_test_result(data_path, result_path):
    base_path = os.path.dirname(data_path)
    result_data = pd.read_csv(result_path, header=None).iloc[:, 0]
    f_remove_lable = lambda x: x.replace("__label__", "")
    result_data = result_data.apply(f_remove_lable)
    print(result_data.head())

    test_data = pd.read_csv(data_path)
    test_data["class"] = result_data
    print(test_data.head())
    output_result = test_data.loc[:, ["id", "class"]]
    output_result.to_csv(os.path.join(base_path, "final_result.csv"), index=False)


if __name__ == "__main__":
    data_path = "E:\\ML_learning\\Daguan\\data\\test_set.csv"
    result_path = "E:\\ML_learning\\Daguan\\data\\result.tsv"
    # parse_data_fastText(data_path, column_name="word_seg")
    generate_test_result(data_path, result_path)
