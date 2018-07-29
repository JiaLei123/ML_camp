import pandas as pd

data_path = "E:\\ML_learning\\Daguan\\data\\train_set.csv"
total_data = pd.read_csv(data_path)
total_useful_data = total_data.loc[:, ["article", "class"]]
total_useful_data = total_useful_data.sample(frac=1)
total_len = len(total_useful_data.index)
train_data_len = int(total_len * 0.6)
test_data = total_len - train_data_len
train_data = total_useful_data[0:train_data_len]
test_data = total_useful_data[train_data_len:total_len + 1]
train_data.to_csv("E:\\ML_learning\\Daguan\\data\\train_data.csv")
test_data.to_csv("E:\\ML_learning\\Daguan\\data\\test_data.csv")
