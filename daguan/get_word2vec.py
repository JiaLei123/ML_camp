from gensim.models import word2vec
import pandas as pd

data_path = "E:\\ML_learning\\Daguan\\data\\train_set.csv"
train_data = pd.read_csv(data_path)
article = train_data["article"]
f = lambda x: x.split(" ")
article_sentences = article.apply(f)

model = word2vec.Word2Vec(article_sentences, min_count=1)
model.save('E:\\ML_learning\\Daguan\\data\\mymodel')