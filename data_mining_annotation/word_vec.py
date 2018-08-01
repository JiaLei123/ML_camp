from gensim.models import word2vec
import pandas as pd

mopdelfilePath = 'D:\\ML_learning\\annotation\\model.bin'
fileSegWordDonePath = 'D:\\ML_learning\\annotation\\incorrect_annotation.csv'

fileTrainRead = pd.read_csv(fileSegWordDonePath)
train_sentences = pd.Series(fileTrainRead.iloc[:, 3])
f = lambda x: str(x).replace("|", "").replace("/", "").split(" ")
train_sentences = train_sentences.apply(f)

model = word2vec.Word2Vec(train_sentences, size=300)
model.save(mopdelfilePath)