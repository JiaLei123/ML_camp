from gensim.models import word2vec
import pandas as pd

mopdelfilePath = 'E:\\ML_learning\\NLP_data\\model.bin'
fileSegWordDonePath = 'E:\\ML_learning\\NLP_data\\corpusSegDone.txt'

fileTrainRead = pd.read_csv(fileSegWordDonePath)
train_sentences = pd.Series(fileTrainRead.iloc[:, 1])
f = lambda x: str(x).split(" ")
train_sentences = train_sentences.apply(f)

model = word2vec.Word2Vec(train_sentences, size=100)
model.save(mopdelfilePath)