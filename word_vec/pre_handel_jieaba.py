import jieba
import pandas as pd

filePath = 'E:\\ML_learning\\NLP_data\\corpus_1.txt'
fileSegWordDonePath = 'E:\\ML_learning\\NLP_data\\corpusSegDone.txt'
fileTrainRead = pd.read_csv(filePath)
fileTrain = pd.Series(fileTrainRead.values.reshape((-1)))
f = lambda x: x[9:-11]
fileTrain = fileTrain.apply(f)

fileTrain.dropna(how='any')

fileTrainSeg = []
for line in fileTrain:
    data = jieba.cut(line, cut_all=False)
    # print(list(data))
    fileTrainSeg.append(" ".join(list(data)))

output_list = pd.Series(fileTrainSeg)
output_list.to_csv(fileSegWordDonePath, encoding='utf-8')