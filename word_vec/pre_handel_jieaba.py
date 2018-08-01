import jieba
import pandas as pd

filePath = 'D:\\ML_learning\\NLP_data\\corpus1.txt'
fileSegWordDonePath = 'D:\\ML_learning\\NLP_data\\corpusSegDone.txt'
fileTrainRead = pd.read_csv(filePath)
fileTrain = pd.Series(fileTrainRead.iloc[:,0])
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