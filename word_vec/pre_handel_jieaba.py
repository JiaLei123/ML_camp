import jieba

filePath = 'corpus.txt'
fileSegWordDonePath = 'corpusSegDone.txt'
fileTrainRead = []
with open(filePath, encoding="utf-8") as fileTrainRaw:
    fileTrainRead = [line[9:-11] for line in fileTrainRaw.readlines()]

fileTrainSeg = []
for line in fileTrainRead:
    fileTrainSeg.append(" ".join(list(jieba.cut(line, cut_all=False))))

with open(fileSegWordDonePath, 'wb') as fw:
    for line in fileTrainSeg:
        fw.write(line + "\n")