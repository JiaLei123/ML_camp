import jieba

filePath = 'test_out.sq3.pointaddr.txt'
fileSegWordDonePath = 'test_jiebao.sq3.pointaddr.txt'
fileTrainRead = []
with open(filePath, encoding="utf-8") as fileTrainRaw:
    fileTrainRead = [line for line in fileTrainRaw.readlines() if line]


fileTrainSeg = []
# fileTrainSeg.append(" ".join(list(jieba.cut(fileTrainRead[1], cut_all=False))))
for line in fileTrainRead:
    fileTrainSeg.append("|".join(list(jieba.cut(line, cut_all=False))))

with open(fileSegWordDonePath, 'w', encoding='utf-8') as fw:
    for line in fileTrainSeg:
        fw.write(line + "\n")