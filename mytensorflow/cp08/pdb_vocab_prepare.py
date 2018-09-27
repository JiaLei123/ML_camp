import collections
import sys
from operator import itemgetter

RAW_DATA = r"E:\ML_learning\tensorFlow\simple-examples\data\ptb.train.txt"
OUTPUT_VOCAB = 'ptd.vocab'

counter = collections.Counter()
with open(RAW_DATA, 'r', encoding='utf-8') as f:
    for line in f:
        counter.update(line.strip().split())

sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

sorted_words = ["<eos>"] + sorted_words

with open(OUTPUT_VOCAB, 'w', encoding='utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
