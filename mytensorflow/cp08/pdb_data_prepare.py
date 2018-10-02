import sys
import numpy as np

RAW_DATA = r"E:\ML_learning\tensorFlow\simple-examples\data\ptb.test.txt"
VOCAB = "ptd.vocab"
OUTPUT_DATA = 'ptd.test'

with open(VOCAB, 'r', encoding='utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]

word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}


def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


fin = open(RAW_DATA, 'r', encoding='utf-8')
fout = open(OUTPUT_DATA, 'w', encoding='utf-8')
for line in fin:
    words = line.strip().split() + ["<eos>"]
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()