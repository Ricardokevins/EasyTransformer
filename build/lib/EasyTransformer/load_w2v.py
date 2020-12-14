from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import gensim



import pandas as pd
from transformers import AdamW
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.tsv', delimiter='\t', header=None)
#print(df.head())
#tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
sent = []
for i in df[0]:
    sent.append(i)
print(len(sent))
from EasyTransformer import transformer
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.50d.word2vec.txt', binary=False, encoding='utf-8')
tokenizer = transformer.TransformerTokenizer(20000, 64, sent)

idx_to_word=tokenizer.idx2word
word_to_idx=tokenizer.word2idx
import numpy as np
import torch
projector=torch.nn.Linear(50,512)
def pre_weight(vocab_size):
    weight = torch.zeros(vocab_size,512)
    #初始权重
    hit=0
    for i in range(len(word2vec_model.index2word)):#预训练中没有word2ix，所以只能用索引来遍历
        if word2vec_model.index2word[i] in word_to_idx:
            index = word_to_idx[word2vec_model.index2word[i]]  #得到预训练中的词汇的新索引
            if index == 0 or index == 1:
                continue
            #print(index,word2vec_model.index2word[i],idx_to_word[index])
            weight[index, :] = projector(torch.from_numpy(word2vec_model.get_vector(idx_to_word[word_to_idx[word2vec_model.index2word[i]]])))
            hit += 1
    print(hit)
    return weight

result = pre_weight(20000)
print(result.shape)
print(result)