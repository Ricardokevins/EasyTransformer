import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.tsv', delimiter='\t', header=None)
#print(df.head())
#tokenized = df[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
sent = []
for i in df[0]:
    sent.append(i)
print(len(sent))
import transformer
import bert
#tokenizer = transformer.TransformerTokenizer(30000, 256, sent)
tokenizer=bert.BertTokenizer(sent,6000,128)
for i in range(5):
    print(tokenizer.idx2word[i])
exit()
labels = df[1]

text = []
position = []
segment = []
for i in sent:
    indexed_tokens, pos, segment_label = tokenizer.encodepro(i)
    text.append(indexed_tokens)
    position.append(pos)
    segment.append(segment_label)


#train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
import torch

text = torch.tensor(text)
position = torch.tensor(position)
segment = torch.tensor(segment)
labels = torch.tensor(labels)

import torch.utils.data
batch_size=16
dataset = torch.utils.data.TensorDataset(text,position,segment, labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


import torch.nn as nn
import math
import torch.nn.functional as F
from EasyTransformer import transformer
criterion = nn.CrossEntropyLoss()
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        #self.Encoder = transformer.TransformerEncoder(30000)
        self.Encoder = bert.BERT(6000,max_len=128)
        self.linear=nn.Linear(768,2)

    def forward(self, text,position,segment):
        word_vec,sent_vec = self.Encoder(text,position,segment)
        return self.linear(sent_vec)

model = Model().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.04)
from util import ProgressBar
pbar = ProgressBar(n_total=len(train_iter), desc='Training')
for epoch in range(5):
    correct = 0
    total=0
    iter = 0
    avg_loss=0
    for a,b,c,y in train_iter:
        iter += 1
        optimizer.zero_grad()
        if a.size(0)!=batch_size:
            break
        a = a.reshape(batch_size,-1).cuda()
        b = b.reshape(batch_size, -1).cuda()
        c = c.reshape(batch_size, -1).cuda()
        y = y.cuda()
        
        state= model(a,b,c)
        loss = criterion(state, y)
        loss.backward()
        optimizer.step()
        correct += state.argmax(dim=-1).eq(y).sum().item()
        total += a.size(0)
        avg_loss+=loss.item()
        # print(state.argmax(dim=-1))
        # print(y)
        # print(state.argmax(dim=-1).eq(y).sum().item())
        pbar(iter, {'loss':avg_loss/iter})
    print("Acc: ",correct,"  ",correct/total)
