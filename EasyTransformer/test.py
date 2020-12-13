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
tokenizer = transformer.TransformerTokenizer(20000, 32, sent)
#tokenizer=bert.BertTokenizer(sent,20000,128)
# for i in range(10):
#     print(tokenizer.idx2word[i])

labels = df[1]

text = []
# position = []
# segment = []
for i in sent:
    #indexed_tokens, pos, segment_label = tokenizer.encodepro(i)
    indexed_tokens = tokenizer.encode(i)
    text.append(indexed_tokens)

    # position.append(pos)
    # segment.append(segment_label)



#train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
import torch

text = torch.tensor(text)
# position = torch.tensor(position)
# segment = torch.tensor(segment)
labels = torch.tensor(labels)

import torch.utils.data
batch_size=128
dataset = torch.utils.data.TensorDataset(text, labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


import torch.nn as nn
import math
import torch.nn.functional as F
import transformer
criterion = nn.CrossEntropyLoss()
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Encoder = transformer.TransformerEncoder(30000)
        #self.Encoder = bert.BERT(20000,max_len=128)
        self.linear=nn.Linear(512,2)

    def forward(self, text):
        word_vec,sent_vec = self.Encoder(text)
        return self.linear(sent_vec)

model = Model().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)


for epoch in range(5):
    correct = 0
    total=0
    iter = 0
    avg_loss=0
    for a,y in train_iter:
        iter += 1
        optimizer.zero_grad()
        if a.size(0)!=batch_size:
            break
        a = a.reshape(batch_size,-1).cuda()
        # b = b.reshape(batch_size, -1).cuda()
        # c = c.reshape(batch_size, -1).cuda()
        y = y.cuda()
        
        state= model(a)
        loss = criterion(state, y)
        loss.backward()
        optimizer.step()
        correct += state.argmax(dim=-1).eq(y).sum().item()
        total += a.size(0)
        avg_loss+=loss.item()
        # print(state.argmax(dim=-1))
        # print(y)
        # print(state.argmax(dim=-1).eq(y).sum().item())
       
    print("Acc: ",correct,"  ",correct/total)
