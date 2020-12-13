# EasyTransformer

Simple implement of Transformer Baseline and BERT **extracted from other repo**

Both Model are unpretrained and random initialize    <br/> 

BERT from https://github.com/ne7ermore/torch-light

Transformer from https://github.com/whr94621/NJUNMT-pytorch   <br/> 



Create this repo to provide simple and easy quick start of using **Transformer** and **Bert** as baseline

you can have a quick start with 

```
pip install EasyTransformer
# new version is 0.0.4
```



<br/> Here is a simple demo（Also available in demo.py）

```Python
import EasyTransformer
from EasyTransformer import bert
from EasyTransformer import transformer
import torch

lines =[
        "I love NJU",
        "Good morning"
]
Encoder = bert.BERT()
tokenizer=bert.BertTokenizer(lines,30000,512)
text = []
position = []
segment=[]
indexed_tokens, pos, segment_label = tokenizer.encodepro(lines[0])
text.append(indexed_tokens)
position.append(pos)
segment.append(segment_label)
indexed_tokens, pos, segment_label = tokenizer.encodepro(lines[1])
text.append(indexed_tokens)
position.append(pos)
segment.append(segment_label)

text= torch.tensor(text)
position = torch.tensor(position)
segment = torch.tensor(segment)
out1,out2 = Encoder(text,position,segment)
print(out1.shape)
print(out2.shape)


Encoder =  transformer.TransformerEncoder(30000)
tokenizer= transformer.TransformerTokenizer(30000,512,lines)
text = []
indexed_tokens= tokenizer.encode(lines[0])
text.append(indexed_tokens)

indexed_tokens= tokenizer.encode(lines[1])
text.append(indexed_tokens)
text= torch.tensor(text)
out1,out2 = Encoder(text)
print(out1.shape)
print(out2.shape)
```



<br/> Here are some parameter you can choose to get your own Model as you like

**TransformerTokenizer**

```
def __init__(self, max_wordn,max_length, lines)
```

**TransformerEncoder**

```
def __init__(
            self, n_src_vocab, n_layers=6, n_head=8,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None)
```



**BertTokenizer**

```
def __init__(self,lines,max_wordn,max_len)
```

**BERT**

```
def __init__(self, vacabulary_size=30000,d_model=768,dropout=0.1,max_len=512,n_stack_layers=12,d_ff=3072,n_head=12):
```

