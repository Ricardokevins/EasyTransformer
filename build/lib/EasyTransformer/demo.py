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