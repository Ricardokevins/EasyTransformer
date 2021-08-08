# EasyTransformer

Simple implement of Transformer Baseline and BERT **extracted from other repo**

Both Model are unpretrained and random initialize   

 <br/> 

BERT from https://github.com/ne7ermore/torch-light

Transformer from https://github.com/whr94621/NJUNMT-pytorch    

<br/>



Create this repo to provide simple and easy quick start of using **Transformer** and **Bert** as baseline

you can have a quick start with 

```
pip install EasyTransformer
# new version is 1.2.3
```



<br/> Here is a simple demo（Also available in demo.py）

```Python
class Transformer():
    def __init__(self , n_src_vocab=30000,max_length=512, n_layers=6, n_head=8, d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1, dim_per_head=None):
        super().__init__()
        self.n_src_vocab = n_src_vocab
        self.max_length = max_length
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_word_vec = d_word_vec
        self.d_model = d_model
        self.d_inner_hid = d_inner_hid
        self.dropout = dropout
        self.dim_per_head=dim_per_head
        print("==== Transformer Init successfully ====")

    def get_base_tokenzier(self, corpus):
        from EasyTransformer.tokenizer import Tokenizer
        self.TransformerTokenizer = Tokenizer(self.n_src_vocab, self.max_length, corpus)
        return self.TransformerTokenizer
    
    def get_BPE_tokenizer(self,corpus):
        from EasyTransformer.tokenizer import BPE_Tokenizer
        self.TransformerTokenizer = BPE_Tokenizer(self.n_src_vocab, self.max_length, corpus)
        return self.TransformerTokenizer

    def get_Char_tokenizer(self,corpus):
        from EasyTransformer.tokenizer import Char_Tokenizer
        self.TransformerTokenizer = Char_Tokenizer(self.n_src_vocab, self.max_length, corpus)
        return self.TransformerTokenizer

    def get_model(self):
        self.TransformerModel = TransformerEncoder(self.n_src_vocab, self.n_layers, self.n_head, self.d_word_vec, self.d_model, self.d_inner_hid, self.dropout, dim_per_head=self.dim_per_head)
        return self.TransformerModel
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

