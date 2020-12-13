import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

'''
extract from https://github.com/ne7ermore/torch-light
Aim to get a simple implement of Bert without pretrain and deploy quickly
'''

PAD = 0
UNK = 1
SEP = 2
CLS = 3
MASK = 4


WORD = {
    UNK: '<unk>',
    PAD: '<pad>',
    SEP: '<sep>',
    CLS: '<cls>',
    MASK: '<mask>',
}


SEGMENTA = 1
SEGMENTB = 2

INIT_RANGE = 0.02

NOT_USE_WEIGHT_DECAY = ['bias', 'gamma', 'beta']
SPLIT_CODE = "@@@###@@@"

class BertTokenizer():
    def __init__(self,lines,max_wordn,max_len):
        self.max_wordn = max_wordn
        self.max_len = max_len
        self.build_dict(lines)
        
        
    def build_dict(self,sent):
        words = set([])
        for i in sent:
            if i not in words:
                words.add(i)
        words = list(words)
        words = words[:self.max_wordn-4]
        self.word2idx = {}
        self.idx2word = {}
        for pos,i in enumerate( words):
            self.word2idx[i] = pos
            self.idx2word[pos] = i
        self.word2idx['[OOV]'] = 1
        self.idx2word[1] = '[OOV]'
        self.word2idx['[PAD]'] = 0
        self.idx2word[0] = '[PAD]'
        self.word2idx['[CLS]'] = 2
        self.idx2word[2] = '[CLS]'
        self.word2idx['[SEP]'] = 3
        self.idx2word[3] = '[SEP]'
        print("Dict len: ", len(self.word2idx))
        
    def encode(self, sent,add_special_token=True):
        sent_idx = []
        tokens = sent.split(" ")
        if add_special_token:
            tokens=tokens[:self.max_len-2]
            tokens = ['CLS'] + tokens + ['[SEP]']
        else:
            tokens=tokens[:self.max_len]
        for i in tokens:
            if i in self.word2idx:
                sent_idx.append(self.word2idx[i])
            else:
                sent_idx.append(self.word2idx['[OOV]'])

        return sent_idx

    def encodepro(self, sent, add_special_token=True):
        indexed_tokens = self.encode(sent)
        segment_label = [SEGMENTA for _ in range(len(indexed_tokens))]
        while len(segment_label) <  self.max_len:
            segment_label.append(0)  # 0 is id for [PAD]
        #segment_label= torch.LongTensor(segment_label)
        pos = [pos + 1 if w != PAD else 0 for pos, w in enumerate(indexed_tokens)]
        while len(pos) <  self.max_len:
            pos.append(0)
        #pos= torch.LongTensor(pos)
        avai_len = len(indexed_tokens)
        while len(indexed_tokens) <  self.max_len:
            indexed_tokens.append(0)  # 0 is id for [PAD]
        indexed_tokens = indexed_tokens[: self.max_len]
        #indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        # att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        # att_mask[0, :avai_len] = 1
        return indexed_tokens,pos,segment_label
        
def position(n_position, d_model):
    position_enc = np.array([[pos / np.power(10000, 2 * i / d_model)
                              for i in range(d_model)] for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.sin(position_enc[1:, 1::2])

    return torch.from_numpy(position_enc).float()

def get_attn_padding_mask(seq_q):
    assert seq_q.dim() == 2
    bsz, len_q = seq_q.size()
    pad_attn_mask = seq_q.data.eq(PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(bsz, len_q, len_q)
    return pad_attn_mask


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, input):
        mu = torch.mean(input, dim=-1, keepdim=True)
        sigma = torch.std(input, dim=-1, keepdim=True).clamp(min=self.eps)
        output = (input - mu) / sigma
        return output * self.gamma.expand_as(output) + self.beta.expand_as(output)


class GELU(nn.Module):
    """
    different from 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWise(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            GELU(),
            nn.Conv1d(d_ff, d_model, 1),
            nn.Dropout(dropout)
        )
        self.lm = LayerNorm(d_model)

    def forward(self, input):
        residual = input
        out = self.seq(input.transpose(1, 2)).transpose(1, 2)
        return self.lm(residual + out)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.temper = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, attn_mask):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn.view(-1, attn.size(2))).view(*attn.size())
        attn = self.dropout(attn)
        return torch.bmm(attn, v)


class MultiHeadAtt(nn.Module):
    def __init__(self, n_head, d_model, dropout):
        super().__init__()
        self.n_head = n_head
        self.d_v = self.d_k = d_k = d_model // n_head

        for name in ["w_qs", "w_ks", "w_vs"]:
            self.__setattr__(name,
                             nn.Parameter(torch.FloatTensor(n_head, d_model, d_k)))

        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.lm = LayerNorm(d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        self.reset_parameters()

    def forward(self, q, k, v, attn_mask):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q

        bsz, len_q, d_model = q.size()
        len_k, len_v = k.size(1), v.size(1)

        def reshape(x):
            """[bsz, len, d_*] -> [n_head x (bsz*len) x d_*]"""
            return x.repeat(n_head, 1, 1).view(n_head, -1, d_model)

        q_s, k_s, v_s = map(reshape, [q, k, v])

        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)

        outputs = self.attention(q_s, k_s, v_s, attn_mask.repeat(n_head, 1, 1))
        outputs = torch.cat(torch.split(outputs, bsz, dim=0),
                            dim=-1).view(-1, n_head * d_v)
        outputs = F.dropout(self.w_o(outputs),
                            p=self.dropout).view(bsz, len_q, -1)
        return self.lm(outputs + residual)

    def reset_parameters(self):
        self.w_qs.data.normal_(std=INIT_RANGE)
        self.w_ks.data.normal_(std=INIT_RANGE)
        self.w_vs.data.normal_(std=INIT_RANGE)
        self.w_o.weight.data.normal_(std=INIT_RANGE)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super().__init__()
        self.mh = MultiHeadAtt(n_head, d_model, dropout)
        self.pw = PositionWise(d_model, d_ff, dropout)

    def forward(self, enc_input, slf_attn_mask):
        enc_output = self.mh(
            enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = self.pw(enc_output)
        return enc_output


class Pooler(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.linear = nn.Linear(d_model, d_model)
        self.linear.weight.data.normal_(std=INIT_RANGE)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x[:, 0])
        return F.tanh(x)


class BERT(nn.Module):
    def __init__(self, vacabulary_size=30000,d_model=768,dropout=0.1,max_len=512,n_stack_layers=12,d_ff=3072,n_head=12):
        super().__init__()

        n_position = max_len + 1

        self.enc_ebd = nn.Embedding(vacabulary_size, d_model)
        self.seg_ebd = nn.Embedding(3, d_model)
        self.pos_ebd = nn.Embedding(n_position, d_model)
        self.pos_ebd.weight.data = position(n_position, d_model)
        self.pos_ebd.weight.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.ebd_normal = LayerNorm(d_model)
        self.out_normal = LayerNorm(d_model)

        self.encodes = nn.ModuleList([EncoderLayer(
            d_model, d_ff, n_head, dropout) for _ in range(n_stack_layers)])

        self.pooler = Pooler(d_model)
        self.transform = nn.Linear(
            d_model, d_model)  # word hidden layer
        self.gelu = GELU()

    def reset_parameters(self):
        self.enc_ebd.weight.data.normal_(std=INIT_RANGE)
        self.seg_ebd.weight.data.normal_(std=INIT_RANGE)

        self.transform.weight.data.normal_(std=INIT_RANGE)
        self.transform.bias.data.zero_()

    def forward(self, inp, pos, segment_label):
        encode = self.enc_ebd(
            inp) + self.seg_ebd(segment_label) + self.pos_ebd(pos)

        encode = self.dropout(self.ebd_normal(encode))

        slf_attn_mask = get_attn_padding_mask(inp)

        for layer in self.encodes:
            encode = layer(encode, slf_attn_mask)

        sent_encode = self.pooler(encode)
        word_encode = self.out_normal(self.gelu(self.transform(encode)))

        return word_encode, sent_encode

    def get_trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def parameters_count(self):
        return sum(x.numel() for x in self.parameters())

    def save_model(self, args, data, path="model.pt"):
        torch.save({
            "args": args,
            "weights": self.state_dict(),
            "dict": data["dict"],
            "max_len": data["max_len"]
        }, path)

    def load_model(self, weights):
        self.load_state_dict(weights)
        self.cuda()


