# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unicodedata
from io import open
def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.
        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        
        # TODO:disabled!!!!!
        #text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for i,token in enumerate(orig_tokens):
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            #split_tokens.append(token)
            #split_tokens.extend([(i,t) for t in self._run_split_on_punc(token)])
            split_tokens.extend(self._run_split_on_punc(token))
            #output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return split_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

Special_Tokens = ['[PAD]','[OOV]','[<s>]','[/<s>]','[MASK]']
class Tokenizer():
    def __init__(self, max_wordn,max_length,lines):
        self.max_wordn = max_wordn
        self.max_length = max_length
        self.divide = BasicTokenizer()
        self.word2idx = {}
        self.idx2word = {}
        self.build_dict(lines)
        
    def build_dict(self, sents):
        import os
        from collections import Counter
        if os.path.exists('dict.txt'):
            print("------------------ Using exsit dict ------------------")
            f = open('dict.txt', 'r',encoding='utf-8')
            lines = f.readlines()
            index =0 
            for i in lines:
                word = i.replace('\n', '').split('  ')[0]
                self.word2idx[word] = index
                self.idx2word[index] = word
                index+=1
            print("Dict len: ", len(self.word2idx))
        else:        
            all_vocab = []
            words = set([])
            print("------------------ Building New Dict ------------------")
            from tqdm import tqdm
            for sent in tqdm(sents):
                sent=self.divide.tokenize(sent)
                all_vocab.extend(sent)
            counter = Counter(all_vocab) 
            count_pairs = counter.most_common(self.max_wordn-5) 
            words, _ = list(zip(*count_pairs))
            _ = [1,1,1,1,1]+list(_)
            
            words = Special_Tokens +  list(words)

            for pos,i in enumerate(words):
                self.word2idx[i] = pos
                self.idx2word[pos] = i
            
            print("Dict len: ", len(self.word2idx))
            f = open('dict.txt','w',encoding='utf-8')
            for i in range(len(self.word2idx)):
                f.write(self.idx2word[i]+"\t" +  str(_[i]) + '\n')
            f.close()
    
    def cut(self, sent):
        return self.divide.tokenize(sent)
        
    def encode(self, sent):
        sent_idx = []
        sent=self.divide.tokenize(sent)
        sent = sent[: self.max_length]
        
        for i in sent:
            if i in self.word2idx:
                sent_idx.append(self.word2idx[i])
            else:
                sent_idx.append(self.word2idx['[OOV]'])
        while len(sent_idx) < self.max_length:
            sent_idx.append(0)
        return sent_idx

#Code Borrowed and finetued from https://wmathor.com/index.php/archives/1517/
import re, collections
class BPE_Tokenizer():
    def __init__(self, max_wordn,max_length,lines):
        self.max_wordn = max_wordn
        self.max_length = max_length
        self.divide = BasicTokenizer()
        import os
        if os.path.exists('bpe.txt'):
            self.sorted_tokens = []
            print("------------------ Using exsit dict ------------------")
            f = open('bpe.txt', 'r',encoding='utf-8')
            lines = f.readlines()
            index =0 
            for i in lines:
                word = i.replace('\n', '').split('  ')[0]
                self.sorted_tokens.append(word)
            print("Dict len: ", len(self.sorted_tokens))
        else:
            print("------------------ Building New Dict ------------------")
            self.vocab = self.get_vocab(lines)
            self.Learn_bpe()
        
    def get_vocab(self,lines):
        vocab = collections.defaultdict(int)
        from tqdm import tqdm
        for sent in tqdm(lines):
            sent=self.divide.tokenize(sent)
            for word in sent:
                vocab[' '.join(list(word)) + ' </w>'] += 1
        return vocab

    def get_tokens_from_vocab(self,vocab):
        tokens_frequencies = collections.defaultdict(int)
        vocab_tokenization = {}
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens_frequencies[token] += freq
            vocab_tokenization[''.join(word_tokens)] = word_tokens
        return tokens_frequencies, vocab_tokenization

    def get_stats(self,vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self,pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def measure_token_length(self,token):
        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)

    def Learn_bpe(self):
        while True:
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(best, self.vocab)
            tokens_frequencies, vocab_tokenization = self.get_tokens_from_vocab(self.vocab)
            if len(tokens_frequencies.keys()) == self.max_wordn - 5:
                break
        
        sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (self.measure_token_length(item[0]), item[1]), reverse=True)
        self.sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
        self.sorted_tokens = Special_Tokens + self.sorted_tokens
        f = open('bpe.txt','w',encoding='utf-8')
        for i in self.sorted_tokens:
            f.write(i+'\n')
        f.close()
        

    def tokenize_word(self,string, sorted_tokens,unknown_token='[OOV]'):
        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token)
            #token_reg = re.escape(token.replace('.', '[.]'))
            #print(string,token_reg)
            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            #print(matched_positions)
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]
            
            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            break
        if len(string_tokens) == 0:
            return [unknown_token]
        return string_tokens

    def encode(self, sent):
        sent_idx = []
        sent=self.divide.tokenize(sent)
        tokens = []
        for i in sent:
            token = i+"</w>"
            result = self.tokenize_word(token,self.sorted_tokens)
            for j in result:
                tokens.append(self.sorted_tokens.index(j))

        tokens = tokens[: self.max_length]
        while len(tokens) < self.max_length:
            tokens.append(0)
        return tokens

class Char_Tokenizer():
    def __init__(self, _ , max_length,lines):
        self.max_length = max_length
        self.divide = BasicTokenizer()
        self.word2idx = {}
        self.idx2word = {}
        self.build_dict(lines)
        
    def build_dict(self, sents):
        import os
        from collections import Counter
        if os.path.exists('char.txt'):
            print("------------------ Using exsit dict ------------------")
            f = open('char.txt', 'r',encoding='utf-8')
            lines = f.readlines()
            index =0 
            for i in lines:
                word = i.replace('\n', '').split('  ')[0]
                self.word2idx[word] = index
                self.idx2word[index] = word
                index+=1
            print("Dict len: ", len(self.word2idx))
        else:        
            all_vocab = []
            words = set([])
            print("------------------ Building New Dict ------------------")
            from tqdm import tqdm
            for sent in tqdm(sents):
                sent=self.divide.tokenize(sent)    
                for j in sent:
                    all_vocab.extend([i for i in j])
            words = list(set(all_vocab))

            words = Special_Tokens +  list(words)

            for pos,i in enumerate(words):
                self.word2idx[i] = pos
                self.idx2word[pos] = i
            
            print("Dict len: ", len(self.word2idx))
            f = open('char.txt','w',encoding='utf-8')
            for i in range(len(self.word2idx)):
                f.write(self.idx2word[i] + '\n')
            f.close()

    def encode(self, sent):
        sent_idx = []
        sent=self.divide.tokenize(sent)

    
        for j in sent:
            for i in j:
                if i in self.word2idx:
                    sent_idx.append(self.word2idx[i])
                else:
                    sent_idx.append(self.word2idx['[OOV]'])
        
        sent_idx = sent_idx[: self.max_length]
        while len(sent_idx) < self.max_length:
            sent_idx.append(0)
        return sent_idx