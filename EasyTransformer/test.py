import transformer

def get_dataset():
    texts = []
    labels = []
    with open('trains.txt') as f:
        li = []
        while True:
            content = f.readline().replace('\n', '')
            if not content:              #为空行，表示取完一次数据（一次的数据保存在li中）
                if not li:               #如果列表也为空，则表示数据读完，结束循环
                    break
                label = li[0][10]
                text = li[1][6:-7]
                texts.append(text)
                labels.append(int(label))
                li = []
            else:
                li.append(content)       #["<Polarity>标签</Polarity>", "<text>句子内容</text>"]
    return texts, labels

test_data,_ = get_dataset()
model = transformer.Transformer(max_length = 16)
#tokenizer = model.get_base_tokenzier(test_data)
tokenizer = model.get_BPE_tokenizer(test_data)

test_data = [
    'I love NLP.',
    'NLP is interesting.',
    'He said: " how are you doing". ',
    'If I could, I surely would.',
    'The worst way to miss someone is to be sitting right beside them knowing you can’t have them.'
    ]

for i in test_data:
    print(tokenizer.encode(i))

# tokenizer.encode(".")