#https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

first_1000 = text[:1000]
print(first_1000)

import tiktoken

enc = tiktoken.get_encoding('gpt2')
# print(enc.n_vocab)

# print(enc.encode(first_1000))

import torch
data = torch.tensor(enc.encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])