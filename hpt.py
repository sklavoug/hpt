###########################################
### Horus Heresy Pretrained Transformer ###
###########################################

# I've seen the video (https://www.youtube.com/watch?v=kCc8FmEb1nY&t=652s) before but I'm interested to try it after doing
# a lot of the base work on the Horus Heresy book series. A few reasons I think this might be interesting:
# 1. It's a lot more data than the dataset in the video
# 2. It's written by multiple authors with some differences in writing style, but all iterating on a theme
# 3. It's consistent enough that it should be clear whether the model is working correctly or not based on the output

import torch

status = False

def printer(to_print, status=False):
    if status == True:
        print(to_print)

# Reading the file (not uploaded for copyright reasons)
with open('all_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Quite a lot of characters, almost 42 million total
printer(f'Number of characters in dataset: {len(text)}', status)

# A lot more unique characters too
chars = sorted(list(set(text)))
vocab_size = len(chars)
printer(''.join(chars), status)
printer(vocab_size, status)

# Tokenise: characters to integers and back
strings = {}
ints = {}

for idx, i in enumerate(chars):
    strings[i] = idx
    ints[idx] = i

# Char to int
def encode(chars):
    ret = []
    for letter in chars:
        ret.append(strings[letter])
    return ret

# Int to char
def decode(idx):
    ret = ''
    for id in idx:
        ret += ints[id]
    return ret

# Set up the tensor converting text to integers and split into train and validation
data = torch.tensor(encode(text), dtype=torch.long)

trainval_split = int(0.9*len(data))
train_data = data[:trainval_split]
val_data = data[trainval_split:]

# The batch size is context length, i.e., what the maximum length for predictions is. We take batch_size+1 so it has 9 total (so it can predict the 9th character)
# Kind of a sliding window where y (target) is always 1 step ahead of x (train)
# The batch size is set to 4, that's just how many independent random samples we process at a time
torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]