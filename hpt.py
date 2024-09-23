# Horus Heresy Pretrained Transformer
with open('all_text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f'Number of characters in dataset: {len(text)}')

# Unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Tokenise
