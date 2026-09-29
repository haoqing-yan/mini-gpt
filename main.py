import torch
import torch.nn as nn
import math

text = "hello world hello world"

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encoded = [stoi[c] for c in text]

print(encoded)

vocab_size = len(chars)
d_model = 16

embedding = nn.Embedding(vocab_size, d_model)

x = torch.tensor(encoded[:5])

out = embedding(x)

print(out.shape)

Wq = nn.Linear(d_model, d_model, bias=False)
Wk = nn.Linear(d_model, d_model, bias=False)
Wv = nn.Linear(d_model, d_model, bias=False)

Q = Wq(out)
K = Wk(out)
V = Wv(out)

scores = Q @ K.T

scores = scores / math.sqrt(d_model)

mask = torch.tril(torch.ones(5, 5))

scores = scores.masked_fill(mask == 0, float("-inf"))

weights = torch.softmax(scores, dim=-1)

attention_output = weights @ V

Q = Wq(x)
K = Wk(x)
V = Wv(x)

scores = Q @ K.T
scores = scores / math.sqrt(d_model)

scores = scores.masked_fill(mask == 0, float("-inf"))

weights = torch.softmax(scores, dim=-1)

out = weights @ V

lm_head = nn.Linear(d_model, vocab_size)

logits = lm_head(attention_output)

loss_fn = nn.CrossEntropyLoss()

target = torch.tensor(encoded[1:6])

loss = loss_fn(logits, target)

optimizer = torch.optim.Adam(
    list(embedding.parameters())
    + list(Wq.parameters())
    + list(Wk.parameters())
    + list(Wv.parameters())
    + list(lm_head.parameters()),
    lr=1e-3
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
