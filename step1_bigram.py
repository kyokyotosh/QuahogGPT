"""Step 1: a character-level bigram language model.

Each character predicts the next one using a single lookup table of scores.
No context beyond the previous character, so this is the smallest complete
language model. The same pieces reappear in the GPT later:

    text -> token ids -> embeddings -> logits -> cross-entropy loss -> sampling
"""

import math

import torch
import torch.nn.functional as F

torch.manual_seed(1337)

# 1. Tokenizer: every unique character gets an integer id.
text = open("data/seed.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)

# 2. Training pairs: the input at position t is data[t], the target is data[t+1].
x = data[:-1]
y = data[1:]


# 3. The model: a (vocab x vocab) table. Row i holds the scores for what comes
#    after character i. The embedding layer is just a learnable lookup table.
class BigramLM(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # shape (..., vocab_size), raw unnormalized scores
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n_new):
        for _ in range(n_new):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # scores -> probabilities
            next_id = torch.multinomial(probs, num_samples=1)  # sample one char
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# 4. Training. A random model predicts uniformly, so its loss starts near ln(vocab_size).
model = BigramLM(vocab_size)
print(f"vocab size: {vocab_size}")
print(f"random-guess loss = ln({vocab_size}) = {math.log(vocab_size):.4f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
for step in range(3000):
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()  # gradients of loss w.r.t. every table entry
    optimizer.step()  # nudge the table to lower the loss
    if step % 500 == 0 or step == 2999:
        print(f"step {step:5d}  loss {loss.item():.4f}")

# 5. Sampling: start from character 0 and generate 200 characters.
start = torch.zeros((1, 1), dtype=torch.long)
out = model.generate(start, 200)
print("\n--- sample ---")
print(decode(out[0].tolist()))
