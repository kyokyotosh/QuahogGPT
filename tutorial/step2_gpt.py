"""Step 2: a tiny GPT.

The bigram model only saw one previous character. A GPT lets each position
look at every earlier position in a window, through causal self-attention,
then processes the result with an MLP. Stacking these blocks gives the
decoder-only transformer behind GPT-style models.

    tokens -> embeddings (token + position) -> [attention, MLP] x N -> logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# Hyperparameters
block_size = 32  # context length: how many previous characters the model sees
batch_size = 32
n_embd = 64  # width of the residual stream (every position is a vector this size)
n_head = 4  # attention heads; each head works on n_embd // n_head dimensions
n_layer = 2  # number of transformer blocks
dropout = 0.1
lr = 3e-3
max_iters = 3000
eval_interval = 500
eval_iters = 50

# Data: same tokenizer as step 1, plus a held-out validation split.
text = open("data/seed.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
split_at = int(0.9 * len(data))
train_data, val_data = data[:split_at], data[split_at:]


def get_batch(split):
    """Sample random windows. x is the input; y is x shifted by one (the next char at every position)."""
    src = train_data if split == "train" else val_data
    starts = torch.randint(len(src) - block_size, (batch_size,))
    x = torch.stack([src[i : i + block_size] for i in starts])
    y = torch.stack([src[i + 1 : i + block_size + 1] for i in starts])
    return x, y


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular mask: position t may attend to positions <= t only.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size): what each position offers
        q = self.query(x)  # (B, T, head_size): what each position is looking for
        # Affinity of every query with every key. Scaling keeps softmax from saturating.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # block future positions
        wei = self.dropout(F.softmax(wei, dim=-1))  # each row sums to 1
        v = self.value(x)  # (B, T, head_size): what each position contributes
        return wei @ v  # each position: weighted average of values it attends to


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated, then projected back to n_embd."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-position MLP: each position processes what attention gathered."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communicate (attention), then compute (MLP). Residual connections keep gradients flowing."""

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)  # one learned vector per position 0..block_size-1
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))  # (B, T, n_embd)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # the model can only see the last block_size characters
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    """Average loss over several random batches, for train and val."""
    model.eval()  # turns off dropout for a fair measurement
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


model = TinyGPT()
n_params = sum(p.numel() for p in model.parameters())
print(f"vocab={vocab_size}  params={n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

start = torch.zeros((1, 1), dtype=torch.long)
print("\n--- sample ---")
print(decode(model.generate(start, 300)[0].tolist()))
