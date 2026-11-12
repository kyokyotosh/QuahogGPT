"""Step 4: train the GPT on the Family Guy corpus, using BPE tokens.

Compared with step 2, the data is real (wiki corpus) and the tokens are BPE
subwords. The model keeps the best checkpoint by validation loss, so overfitting
can't ruin the saved model.
"""

import json
import pathlib

import torch

from gpt import GPT, GPTConfig
from step3_bpe import build_vocab, decode, encode

torch.manual_seed(1337)

CORPUS_PATH = pathlib.Path("data/corpus.txt")
TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
CKPT_PATH = pathlib.Path("checkpoints/quahog_gpt_v3.pt")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer and corpus.
merges = {(a, b): idx for a, b, idx in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
vocab = build_vocab(merges)
text = CORPUS_PATH.read_text(encoding="utf-8")
data = torch.tensor(encode(text, merges), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"{len(data):,} tokens  (train {len(train_data):,}, val {len(val_data):,})")

# Hyperparameters. Corpus is ~1.7M tokens. v1 (128-wide, 4-layer, 1.07M params) reached
# val 2.97 at 6000 steps and was still improving; this model is ~2.6M params, trained on CPU.
cfg = GPTConfig(vocab_size=len(vocab), block_size=128, n_embd=192, n_head=6, n_layer=6, dropout=0.2)
batch_size = 32
lr = 3e-3
weight_decay = 0.1
max_iters = 12000
eval_interval = 250
eval_iters = 40


def get_batch(source):
    starts = torch.randint(len(source) - cfg.block_size, (batch_size,))
    x = torch.stack([source[i : i + cfg.block_size] for i in starts])
    y = torch.stack([source[i + 1 : i + cfg.block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    model.eval()  # disables dropout for a fair measurement
    out = {}
    for name, source in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


model = GPT(cfg).to(device)
print(f"device: {device}")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
best_val = float("inf")

for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss(model)
        marker = ""
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"config": cfg.__dict__, "model": model.state_dict()}, CKPT_PATH)
            marker = "  (saved)"
        print(f"step {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}{marker}")
    if it == max_iters:
        break
    x, y = get_batch(train_data)
    _, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Reload the best checkpoint (by val loss) and sample from it.
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"\nbest val loss: {best_val:.4f}")
for prompt in ["Peter", "Stewie", "The Griffins live"]:
    ids = torch.tensor([encode(prompt, merges)], dtype=torch.long, device=device)
    out = model.generate(ids, 150)
    print(f"\n--- prompt: {prompt!r} ---")
    print(decode(out[0].tolist(), vocab))
