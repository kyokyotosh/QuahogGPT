"""Step 4: train the GPT on the Family Guy corpus, using BPE tokens.

Compared with step 2, the data is real (wiki corpus) and the tokens are BPE
subwords. The model keeps the best checkpoint by validation loss, so overfitting
can't ruin the saved model.

Usage (from the repo root):
    python -m pretrain.train                                  # v5 defaults
    python -m pretrain.train --max-iters 24000 --ckpt checkpoints/quahog_gpt_v7_long.pt
    python -m pretrain.train --ckpt checkpoints/quahog_gpt_v8.pt --tokenizer data/tokenizer_4k.json \\
        --block-size 512 --batch-size 16 --n-embd 256 --n-head 8 --n-layer 8 --lr 1e-3 --max-iters 20000
"""

import argparse
import json
import math
import pathlib

import torch

from quahog.gpt import GPT, GPTConfig
from quahog.bpe import build_vocab, decode, encode

torch.manual_seed(1337)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/quahog_gpt_v5.pt")
parser.add_argument("--n-embd", type=int, default=192)
parser.add_argument("--n-head", type=int, default=6)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--max-iters", type=int, default=12000)
parser.add_argument("--block-size", type=int, default=128)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-3, help="peak learning rate")
parser.add_argument("--tokenizer", default="data/tokenizer.json")
parser.add_argument("--corpus", default="data/corpus.txt")
args = parser.parse_args()
CKPT_PATH = pathlib.Path(args.ckpt)
CORPUS_PATH = pathlib.Path(args.corpus)
TOKENIZER_PATH = pathlib.Path(args.tokenizer)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load tokenizer and corpus.
merges = {(a, b): idx for a, b, idx in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
vocab = build_vocab(merges)
text = CORPUS_PATH.read_text(encoding="utf-8")
data = torch.tensor(encode(text, merges), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"{len(data):,} tokens  (train {len(train_data):,}, val {len(val_data):,})")

# Hyperparameters. Defaults are v5/v7: 192-wide, 6 layers, 128-token window, batch 32,
# peak LR 3e-3. The v8 QA model uses a 512-token window and a 4096-token vocabulary.
cfg = GPTConfig(vocab_size=len(vocab), block_size=args.block_size, n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer, dropout=0.2)
batch_size = args.batch_size
lr = args.lr  # peak LR; cosine decays to 10% of peak at max_iters
weight_decay = 0.1
max_iters = args.max_iters
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
            torch.save({"config": cfg.__dict__, "model": model.state_dict(), "tokenizer": str(TOKENIZER_PATH)}, CKPT_PATH)
            marker = "  (saved)"
        print(f"step {it:5d}  train {losses['train']:.4f}  val {losses['val']:.4f}{marker}")
    if it == max_iters:
        break
    for g in optimizer.param_groups:
        g["lr"] = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / max_iters)))
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
