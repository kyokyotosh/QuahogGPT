"""How often the target page shares an 8-word run with the retrieved text.

Usage (from the repo root): python -m evals.overlap
"""

import json
import pathlib

import torch

from evals import same_split as ev
from quahog.bpe import encode
from quahog.retrieve import load_index
from quahog.scrape import load_docs

merges = {(a, b): i for a, b, i in json.loads(pathlib.Path("data/tokenizer.json").read_text(encoding="utf-8"))}
corpus = pathlib.Path("data/corpus.txt").read_text(encoding="utf-8")
docs, _ = load_docs()
keep = ev.unseen(docs, corpus, None)
index = load_index()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = ev.load_model("checkpoints/quahog_gpt_v5.pt", device)
BODY = 24
N = 8


def grams(words):
    return {" ".join(words[i : i + N]) for i in range(len(words) - N + 1)}


rows = []
for title, body in keep:
    body_ids = encode(body, merges)[:BODY]
    if len(body_ids) != BODY:
        continue
    head = encode(f"{title}\n", merges)
    hits = [h for h in index.search(title, k=11) if h[0] != title][:1]
    if not hits:
        continue
    ht, hp = hits[0]
    ctx_ids = encode(f"{ht}\n{hp}\n\n", merges)
    body_words = body.split()[:40]
    dup = bool(grams(body_words) & grams(hp.split()))
    plain = ev.nll(model, head + body_ids, len(head), device)
    g64 = ev.nll(model, ctx_ids[:64] + head + body_ids, len(ctx_ids[:64]) + len(head), device) if len(ctx_ids[:64]) + len(head) + BODY <= 128 else None
    g96 = ev.nll(model, ctx_ids[:96] + head + body_ids, len(ctx_ids[:96]) + len(head), device) if len(ctx_ids[:96]) + len(head) + BODY <= 128 else None
    rows.append((dup, len(ctx_ids), plain, g64, g96))

n = len(rows)
dups = [r for r in rows if r[0]]
print(f"pages: {n}; ctx with 8-word overlap with body: {len(dups)} ({len(dups)/n:.1%})")
print(f"retrieved passage > 64 tokens: {sum(r[1] > 64 for r in rows)}")


def gain_stats(group, idx):
    xs = [r[2] - r[idx] for r in group if r[idx] is not None]
    if not xs:
        return "n/a"
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{mu:+.4f} +/- {sd / len(xs) ** 0.5:.4f} (n={len(xs)})"


nodup = [r for r in rows if not r[0]]
print(f"gain64 all: {gain_stats(rows, 3)}")
print(f"gain64 no-overlap: {gain_stats(nodup, 3)}")
print(f"gain64 overlap: {gain_stats(dups, 3)}")
print(f"gain96 all: {gain_stats(rows, 4)}")
print(f"gain96 no-overlap: {gain_stats(nodup, 4)}")
print(f"gain96 overlap: {gain_stats(dups, 4)}")
