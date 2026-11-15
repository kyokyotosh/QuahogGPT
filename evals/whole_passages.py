"""Passage budgets with whole passages: 64, 96, window cap (title query, v5).

Usage (from the repo root): python -m evals.whole_passages
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
BLOCK = 128
MAX_HITS = 5


def pack(passages, budget):
    """Whole passages in rank order; stop at the first one that does not fit."""
    out = []
    count = 0
    for p in passages:
        if len(out) + len(p) > budget:
            break
        out.extend(p)
        count += 1
    return out, count


rows = []
for title, body in keep:
    body_ids = encode(body, merges)[:BODY]
    if len(body_ids) != BODY:
        continue
    head = encode(f"{title}\n", merges)
    hits = [h for h in index.search(title, k=20) if h[0] != title][:MAX_HITS]
    passages = [encode(f"{ht}\n{hp}\n\n", merges) for ht, hp in hits]
    if not passages or len(passages[0]) > 64:
        continue
    cap = BLOCK - BODY - len(head)
    plain = ev.nll(model, head + body_ids, len(head), device)
    rec = {}
    for name, budget in [("B64", 64), ("B96", 96), ("Bcap", cap)]:
        c, count = pack(passages, min(budget, cap))
        rec[name] = plain - ev.nll(model, c + head + body_ids, len(c) + len(head), device)
        rec[name + "_count"] = count
    rows.append(rec)


def stats(xs):
    if len(xs) < 2:
        return "n/a"
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{mu:+.4f} +/- {sd / len(xs) ** 0.5:.4f} (n={len(xs)})"


print(f"pages (rank-1 <= 64 tok): {len(rows)}")
for name in ["B64", "B96", "Bcap"]:
    print(f"{name:5s} gain (all):          {stats([r[name] for r in rows])}")
    multi = [r[name] for r in rows if r[name + "_count"] >= 2]
    print(f"{'':5s} gain (>=2 whole passages): {stats(multi)}")


def paired(a, b):
    d = [r[a] - r[b] for r in rows]
    return stats(d)


print(f"paired B96 - B64:   {paired('B96', 'B64')}")
print(f"paired Bcap - B64:  {paired('Bcap', 'B64')}")
print(f"paired Bcap - B96:  {paired('Bcap', 'B96')}")
