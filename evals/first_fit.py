"""Truncated top-1 passage vs the first whole passage that fits (title query, v5).

Usage (from the repo root): python -m evals.first_fit
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
CONTEXT = 64

rows = []  # (gain_trunc, gain_fit or None)
for title, body in keep:
    body_ids = encode(body, merges)[:BODY]
    if len(body_ids) != BODY:
        continue
    head = encode(f"{title}\n", merges)
    cap = min(CONTEXT, BLOCK - BODY - len(head))
    hits = [h for h in index.search(title, k=12) if h[0] != title]
    if not hits:
        continue
    plain = ev.nll(model, head + body_ids, len(head), device)

    top = encode(f"{hits[0][0]}\n{hits[0][1]}\n\n", merges)[:cap]
    g_trunc = plain - ev.nll(model, top + head + body_ids, len(top) + len(head), device)

    fit = None
    for ht, hp in hits:
        p = encode(f"{ht}\n{hp}\n\n", merges)
        if len(p) <= cap:
            fit = p
            break
    g_fit = None
    if fit is not None:
        g_fit = plain - ev.nll(model, fit + head + body_ids, len(fit) + len(head), device)
    rows.append((g_trunc, g_fit, len(hits[0][1])))


def stats(xs):
    if len(xs) < 2:
        return "n/a"
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{mu:+.4f} +/- {sd / len(xs) ** 0.5:.4f} (n={len(xs)})"


both = [(a, b) for a, b, _ in rows if b is not None]
print(f"pages with a hit: {len(rows)}; with a whole passage that fits: {len(both)}")
print(f"truncated top-1 gain (all pages):     {stats([a for a, _, _ in rows])}")
print(f"first-fitting whole passage (fits):   {stats([b for _, b in both])}")
print(f"truncated top-1 gain (same pages):    {stats([a for a, _ in both])}")
print(f"paired fit - truncated (same pages):  {stats([b - a for a, b in both])}")
