"""Rank-1 passage cut at 32/64/96 tokens: where the retrieval gain comes from (v5).

Usage (from the repo root): python -m evals.ablate_retrieval
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

variants = ["R1_32", "R1_64", "R1_96", "R1_tail64_96", "R2_64", "R12_96"]
rows = []
for title, body in keep:
    body_ids = encode(body, merges)[:BODY]
    if len(body_ids) != BODY:
        continue
    head = encode(f"{title}\n", merges)
    hits = [h for h in index.search(title, k=12) if h[0] != title][:2]
    if len(hits) < 2:
        continue
    p1 = encode(f"{hits[0][0]}\n{hits[0][1]}\n\n", merges)
    p2 = encode(f"{hits[1][0]}\n{hits[1][1]}\n\n", merges)
    cap = BLOCK - BODY - len(head)
    ctxs = {
        "R1_32": p1[:32],
        "R1_64": p1[:64],
        "R1_96": p1[:96],
        "R1_tail64_96": p1[64:96],
        "R2_64": p2[:64],
        "R12_96": (p1 + p2)[:96],
    }
    plain = ev.nll(model, head + body_ids, len(head), device)
    rec = {"long1": len(p1) > 64}
    for v in variants:
        c = ctxs[v][:cap]
        if not c:
            rec[v] = None
            continue
        rec[v] = plain - ev.nll(model, c + head + body_ids, len(c) + len(head), device)
    rows.append(rec)


def stats(group, v):
    xs = [r[v] for r in group if r[v] is not None]
    if len(xs) < 2:
        return "n/a"
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    return f"{mu:+.4f} +/- {sd / len(xs) ** 0.5:.4f} (n={len(xs)})"


print(f"pages: {len(rows)}; rank-1 passage > 64 tokens: {sum(r['long1'] for r in rows)}")
for v in variants:
    print(f"{v:14s} all:   {stats(rows, v)}")
    print(f"{'':14s} long1: {stats([r for r in rows if r['long1']], v)}")
    print(f"{'':14s} short1:{stats([r for r in rows if not r['long1']], v)}")
