"""Compare checkpoints on pages that none of them trained on.

Usage:
    python eval_same_split.py checkpoints/quahog_gpt_v4.pt checkpoints/quahog_gpt_v5.pt \
        --old-corpus ../quahog_backups/corpus_v4_trained.txt

A page is unseen if it lies in the last 8% of data/corpus.txt (the 90/10 split
with a 2% margin, since the split is by tokens and this checks characters). With
--old-corpus, it must also lie in the last 8% of that corpus, or be absent from it.
Pass --old-corpus when a checkpoint was trained on an older corpus.

Reports, per checkpoint:
  full-page loss: first 128 tokens of each page, no retrieval
  body-56 loss:   first 56 tokens of the body, given the title (no retrieval)
  retrieval gain: body-56 loss minus loss with the best-matching passage prepended
The first checkpoint is the baseline for paired differences (positive = better).
"""

import argparse
import json
import math
import pathlib

import torch
import torch.nn.functional as F

from gpt import GPT, GPTConfig
from retrieve import load_index
from step3_bpe import encode
from step3_scrape import load_docs

TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
CORPUS_PATH = pathlib.Path("data/corpus.txt")
MARGIN = 0.92
BLOCK = 128
BODY = 56
CONTEXT_TOKENS = 64


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def unseen(docs, corpus, old):
    keep = []
    for title, body in docs:
        s = f"{title}\n{body}"
        if old is not None:
            po = old.find(s)
            if po != -1 and po < MARGIN * len(old):
                continue
        if corpus.find(s) < MARGIN * len(corpus):
            continue
        keep.append((title, body))
    return keep


def nll(model, ids, start, device):
    """Mean loss over target tokens at positions >= start."""
    seq = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(seq[:, :-1])
    lg = logits[:, start - 1 :]
    tg = seq[:, 1:][:, start - 1 :]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1)).item()


def mean_se(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu, sd / math.sqrt(n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--old-corpus", type=pathlib.Path)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    merges = {(a, b): idx for a, b, idx in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
    corpus = CORPUS_PATH.read_text(encoding="utf-8")
    old = args.old_corpus.read_text(encoding="utf-8") if args.old_corpus else None
    docs, _ = load_docs()
    keep = unseen(docs, corpus, old)
    print(f"pages unseen by all checkpoints: {len(keep)} of {len(docs)}")

    index = load_index()
    names = [pathlib.Path(c).stem for c in args.checkpoints]
    models = {name: load_model(c, device) for name, c in zip(names, args.checkpoints)}
    rows = {name: {"full": [], "plain": [], "ground": []} for name in names}
    # Per-page paired arrays, aligned across checkpoints.
    for title, body in keep:
        full = encode(f"{title}\n{body}", merges)[:BLOCK]
        body_ids = encode(body, merges)[:BODY]
        head = encode(f"{title}\n", merges)
        ctx_ids = None
        if len(body_ids) == BODY:
            hits = [h for h in index.search(title, k=10) if h[0] != title]
            budget = min(CONTEXT_TOKENS, BLOCK - BODY - len(head))
            if hits and budget > 0:
                ht, hp = hits[0]
                ctx_ids = encode(f"{ht}\n{hp}\n\n", merges)[:budget]
        for name, model in models.items():
            if len(full) >= 8:
                rows[name]["full"].append(nll(model, full, 1, device))
            if ctx_ids is not None:
                rows[name]["plain"].append(nll(model, head + body_ids, len(head), device))
                rows[name]["ground"].append(
                    nll(model, ctx_ids + head + body_ids, len(ctx_ids) + len(head), device)
                )

    for name in names:
        mu, se = mean_se(rows[name]["full"])
        print(f"{name} full-page loss: {mu:.4f} +/- {se:.4f} (n={len(rows[name]['full'])})")
        mu, se = mean_se(rows[name]["plain"])
        print(f"{name} body-56 loss:   {mu:.4f} +/- {se:.4f} (n={len(rows[name]['plain'])})")
        gain, gse = mean_se(
            [p - g for p, g in zip(rows[name]["plain"], rows[name]["ground"])]
        )
        print(f"{name} retrieval gain: {gain:.4f} +/- {gse:.4f}")
    base = names[0]
    for name in names[1:]:
        d, dse = mean_se(
            [a - b for a, b in zip(rows[base]["full"], rows[name]["full"])]
        )
        print(f"paired full-page {base} - {name}: {d:.4f} +/- {dse:.4f} (positive = {name} better)")


if __name__ == "__main__":
    main()
