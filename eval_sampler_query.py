"""Paired check of retrieval variants with the sampler's own query.

rag_sample.py retrieves with the last 64 tokens of generated text, decoded to
a string, and predicts the next SEGMENT (16) tokens. This script mirrors that on
pages the checkpoint never trained on:

  query:   first PREFIX (64) tokens of "title\\nbody", decoded to text
  target:  the next TARGET (16) tokens
  window:  [context] + prefix + target; the context is capped at CTX tokens so
           the input (window minus one token) fits BLOCK (128)

Variants, all with the same prefix and target:
  plain:      no passage
  truncated:  top-1 passage cut to CTX tokens (the behavior before the fix)
  first-fit:  first passage in rank order that fits CTX whole (the current behavior)

Pages whose query has no hit other than the page itself are skipped. The page's
own title is excluded from hits, as in eval_same_split.py.

Usage (from the repo root):
    python eval_sampler_query.py checkpoints/quahog_gpt_v5.pt
"""

import argparse
import json
import pathlib

import torch

import eval_same_split as ev
from retrieve import load_index
from step3_bpe import build_vocab, decode, encode
from step3_scrape import load_docs

TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
CORPUS_PATH = pathlib.Path("data/corpus.txt")
PREFIX = 64
TARGET = 16
BLOCK = 128
CTX = BLOCK - PREFIX - TARGET + 1  # input = ctx + PREFIX + TARGET - 1 <= BLOCK


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    merges = {(a, b): i for a, b, i in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
    vocab = build_vocab(merges)
    corpus = CORPUS_PATH.read_text(encoding="utf-8")
    docs, _ = load_docs()
    keep = ev.unseen(docs, corpus, None)
    index = load_index()
    model = ev.load_model(args.checkpoint, device)

    rows = []  # (plain, truncated gain, first-fit gain or None)
    for title, body in keep:
        full = encode(f"{title}\n{body}", merges)
        if len(full) < PREFIX + TARGET:
            continue
        prefix = full[:PREFIX]
        target = full[PREFIX : PREFIX + TARGET]
        query = decode(prefix, vocab)
        hits = [h for h in index.search(query, k=12) if h[0] != title]
        if not hits:
            continue

        plain = ev.nll(model, prefix + target, PREFIX, device)
        top = encode(f"{hits[0][0]}\n{hits[0][1]}\n\n", merges)[:CTX]
        trunc = plain - ev.nll(model, top + prefix + target, len(top) + PREFIX, device)

        fit = None
        for ht, hp in hits:
            ids = encode(f"{ht}\n{hp}\n\n", merges)
            if len(ids) <= CTX:
                fit = ids
                break
        first = None
        if fit is not None:
            first = plain - ev.nll(model, fit + prefix + target, len(fit) + PREFIX, device)
        rows.append((plain, trunc, first))

    def paired_stats(xs):
        mu, se = ev.mean_se(xs)
        return f"{mu:+.4f} +/- {se:.4f} (n={len(xs)})"

    both = [(t, f) for _, t, f in rows if f is not None]
    print(f"pages: {len(rows)} scored; plain loss mean {ev.mean_se([p for p, _, _ in rows])[0]:.4f}")
    print(f"truncated top-1 gain (all scored):   {paired_stats([t for _, t, _ in rows])}")
    print(f"first-fit gain (pages with a fit):   {paired_stats([f for _, f in both])}")
    print(f"truncated top-1 gain (same pages):   {paired_stats([t for t, _ in both])}")
    print(f"paired first-fit - truncated:        {paired_stats([f - t for t, f in both])}")


if __name__ == "__main__":
    main()
