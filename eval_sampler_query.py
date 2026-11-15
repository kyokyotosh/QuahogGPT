"""Sampler-faithful check of retrieval variants, for one or more checkpoints.

For pages that no checkpoint trained on (see eval_same_split.py for the split):

  query:   the first CONTEXT (64) tokens of "title\\nbody", decoded to text. This is
           what generate_grounded() retrieves with at the start of a segment.
  target:  the next SEGMENT (16) tokens.
  window:  for each target token, the retrieved context followed by the most
           recent tokens that fit in BLOCK (128), exactly as generate_grounded()
           builds it. Each target token is scored separately.

Variants (all share the same query and target):
  none:       no retrieval
  truncated:  top-1 passage cut to CONTEXT tokens (the behavior before the fix)
  first-fit:  first passage in rank order that fits CONTEXT whole; no retrieval if
              none fits (the current pretrain/rag_sample.py)

Constants match pretrain/rag_sample.py. Keep them in sync if that file changes.

Usage (from the repo root):
    python eval_sampler_query.py checkpoints/quahog_gpt_v5.pt checkpoints/quahog_gpt_v7_wide.pt
The first checkpoint is the baseline for paired differences between checkpoints.
"""

import argparse
import json
import pathlib

import torch
import torch.nn.functional as F

import eval_same_split as ev
from quahog.retrieve import load_index
from quahog.bpe import build_vocab, decode, encode
from quahog.scrape import load_docs

TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
CORPUS_PATH = pathlib.Path("data/corpus.txt")
BLOCK = 128
CONTEXT = 64  # rag_sample.CONTEXT_TOKENS
SEGMENT = 16  # rag_sample.SEGMENT
VARIANTS = ["none", "truncated", "first-fit"]


def target_losses(model, ctx, prefix, target, device):
    """Mean loss over the target tokens, each scored with the sampler's window."""
    total = 0.0
    for j, tok in enumerate(target):
        before = prefix + target[:j]
        window = ctx + before[-(BLOCK - len(ctx)) :]
        with torch.no_grad():
            logits, _ = model(torch.tensor([window], dtype=torch.long, device=device))
        total += -F.log_softmax(logits[0, -1], dim=-1)[tok].item()
    return total / len(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    merges = {(a, b): i for a, b, i in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
    vocab = build_vocab(merges)
    corpus = CORPUS_PATH.read_text(encoding="utf-8")
    docs, _ = load_docs()
    keep = ev.unseen(docs, corpus, None)
    index = load_index()
    names = [pathlib.Path(c).stem for c in args.checkpoints]
    models = {n: ev.load_model(c, device) for n, c in zip(names, args.checkpoints)}

    scores = {n: {v: [] for v in VARIANTS} for n in names}
    for title, body in keep:
        full = encode(f"{title}\n{body}", merges)
        if len(full) < CONTEXT + SEGMENT:
            continue
        prefix = full[:CONTEXT]
        target = full[CONTEXT : CONTEXT + SEGMENT]
        hits = [h for h in index.search(decode(prefix, vocab), k=12) if h[0] != title]
        cands = [encode(f"{ht}\n{hp}\n\n", merges) for ht, hp in hits]
        ctx = {
            "none": [],
            "truncated": cands[0][:CONTEXT] if cands else [],
            "first-fit": next((c for c in cands if len(c) <= CONTEXT), []),
        }
        for n, model in models.items():
            for v in VARIANTS:
                scores[n][v].append(target_losses(model, ctx[v], prefix, target, device))

    print(f"pages scored: {len(scores[names[0]]['none'])}")
    print(f"{'checkpoint':<22}{'variant':<12}{'loss':>22}{'gain vs none':>24}")
    for n in names:
        base = scores[n]["none"]
        for v in VARIANTS:
            mu, se = ev.mean_se(scores[n][v])
            if v == "none":
                gain = "-"
            else:
                g, gse = ev.mean_se([a - b for a, b in zip(base, scores[n][v])])
                gain = f"{g:+.4f} +/- {gse:.4f}"
            print(f"{n:<22}{v:<12}{mu:>10.4f} +/- {se:.4f}{gain:>24}")
    if len(names) > 1:
        ref = names[0]
        print(f"\npaired first-fit, checkpoint minus {ref} (negative = better):")
        for n in names[1:]:
            d, dse = ev.mean_se(
                [a - b for a, b in zip(scores[n]["first-fit"], scores[ref]["first-fit"])]
            )
            print(f"  {n}: {d:+.4f} +/- {dse:.4f}")


if __name__ == "__main__":
    main()
