"""Step 3b: train a byte-level BPE tokenizer from scratch.

Step 1 used one token per character. Real LLMs use subword tokens: frequent
byte sequences become single tokens, so "Griffin" might be one token instead
of seven. BPE builds that vocabulary greedily:

  1. Start with the 256 possible bytes as tokens.
  2. Find the most frequent adjacent pair of tokens.
  3. Merge that pair into a new token. Repeat.

Text is split into words first (the regex below), so merges never cross
word boundaries. This mirrors GPT-2's tokenizer.
"""

import argparse
import collections
import json
import pathlib
import re

CORPUS_PATH = pathlib.Path("data/corpus.txt")
TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
NUM_MERGES = 768  # final vocab = 256 bytes + 768 merges = 1024 tokens

# A piece is an optional leading space plus a run of letters, digits, or punctuation,
# or a run of whitespace.
PIECE = re.compile(r"\s?\w+|\s?[^\w\s]+|\s+")


def merge_pair(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train(text, num_merges):
    """Learn `num_merges` merges. Pair counts are updated incrementally: after a merge,
    only the words containing the merged pair are recounted, which keeps large
    vocabularies (thousands of merges) fast."""
    counts = collections.Counter(tuple(p.encode("utf-8")) for p in PIECE.findall(text))
    words = [list(w) for w in counts]  # unique pieces as token lists
    freq = list(counts.values())  # how often each unique piece occurs
    pair_counts = collections.Counter()
    where = collections.defaultdict(set)  # pair -> indices of words that contain it
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freq[i]
            where[pair].add(i)
    merges = {}  # (token_a, token_b) -> new_token_id, in the order they were learned
    for _ in range(num_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        new_id = 256 + len(merges)
        merges[best] = new_id
        for i in where.pop(best):
            w = words[i]
            for pair in zip(w, w[1:]):  # remove this word's old pairs
                pair_counts[pair] -= freq[i]
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            w = list(merge_pair(w, best, new_id))
            words[i] = w
            for pair in zip(w, w[1:]):  # add its new pairs
                pair_counts[pair] += freq[i]
                where[pair].add(i)
    return merges


def encode(text, merges, cache=None):
    """Token ids for `text`. Pass a dict as `cache` to reuse results for repeated pieces."""
    cache = {} if cache is None else cache
    ids = []
    for piece in PIECE.findall(text):
        if piece in cache:
            ids.extend(cache[piece])
            continue
        word = list(piece.encode("utf-8"))
        # Apply merges in the order they were learned. Lower id = learned earlier.
        while len(word) >= 2:
            pairs = set(zip(word, word[1:]))
            pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            word = list(merge_pair(word, pair, merges[pair]))
        cache[piece] = word
        ids.extend(word)
    return ids


def build_vocab(merges):
    """Token id -> the bytes it stands for."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():  # dict order = learned order, so a and b already exist
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--merges", type=int, default=NUM_MERGES, help="vocab = 256 + merges")
    parser.add_argument("--out", default=str(TOKENIZER_PATH))
    args = parser.parse_args()
    out_path = pathlib.Path(args.out)
    text = CORPUS_PATH.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters")

    merges = train(text, args.merges)
    vocab = build_vocab(merges)
    out_path.write_text(json.dumps([[a, b, idx] for (a, b), idx in merges.items()]), encoding="utf-8")
    print(f"learned {len(merges)} merges, vocab size {len(vocab)} -> {out_path}")

    ids = encode(text, merges)
    print(f"tokens: {len(ids):,}  chars per token: {len(text) / len(ids):.2f}")
    assert decode(ids, vocab) == text, "round trip failed"
    print("round trip ok")

    sample = "Peter Griffin is married to Lois. The Griffins live in Quahog."
    sample_ids = encode(sample, merges)
    print(f"\n'{sample}'")
    print(f"-> {len(sample_ids)} tokens: {[decode([i], vocab) for i in sample_ids]}")
