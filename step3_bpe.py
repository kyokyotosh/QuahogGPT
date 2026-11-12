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
    # Count each distinct piece once; the merge loop works on unique pieces, weighted by count.
    words = collections.Counter(tuple(p.encode("utf-8")) for p in PIECE.findall(text))
    merges = {}  # (token_a, token_b) -> new_token_id, in the order they were learned
    for _ in range(num_merges):
        pair_counts = collections.Counter()
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        new_id = 256 + len(merges)
        merges[best] = new_id
        words = collections.Counter(
            {merge_pair(w, best, new_id): c for w, c in words.items()}
        )
    return merges


def encode(text, merges):
    ids = []
    for piece in PIECE.findall(text):
        word = list(piece.encode("utf-8"))
        # Apply merges in the order they were learned. Lower id = learned earlier.
        while len(word) >= 2:
            pairs = set(zip(word, word[1:]))
            pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
            if pair not in merges:
                break
            word = list(merge_pair(word, pair, merges[pair]))
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
    text = CORPUS_PATH.read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} characters")

    merges = train(text, NUM_MERGES)
    vocab = build_vocab(merges)
    TOKENIZER_PATH.write_text(
        json.dumps([[a, b, idx] for (a, b), idx in merges.items()]), encoding="utf-8"
    )
    print(f"learned {len(merges)} merges, vocab size {len(vocab)} -> {TOKENIZER_PATH}")

    ids = encode(text, merges)
    print(f"tokens: {len(ids):,}  chars per token: {len(text) / len(ids):.2f}")
    assert decode(ids, vocab) == text, "round trip failed"
    print("round trip ok")

    sample = "Peter Griffin is married to Lois. The Griffins live in Quahog."
    sample_ids = encode(sample, merges)
    print(f"\n'{sample}'")
    print(f"-> {len(sample_ids)} tokens: {[decode([i], vocab) for i in sample_ids]}")
