"""Sample from the v7_long checkpoint with retrieval.

The model sees 128 tokens at a time. Every SEGMENT generated tokens, the wiki
passage most relevant to the recent text is retrieved and placed at the start of
the window (up to CONTEXT_TOKENS), followed by the most recent text. So the
passage stays visible while the model writes about it.
"""

import json
import pathlib

import torch
import torch.nn.functional as F

from quahog.gpt import GPT, GPTConfig
from quahog.retrieve import load_index
from quahog.bpe import build_vocab, decode, encode

CKPT_PATH = pathlib.Path("checkpoints/quahog_gpt_v7_long.pt")
TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
PROMPTS = ["Peter", "Stewie", "The Griffins live"]
NEW_TOKENS = 150
TEMPERATURE = 0.8
TOP_K = 40
SEGMENT = 16  # generated tokens between retrievals
CONTEXT_TOKENS = 64  # window budget for the retrieved passage

device = "mps" if torch.backends.mps.is_available() else "cpu"

merges = {(a, b): idx for a, b, idx in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
vocab = build_vocab(merges)

ckpt = torch.load(CKPT_PATH, map_location=device)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()


def retrieve_ids(index, query_ids):
    """Token ids of the best-ranked passage that fits CONTEXT_TOKENS whole, or [] if none does.

    Passages are never cut: a cut passage ends mid-sentence, which hurts the model.
    See HANDOFF.md (retrieval) for the measurements.
    """
    for hit in index.search(decode(query_ids, vocab), k=10):
        ids = encode(f"{hit[0]}\n{hit[1]}\n\n", merges)
        if len(ids) <= CONTEXT_TOKENS:
            return ids, hit
    return [], None


@torch.no_grad()
def generate_grounded(index, prompt_ids):
    block = cfg.block_size
    out = list(prompt_ids)
    generated = 0
    while generated < NEW_TOKENS:
        ctx, _ = retrieve_ids(index, out[-CONTEXT_TOKENS:])
        for _ in range(min(SEGMENT, NEW_TOKENS - generated)):
            window = ctx + out[-(block - len(ctx)) :]
            logits, _ = model(torch.tensor([window], dtype=torch.long, device=device))
            logits = logits[:, -1, :] / TEMPERATURE
            kth = torch.topk(logits, TOP_K).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
            out.append(next_id)
            generated += 1
    return out


if __name__ == "__main__":
    index = load_index()
    print(f"{index.n:,} passages indexed")
    torch.manual_seed(0)
    for prompt in PROMPTS:
        prompt_ids = encode(prompt, merges)
        _, first = retrieve_ids(index, prompt_ids)
        print(f"\n==================== {prompt!r} ====================")
        if first:
            print(f"[first retrieved: {first[0]}] {first[1][:200]}")
        out = generate_grounded(index, prompt_ids)
        print(decode(out, vocab))
