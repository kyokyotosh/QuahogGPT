"""Answer Family Guy questions with QuahogGPT, trained from scratch (no pretrained model).

The prompt is plain text, shared by training (qa_train.py) and inference:

    <title>: <excerpt>
    <title>: <excerpt>
    <title>: <excerpt>
    Question: <question>
    Answer: <reply>\\n

Excerpts come from qa_retrieve.py (top K, at most MAX_DOCS facts documents). Long
excerpts are trimmed so that the prompt plus the reply fits the model's window.
Generation is greedy, never repeats a 4-token sequence, and stops at the first newline.

Usage (main venv, torch):
    .venv/bin/python quahog_qa.py "Who is Peter Griffin?" "When did Joe Swanson first appear?"
"""

import argparse
import json
import pathlib

import torch

from gpt import GPT, GPTConfig
from qa_retrieve import load_retriever
from step3_bpe import build_vocab, decode, encode

CKPT = "checkpoints/quahog_qa_v8.pt"
K = 3  # excerpts per question; the 512-token window holds about three
MAX_DOCS = 1  # facts documents among them (tuned on train questions; more crowd out passages)
MAX_ANSWER = 64  # tokens reserved for the reply
NO_REPEAT_NGRAM = 4  # greedy decoding loops ("Rhode Island, Rhode Island, ..."); block repeats
NEWLINE = "\n"


class Tokenizer:
    def __init__(self, path):
        self.merges = {(a, b): i for a, b, i in json.loads(pathlib.Path(path).read_text(encoding="utf-8"))}
        self.vocab = build_vocab(self.merges)
        self.cache = {}

    def encode(self, text):
        return encode(text, self.merges, self.cache)

    def decode(self, ids):
        return decode(ids, self.vocab)


def fit(lengths, budget):
    """Per-excerpt token caps that sum to at most `budget`: short excerpts stay whole,
    and what they leave over is shared evenly by the longer ones."""
    caps = [0] * len(lengths)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    left = budget
    for n, i in enumerate(order):
        share = left // (len(order) - n)
        caps[i] = min(lengths[i], share)
        left -= caps[i]
    return caps


def repeated_ngram_tokens(out, n):
    """Tokens that would repeat an n-gram already in `out` (given its last n-1 tokens)."""
    if len(out) < n - 1:
        return []
    prefix = out[len(out) - n + 1 :]
    return [out[i + n - 1] for i in range(len(out) - n + 1) if out[i : i + n - 1] == prefix]


def prompt_ids(tok, question, passages, block_size, reply_tokens=MAX_ANSWER):
    """Token ids of the prompt (excerpts, question, "Answer:"), sized to leave
    `reply_tokens` free in the window."""
    tail = tok.encode(f"Question: {question}\nAnswer:")
    excerpts = [tok.encode(f"{title}: {text}\n") for title, text in passages]
    budget = max(0, block_size - len(tail) - reply_tokens)
    caps = fit([len(e) for e in excerpts], budget)
    ids = []
    for e, cap in zip(excerpts, caps):
        ids += e if cap == len(e) else e[: max(0, cap - 1)] + tok.encode("\n")
    return ids + tail


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Tokenizer(ckpt["tokenizer"])


class QuahogQA:
    def __init__(self, ckpt=CKPT):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model, self.tok = load_model(ckpt, self.device)
        self.retriever = load_retriever()

    def passages(self, question):
        return self.retriever.retrieve(question, k=K, max_docs=MAX_DOCS)

    @torch.no_grad()
    def answer(self, question):
        ids = prompt_ids(self.tok, question, self.passages(question), self.model.cfg.block_size)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = []
        for _ in range(MAX_ANSWER):
            logits, _ = self.model(x[:, -self.model.cfg.block_size :])
            logits = logits[0, -1]
            logits[repeated_ngram_tokens(out, NO_REPEAT_NGRAM)] = float("-inf")
            nxt = int(logits.argmax())
            out.append(nxt)
            if NEWLINE in self.tok.decode([nxt]):  # merged tokens can carry the newline
                break
            x = torch.cat([x, torch.tensor([[nxt]], device=self.device)], dim=1)
        return self.tok.decode(out).split(NEWLINE)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--show-passages", action="store_true")
    args = parser.parse_args()
    qa = QuahogQA(args.ckpt)
    for q in args.questions:
        print(f"Q: {q}")
        if args.show_passages:
            for title, text in qa.passages(q):
                print(f"   [{title}] {text[:150]}")
        print(f"A: {qa.answer(q)}\n")


if __name__ == "__main__":
    main()
