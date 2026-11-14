"""Fine-tune a pretrained QuahogGPT checkpoint to answer questions from wiki excerpts.

No pretrained model from outside the project is involved: the starting checkpoint
comes from step4_train.py on the Family Guy corpus.

Training examples (train pages only; see build_qa.py for the page split):
  - structured infobox questions, each rewritten with every phrasing in
    build_qa.TEMPLATES, with template replies (build_qa.structured_reply)
  - generated questions (generated_train.jsonl, generated_train_2.jsonl): "who is"
    questions keep their 1-2 sentence reply; passage questions use the short gold
    answer, which an 8.5M-parameter model learns to copy far more reliably
Each example is quahog_qa.prompt_ids(...) followed by " <reply>\\n". The loss covers the
reply tokens only. Examples whose trimmed excerpts do not contain a gold answer are
dropped, so the model is never taught to answer from nothing.

A slice of train pages (adler32(page) % 20 == 0) is held out for validation loss;
the checkpoint with the lowest validation loss is saved.

Usage:
    .venv/bin/python qa_train.py --init checkpoints/quahog_gpt_v8.pt --out checkpoints/quahog_qa_v8.pt
"""

import argparse
import json
import math
import pathlib
import random
import zlib

import torch
import torch.nn.functional as F

from build_qa import TEMPLATES, structured_reply
from eval_qa import norm
from quahog_qa import K, MAX_ANSWER, MAX_DOCS, load_model, prompt_ids
from qa_retrieve import load_retriever

QA_DIR = pathlib.Path("data/qa")


def is_valid_page(page):
    return zlib.adler32(page.encode("utf-8")) % 20 == 0


def qa_items(facts):
    """(question, reply, answers, page) for every train-page example."""
    items = []
    for line in (QA_DIR / "structured.jsonl").open(encoding="utf-8"):
        x = json.loads(line)
        if x["split"] != "train":
            continue
        reply = structured_reply(x, facts)
        for template in TEMPLATES[x["kind"]]:
            items.append((template.format(t=x["page"]), reply, x["answers"], x["page"]))
    for name in ("generated_train.jsonl", "generated_train_2.jsonl"):
        for line in (QA_DIR / name).open(encoding="utf-8"):
            x = json.loads(line)
            assert x["split"] == "train"
            reply = x["reply"] if x["kind"] == "who_is" else x["answers"][0]
            items.append((x["question"], reply, x["answers"], x["page"]))
    return items


def build_examples(tok, retriever, block_size):
    examples = {"train": [], "valid": []}
    dropped = 0
    for question, reply, answers, page in qa_items(retriever.facts):
        prompt = prompt_ids(tok, question, retriever.retrieve(question, k=K, max_docs=MAX_DOCS), block_size)
        context = f" {norm(tok.decode(prompt))} "
        if not any(f" {norm(a)} " in context for a in answers if norm(a)):
            dropped += 1
            continue
        target = tok.encode(f" {reply}\n")[:MAX_ANSWER]
        examples["valid" if is_valid_page(page) else "train"].append((prompt, target))
    return examples, dropped


def batch(examples, device):
    """Right-padded inputs and targets; non-reply positions get target -1 (ignored)."""
    seqs = [p + t for p, t in examples]
    width = max(len(s) for s in seqs) - 1
    x = torch.zeros(len(seqs), width, dtype=torch.long)
    y = torch.full((len(seqs), width), -1, dtype=torch.long)
    for i, ((p, t), s) in enumerate(zip(examples, seqs)):
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
        y[i, len(p) - 1 : len(s) - 1] = torch.tensor(t)  # predict each reply token
    return x.to(device), y.to(device)


def loss_of(model, x, y):
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)


@torch.no_grad()
def valid_loss(model, examples, device, bs):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), bs):
        chunk = examples[i : i + bs]
        total += loss_of(model, *batch(chunk, device)).item() * len(chunk)
        n += len(chunk)
    model.train()
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", default="checkpoints/quahog_gpt_v8.pt")
    parser.add_argument("--out", default="checkpoints/quahog_qa_v8.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, tok = load_model(args.init, device)
    ckpt = torch.load(args.init, map_location="cpu")
    retriever = load_retriever()
    examples, dropped = build_examples(tok, retriever, model.cfg.block_size)
    train, valid = examples["train"], examples["valid"]
    print(f"train {len(train)}, valid {len(valid)}, dropped {dropped} (answer not in trimmed excerpts)")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    warmup = 100
    best = valid_loss(model, valid, device, args.batch_size)
    print(f"step 0  valid {best:.4f}")
    step = 0
    for epoch in range(args.epochs):
        rng.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            lr = args.lr * min(1, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            loss = loss_of(model, *batch(train[i : i + args.batch_size], device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        v = valid_loss(model, valid, device, args.batch_size)
        marker = ""
        if v < best:
            best = v
            torch.save({**ckpt, "model": model.state_dict()}, args.out)
            marker = "  (saved)"
        print(f"epoch {epoch + 1}  step {step}  train {loss.item():.4f}  valid {v:.4f}{marker}", flush=True)
    print(f"best valid loss {best:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
