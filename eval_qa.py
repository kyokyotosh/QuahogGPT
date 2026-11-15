"""Score a question-answering reader on the held-out questions in data/qa/test.jsonl.

A prediction is correct when it contains any accepted answer, after lowercasing and
dropping punctuation and articles. Reports accuracy overall and per question kind,
with a standard error, and writes every prediction to a JSONL file for inspection.

Usage:
    .venv-mlx/bin/python eval_qa.py --out logs_qa_base.jsonl
    .venv-mlx/bin/python eval_qa.py --no-retrieval --out logs_qa_norag.jsonl
    .venv-mlx/bin/python eval_qa.py --adapter adapters/qa --out logs_qa_lora.jsonl
    .venv/bin/python eval_qa.py --reader quahog --ckpt checkpoints/quahog_qa_v8c.pt --out logs_qa_quahog_v8c.jsonl
"""

import argparse
import collections
import json
import math
import pathlib
import re
import time

TEST = pathlib.Path("data/qa/test.jsonl")


def norm(text):
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return " ".join(text.split())


def correct(prediction, answers):
    p = f" {norm(prediction)} "
    return any(f" {norm(a)} " in p for a in answers if norm(a))


def acc(flags):
    n = len(flags)
    p = sum(flags) / n
    return p, math.sqrt(p * (1 - p) / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader", choices=["qwen", "quahog"], default="qwen")
    parser.add_argument("--adapter", default=None, help="qwen: LoRA adapter directory")
    parser.add_argument("--model", default=None, help="qwen: model repo (default: ask.MODEL)")
    parser.add_argument("--ckpt", default="checkpoints/quahog_qa_v8c.pt", help="quahog: checkpoint")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines()]
    items = items[: args.limit] if args.limit else items
    if args.reader == "qwen":  # imported here so each reader only needs its own venv
        from ask import MODEL, QA

        qa = QA(adapter=args.adapter, retrieval=not args.no_retrieval, model=args.model or MODEL)
    else:
        from quahog_qa import QuahogQA

        qa = QuahogQA(args.ckpt)
    kinds = collections.defaultdict(list)
    start = time.time()
    with open(args.out, "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            pred = qa.answer(it["question"])
            ok = correct(pred, it["answers"])
            kinds[it["kind"]].append(ok)
            fh.write(json.dumps({**it, "prediction": pred, "correct": ok}, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(items)}  {time.time() - start:.0f}s", flush=True)
    every = [ok for flags in kinds.values() for ok in flags]
    p, se = acc(every)
    print(f"\noverall: {p:.3f} +/- {se:.3f}  (n={len(every)})")
    for kind in sorted(kinds):
        p, se = acc(kinds[kind])
        print(f"  {kind:<14}{p:.3f} +/- {se:.3f}  (n={len(kinds[kind])})")


if __name__ == "__main__":
    main()
