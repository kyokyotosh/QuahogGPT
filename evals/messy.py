"""Casually typed input (data/qa/messy_test.jsonl, qa/build_messy_test.py).

  fact_*    factual questions: correct = the answer contains a gold answer
  on_open   open Family Guy requests: correct = not refused
  casual    hand-written backstory questions: correct = not refused
  off_*     off-topic messages: correct = refused

--gate-only scores just the topic gate (fast): allowed for on-topic, refused for off-topic.

Usage:
    .venv-mlx/bin/python -m evals.messy --gate-only
    .venv-mlx/bin/python -m evals.messy --out logs/eval/messy.jsonl
"""

import argparse
import collections
import json
import math
import pathlib

MESSY = pathlib.Path("data/qa/messy_test.jsonl")


def group(kind):
    return "off-topic refused" if kind.startswith("off_") else "factual correct" if kind.startswith("fact_") else f"{kind} answered"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--adapter", default=None, help="gate adapter (default: ask.GATE_ADAPTER)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    items = [json.loads(l) for l in MESSY.read_text(encoding="utf-8").splitlines()]
    res = collections.defaultdict(list)
    out = open(args.out, "w", encoding="utf-8") if args.out else None

    if args.gate_only:
        from qa.ask import GATE_ADAPTER, Gate
        from quahog.qa_retrieve import load_retriever

        gate, retriever = Gate(adapter=args.adapter or GATE_ADAPTER), load_retriever()
        for it in items:
            on = gate.on_topic(it["question"], retriever.link(it["question"]))
            ok = on != it["kind"].startswith("off_")
            res["gate: " + ("off-topic refused" if it["kind"].startswith("off_") else f"{it['kind'].split('_')[0]} allowed")].append(ok)
            if out:
                out.write(json.dumps({**it, "on_topic": on, "correct": ok}, ensure_ascii=False) + "\n")
    else:
        from qa.ask import QA, is_refusal
        from quahog.metrics import correct

        qa = QA()
        for it in items:
            answer = qa.answer(it["question"])
            if it["kind"].startswith("off_"):
                ok = is_refusal(answer)
            elif it["kind"].startswith("fact_"):
                ok = correct(answer, it["answers"])
            else:
                ok = not is_refusal(answer)
            res[group(it["kind"])].append(ok)
            if out:
                out.write(json.dumps({**it, "answer": answer, "correct": ok}, ensure_ascii=False) + "\n")

    for k, v in sorted(res.items()):
        p = sum(v) / len(v)
        print(f"  {k:<26} {sum(v)}/{len(v)}  ({p:.3f} +/- {math.sqrt(p * (1 - p) / len(v)):.3f})")


if __name__ == "__main__":
    main()
