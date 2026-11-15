"""Does the assistant stay on Family Guy? Scores qa/ask.py on data/qa/guard_test.jsonl.

  off_*    off-topic messages (general knowledge, tasks, other shows, celebrities outside
           the show, chit-chat, tricks): should be refused (ask.REFUSAL)
  on_open  open-ended Family Guy requests (summaries, explanations, short scenes):
           should be answered, not refused

Factual Family Guy accuracy (and false refusals on it) comes from evals/qa.py.

Usage (defaults: ask.MODEL with the topic gate):
    .venv-mlx/bin/python -m evals.guard --out logs_guard_final.jsonl
"""

import argparse
import collections
import json
import math
import pathlib

from qa.ask import MODEL, QA, is_refusal

GUARD = pathlib.Path("data/qa/guard_test.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = [json.loads(line) for line in GUARD.read_text(encoding="utf-8").splitlines()]
    qa = QA(adapter=args.adapter, model=args.model)
    refused = collections.defaultdict(list)
    with open(args.out, "w", encoding="utf-8") as fh:
        for it in items:
            pred = qa.answer(it["question"])
            r = is_refusal(pred)
            refused[it["kind"]].append(r)
            fh.write(json.dumps({**it, "prediction": pred, "refused": r}, ensure_ascii=False) + "\n")

    def rate(flags):
        p = sum(flags) / len(flags)
        return f"{p:.3f} +/- {math.sqrt(p * (1 - p) / len(flags)):.3f} (n={len(flags)})"

    off = [r for k, v in refused.items() if k.startswith("off_") for r in v]
    print(f"off-topic refused (want 1): {rate(off)}")
    for kind in sorted(k for k in refused if k.startswith("off_")):
        print(f"  {kind:<16}{rate(refused[kind])}")
    print(f"on-topic open refused (want 0): {rate(refused['on_open'])}")


if __name__ == "__main__":
    main()
