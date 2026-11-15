"""Multi-turn check: does the assistant answer follow-ups in context, and keep the topic?

For each conversation in data/qa/chat_test.jsonl (qa/build_chat_test.py), the opener is
answered first and added to the history; the follow-up is scored:
  episode_*, debut, song   correct = the answer contains a gold answer
  off_topic                correct = refused
  open                     correct = not refused
It also reports how often the rewritten follow-up names the conversation's page.

Usage:
    .venv-mlx/bin/python -m evals.chat --out logs/eval/chat.jsonl
    .venv-mlx/bin/python -m evals.chat --no-rewrite --out logs/eval/chat_norewrite.jsonl
"""

import argparse
import collections
import json
import math
import pathlib

from qa.ask import QA, is_refusal, strip_citations
from quahog.metrics import correct, norm

CHAT = pathlib.Path("data/qa/chat_test.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rewrite", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    convos = [json.loads(l) for l in CHAT.read_text(encoding="utf-8").splitlines()]
    qa = QA(rewrite=not args.no_rewrite)
    ok, named = collections.defaultdict(list), []
    with open(args.out, "w", encoding="utf-8") as fh:
        for c in convos:
            first, follow = c["turns"]
            r1 = qa.reply([], first)
            history = [{"role": "user", "content": first}, {"role": "assistant", "content": r1["answer"]}]
            r2 = qa.reply(history, follow)
            answer = strip_citations(r2["answer"])
            if c["kind"] == "off_topic":
                good = is_refusal(answer)
            elif c["kind"] == "open":
                good = not is_refusal(answer)
            else:
                good = correct(answer, c["answers"])
            ok[c["kind"]].append(good)
            if c["kind"] != "off_topic":
                named.append(f" {norm(c['page'])} " in f" {norm(r2['standalone'])} ")
            fh.write(json.dumps({**c, "first_answer": r1["answer"], "standalone": r2["standalone"],
                                 "answer": r2["answer"], "correct": good}, ensure_ascii=False) + "\n")

    def rate(flags):
        p = sum(flags) / len(flags)
        return f"{p:.3f} +/- {math.sqrt(p * (1 - p) / len(flags)):.3f} (n={len(flags)})"

    every = [v for flags in ok.values() for v in flags]
    print(f"follow-ups handled: {rate(every)}")
    for kind in sorted(ok):
        print(f"  {kind:<18}{rate(ok[kind])}")
    print(f"rewrite names the page: {rate(named)}")


if __name__ == "__main__":
    main()
