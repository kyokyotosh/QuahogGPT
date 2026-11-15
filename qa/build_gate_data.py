"""Training data for the topic classifier (ask.Gate): each example is ask.gate_messages(...)
with the assistant reply "ON" or "OFF".

Train pages and train-only messages only:
  OFF  data/qa/guard_train.jsonl off_* messages (generated; labels confirmed by a judge)
  ON   data/qa/guard_train.jsonl on_open requests, plus factual train questions from
       structured.jsonl (every kind), generated_train.jsonl and generated_train_2.jsonl

Usage:
    .venv-mlx/bin/python -m qa.build_gate_data   # writes data/qa/gate/{train,valid}.jsonl
    .venv-mlx/bin/python -m mlx_lm lora -c configs/lora_gate.yaml
"""

import json
import pathlib
import random

from qa.ask import gate_messages
from quahog.qa_retrieve import load_retriever

QA_DIR = pathlib.Path("data/qa")
OUT_DIR = QA_DIR / "gate"
FACTUAL = {"structured.jsonl": 900, "generated_train.jsonl": 600, "generated_train_2.jsonl": 600}
VALID = 200


def read(name):
    return [json.loads(line) for line in (QA_DIR / name).open(encoding="utf-8")]


def main():
    rng = random.Random(0)
    retriever = load_retriever()
    guard = read("guard_train.jsonl")
    labeled = [(x["question"], "OFF" if x["kind"].startswith("off_") else "ON") for x in guard]
    for name, n in FACTUAL.items():
        rows = [x for x in read(name) if x["split"] == "train"]
        labeled += [(x["question"], "ON") for x in rng.sample(rows, min(n, len(rows)))]
    rng.shuffle(labeled)
    examples = [
        {"messages": gate_messages(q, retriever.link(q)) + [{"role": "assistant", "content": label}]}
        for q, label in labeled
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid", examples[:VALID]), ("train", examples[VALID:])):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    off = sum(label == "OFF" for _, label in labeled)
    print(f"{len(examples)} examples ({off} OFF, {len(examples) - off} ON) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
