"""Chat-format training data for the LoRA reader (mlx_lm.lora).

Inputs (train pages only; see build_qa.py for the page split):
  data/qa/structured.jsonl       infobox questions; replies come from templates here
  data/qa/generated_train.jsonl  passage questions with an LLM-written reply
  data/qa/guard_train.jsonl      off-topic messages (off_*) and open-ended Family Guy
                                 requests (on_open) with LLM-written grounded replies

Each example is the exact prompt ask.py sends (system + retrieved excerpts + question)
and the reply as the assistant turn. Examples whose retrieved excerpts do not contain a
gold answer are dropped. ABSTAIN examples remove every excerpt containing the answer
and teach the reply NOT_SURE, so the model declines instead of guessing. Off-topic
messages teach ask.REFUSAL, with whatever excerpts retrieval returns for them.

Usage:
    .venv-mlx/bin/python build_lora_data.py   # writes data/qa/lora/{train,valid}.jsonl
"""

import json
import pathlib
import random

from ask import REFUSAL, messages
from build_qa import structured_reply
from eval_qa import norm
from qa_retrieve import load_retriever

QA_DIR = pathlib.Path("data/qa")
OUT_DIR = QA_DIR / "lora"
DEBUT_CAP = 700  # debut questions outnumber every other kind; keep the mix balanced
ABSTAIN = 300
VALID = 150
NOT_SURE = "I'm not sure. The wiki excerpts I found don't say."


def contains(passages, answers):
    ctx = f" {norm(' '.join(f'{t} {x}' for t, x in passages))} "
    return any(f" {norm(a)} " in ctx for a in answers if norm(a))


def main():
    rng = random.Random(0)
    retriever = load_retriever()
    facts = retriever.facts

    structured = [json.loads(l) for l in (QA_DIR / "structured.jsonl").open(encoding="utf-8")]
    structured = [x for x in structured if x["split"] == "train"]
    debut = [x for x in structured if x["kind"] == "debut"]
    structured = [x for x in structured if x["kind"] != "debut"] + rng.sample(debut, min(DEBUT_CAP, len(debut)))
    for x in structured:
        x["reply"] = structured_reply(x, facts)
    generated = [json.loads(l) for l in (QA_DIR / "generated_train.jsonl").open(encoding="utf-8")]
    assert all(x["split"] == "train" for x in structured + generated)

    examples, dropped = [], 0
    for x in structured + generated:
        passages = retriever.retrieve(x["question"])
        if not contains(passages, x["answers"]):
            dropped += 1
            continue
        examples.append({"messages": messages(x["question"], passages) + [{"role": "assistant", "content": x["reply"]}]})

    abstain = 0
    for x in rng.sample(generated, len(generated)):
        if abstain == ABSTAIN:
            break
        passages = [p for p in retriever.retrieve(x["question"]) if not contains([p], x["answers"])]
        if len(passages) < 3:
            continue
        examples.append({"messages": messages(x["question"], passages) + [{"role": "assistant", "content": NOT_SURE}]})
        abstain += 1

    guard = [json.loads(l) for l in (QA_DIR / "guard_train.jsonl").open(encoding="utf-8")]
    for x in guard:
        reply = REFUSAL if x["kind"].startswith("off_") else x["reply"]
        examples.append({"messages": messages(x["question"], retriever.retrieve(x["question"])) + [{"role": "assistant", "content": reply}]})

    rng.shuffle(examples)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid", examples[:VALID]), ("train", examples[VALID:])):
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(examples)} examples ({abstain} abstain, {len(guard)} guard), {dropped} dropped (answer not retrieved)")
    print(f"train {len(examples) - VALID}, valid {VALID} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
