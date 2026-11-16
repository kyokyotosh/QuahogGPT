"""Training data for the topic classifier (ask.Gate): each example is ask.gate_messages(...)
with the assistant reply "ON" or "OFF".

Train pages and train-only messages only:
  OFF  data/qa/guard_train.jsonl off_* messages (generated; labels confirmed by a judge)
  ON   data/qa/guard_train.jsonl on_open requests, plus factual train questions from
       structured.jsonl (every kind), generated_train.jsonl and generated_train_2.jsonl
  both data/qa/gate_extra_train.jsonl: casual questions about characters' backstories (ON)
       and real people's private lives (OFF), judge-confirmed, so "was Brian adopted" is not
       mistaken for a question about a real person
Casual copies: MESSY_SHARE of the on-topic examples and every off-topic one are repeated
lowercased, without punctuation and sometimes with a misspelled name (build_messy_test.mess).
Follow-up form (what QA.rewrite sends, via ask.with_subject): off-topic messages that refer
back get a random train subject appended and stay OFF; follow-up phrasings
(followups_train.jsonl, generated, none equal to a chat_test phrasing) get a subject of the
matching kind and are ON. So the subject suffix alone never decides.
Off-topic examples are then repeated until they are OFF_RATIO of the on-topic count: with a
3:1 imbalance the gate leaned ON and confidently passed off-topic messages.

Usage:
    .venv-mlx/bin/python -m qa.build_gate_data   # writes data/qa/gate/{train,valid}.jsonl
    .venv-mlx/bin/python -m mlx_lm lora -c configs/lora_gate.yaml
"""

import json
import pathlib
import random

from qa.ask import REFERS_BACK, gate_messages, with_subject
from qa.build_messy_test import mess
from qa.build_qa import split_of
from quahog.qa_retrieve import load_retriever

QA_DIR = pathlib.Path("data/qa")
OUT_DIR = QA_DIR / "gate"
FACTUAL = {"structured.jsonl": 900, "generated_train.jsonl": 600, "generated_train_2.jsonl": 600}
VALID = 200
MESSY_SHARE = 0.4
SUBJECTS_PER_FOLLOWUP = 3
OFF_RATIO = 0.8


def read(name):
    return [json.loads(line) for line in (QA_DIR / name).open(encoding="utf-8")]


def train_subjects(facts):
    """Train pages a follow-up can be about, by kind."""
    by_box = {"episode": "EpisodeTabs", "song": "SongBox", "character": "CharacterBox"}
    subjects = {k: [p for p in facts.pages if facts.boxes[p][0] == box and split_of(p) == "train"] for k, box in by_box.items()}
    subjects["character"] += [p for p in facts.debuts if facts.boxes.get(p, (None,))[0] is None and split_of(p) == "train"]
    return subjects


def main():
    rng = random.Random(0)
    retriever = load_retriever()
    guard = read("guard_train.jsonl")
    labeled = [(x["question"], "OFF" if x["kind"].startswith("off_") else "ON") for x in guard]
    for name, n in FACTUAL.items():
        rows = [x for x in read(name) if x["split"] == "train"]
        labeled += [(x["question"], "ON") for x in rng.sample(rows, min(n, len(rows)))]
    labeled += [(x["question"], "OFF" if x["kind"].startswith("off_") else "ON") for x in read("gate_extra_train.jsonl")]
    on = [q for q, label in labeled if label == "ON"]
    off = [q for q, label in labeled if label == "OFF"]
    on += [mess(q, rng) for q in rng.sample(on, int(MESSY_SHARE * len(on)))]
    off += [mess(q, rng) for q in off]
    facts = retriever.facts
    subjects = train_subjects(facts)
    everything = [p for pages in subjects.values() for p in pages]
    off += [with_subject(q, rng.choice(everything), facts) for q in off if REFERS_BACK.search(q) and not retriever.link(q)]
    for x in read("followups_train.jsonl"):
        on += [with_subject(x["question"], p, facts) for p in rng.sample(subjects[x["kind"]], SUBJECTS_PER_FOLLOWUP)]
    off += [rng.choice(off) for _ in range(max(0, int(OFF_RATIO * len(on)) - len(off)))]
    labeled = [(q, "ON") for q in on] + [(q, "OFF") for q in off]
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
