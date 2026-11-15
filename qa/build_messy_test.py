"""Casually typed test messages, for checking that the assistant survives real input.

Real users type in lowercase, skip punctuation, and misspell names. Two parts:
  messy copies   every message in test.jsonl (factual, with gold answers) and
                 guard_test.jsonl (off-topic / open), lowercased, without ? and quotes,
                 and 40% with a one-letter typo in the longest name
  casual         hand-written backstory questions about the main characters, typed the
                 way people type (kind "casual"; must be answered, not refused)

Usage:
    python -m qa.build_messy_test   # writes data/qa/messy_test.jsonl
"""

import json
import pathlib
import random
import re

QA_DIR = pathlib.Path("data/qa")
OUT = QA_DIR / "messy_test.jsonl"
TYPO_RATE = 0.4

CASUAL = [
    "Where did Clevand was born", "Who is quagmire", "where is quahog", "was Brian adopted",
    "where was cleveland born", "whats stewies full name", "how old is peter", "who is lois dad",
    "does meg have a boyfriend", "what does joe do for a living", "is brian a dog", "who voices quagmir",
    "why is joe in a wheelchair", "who is chris griffin", "does stewie have a british accent",
    "who are stewies parents", "where does peter work", "is lois rich", "what is the drunken clam",
    "who is mort goldman", "who is cleveland married to", "does quagmire have kids", "who is herbert",
    "what happened to brian in life of brian", "is meg adopted", "who is bonnie swanson",
    "whats the name of the griffins street", "how did peter meet lois", "who is carter pewterschmidt",
    "who is tom tucker", "who is consuela", "where did quagmire serve in the military",
    "how many kids do peter and lois have", "what is peters job", "who is death", "is stewie gay",
    "who is rupert", "what is brians favorite drink", "who is mayor adam west", "who is evil monkey",
]


def mess(text, rng):
    """Lowercase, drop ? and quotes, and sometimes misspell the longest capitalized word."""
    words = re.findall(r"[A-Z][a-z]{5,}", text)
    if words and rng.random() < TYPO_RATE:
        word = max(words, key=len)
        i = rng.randrange(1, len(word) - 1)
        typo = word[:i] + word[i + 1 :] if rng.random() < 0.5 else word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        text = text.replace(word, typo, 1)
    return re.sub(r"[?\"\u201c\u201d]", "", text.lower()).strip()


def main():
    rng = random.Random(0)
    rows = []
    for x in (json.loads(l) for l in (QA_DIR / "test.jsonl").open(encoding="utf-8")):
        rows.append({**x, "question": mess(x["question"], rng), "clean": x["question"], "kind": f"fact_{x['kind']}"})
    for x in (json.loads(l) for l in (QA_DIR / "guard_test.jsonl").open(encoding="utf-8")):
        rows.append({**x, "question": mess(x["question"], rng), "clean": x["question"]})
    rows += [{"question": q, "kind": "casual"} for q in CASUAL]
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} messages -> {OUT}")


if __name__ == "__main__":
    main()
