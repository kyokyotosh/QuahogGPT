"""Two-turn conversations about test pages, for evals/chat.py.

Each conversation is an opening message about a test page, then a follow-up that only
makes sense in context ("When did it first air?"). Kinds:
  episode_*   opener about an episode; follow-up asks air date / writer / director / season
  debut       "Who is X?" then "When did they first appear?"
  song        opener about a song; follow-up asks its episode
  off_topic   on-topic opener, then an off-topic follow-up from guard_test.jsonl (refuse)
  open        on-topic opener, then an open follow-up ("Summarize what happens in it.")

Gold answers come from data/qa/structured.jsonl (test split), so the scoring is exact.

Usage:
    python -m qa.build_chat_test   # writes data/qa/chat_test.jsonl
"""

import collections
import json
import pathlib
import random

QA_DIR = pathlib.Path("data/qa")
OUT = QA_DIR / "chat_test.jsonl"
PER_KIND = 20

EPISODE_OPENERS = ['Tell me about the Family Guy episode "{t}".', 'What happens in "{t}"?']
FOLLOW = {
    "air_date": ["When did it first air?", "When was that one originally broadcast?"],
    "writer": ["Who wrote it?", "Who was the writer of that episode?"],
    "director": ["Who directed it?", "And who was the director?"],
    "season": ["Which season is it from?", "What season was that in?"],
}
OPEN_FOLLOW = ["Summarize what happens in it.", "Why is it memorable?", "Write a two-line joke a character might say in it."]


def main():
    rng = random.Random(0)
    test = [json.loads(l) for l in (QA_DIR / "structured.jsonl").open(encoding="utf-8")]
    test = [x for x in test if x["split"] == "test"]
    by_kind = collections.defaultdict(list)
    for x in test:
        by_kind[x["kind"]].append(x)
    off = [json.loads(l) for l in (QA_DIR / "guard_test.jsonl").open(encoding="utf-8")]
    off = [x["question"] for x in off if x["kind"].startswith("off_")]

    convos = []
    for kind in FOLLOW:
        for x in rng.sample(by_kind[kind], min(PER_KIND, len(by_kind[kind]))):
            convos.append({"kind": f"episode_{kind}", "page": x["page"], "answers": x["answers"],
                           "turns": [rng.choice(EPISODE_OPENERS).format(t=x["page"]), rng.choice(FOLLOW[kind])]})
    for x in rng.sample(by_kind["debut"], PER_KIND):
        convos.append({"kind": "debut", "page": x["page"], "answers": x["answers"],
                       "turns": [f"Who is {x['page']}?", "When did they first appear on the show?"]})
    for x in rng.sample(by_kind["song_episode"], PER_KIND):
        convos.append({"kind": "song", "page": x["page"], "answers": x["answers"],
                       "turns": [f'Tell me about the song "{x["page"]}".', "Which episode is it from?"]})
    episodes = [x["page"] for x in by_kind["air_date"]]
    for t, q in zip(rng.sample(episodes, PER_KIND), rng.sample(off, PER_KIND)):
        convos.append({"kind": "off_topic", "page": t, "answers": [],
                       "turns": [rng.choice(EPISODE_OPENERS).format(t=t), q]})
    for t in rng.sample(episodes, PER_KIND):
        convos.append({"kind": "open", "page": t, "answers": [],
                       "turns": [rng.choice(EPISODE_OPENERS).format(t=t), rng.choice(OPEN_FOLLOW)]})
    with OUT.open("w", encoding="utf-8") as fh:
        for c in convos:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(convos)} conversations -> {OUT}", dict(collections.Counter(c["kind"] for c in convos)))


if __name__ == "__main__":
    main()
