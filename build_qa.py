"""Question-answer pairs from the wiki infoboxes, split by page.

Pages with crc32(title) % 10 == 0 are test pages; every other page is a train page.
A question belongs to the page it is about, so no test page's facts are trained on.

Structured questions have exact gold answers taken from infobox fields:
  episodes:   air date, writers, director, season
  characters: voice actor, full name
  any page:   debut episode (earliest-aired episode whose infobox lists the page)
  songs:      the episode the song is from

Each item: {"question", "answers" (accepted strings), "page", "kind", "split"}.
Free-form questions generated from passages are added separately (see HANDOFF.md).

Usage:
    python build_qa.py     # writes data/qa/structured.jsonl
"""

import json
import pathlib
import random
import zlib

from qa_retrieve import Facts, link_targets, load_raw, parse_date, plain

OUT = pathlib.Path("data/qa/structured.jsonl")

EPISODE = {
    "air_date": [
        'When did the episode "{t}" first air?',
        'What is the original air date of "{t}"?',
        'When was the Family Guy episode "{t}" first broadcast?',
    ],
    "writer": [
        'Who wrote the episode "{t}"?',
        'Who is credited as the writer of "{t}"?',
    ],
    "director": [
        'Who directed the episode "{t}"?',
        'Who was the director of "{t}"?',
    ],
    "season": [
        'Which season of Family Guy is "{t}" from?',
        'In what season did "{t}" air?',
    ],
}
CHARACTER = {
    "voice": ["Who voices {t}?", "Who is the voice actor for {t}?", "Who provides the voice of {t}?"],
    "full_name": ["What is {t}'s full name?", "What is the full name of {t}?"],
}
DEBUT = [
    "When did {t} first appear?",
    "In which episode did {t} first appear?",
    "What was {t}'s first episode?",
]
SONG = ['Which episode features the song "{t}"?', 'In what episode is the song "{t}" sung?']
TEMPLATES = {**EPISODE, **CHARACTER, "debut": DEBUT, "song_episode": SONG}  # kind -> phrasings


def structured_reply(item, facts):
    """A one-sentence answer for a structured item, used as a training target."""
    t, a = item["page"], item["answers"]
    kind = item["kind"]
    if kind == "air_date":
        return f'"{t}" first aired on {a[0]}.'
    if kind == "writer":
        return f'"{t}" was written by {" and ".join(a)}.'
    if kind == "director":
        return f'"{t}" was directed by {" and ".join(a)}.'
    if kind == "season":
        return f'"{t}" is from {a[0].lower()} of Family Guy.'
    if kind == "voice":
        return f"{t} is voiced by {a[0]}."
    if kind == "full_name":
        return f"{t}'s full name is {a[0]}."
    if kind == "song_episode":
        return f'"{t}" is from the episode "{a[0]}".'
    if kind == "debut":
        ep, date = facts.debuts[t]
        return f'{t} first appeared in the episode "{ep}", which first aired on {date.strftime("%B")} {date.day}, {date.year}.'
    raise ValueError(kind)


def split_of(page):
    return "test" if zlib.crc32(page.encode("utf-8")) % 10 == 0 else "train"


def date_answers(date):
    month = date.strftime("%B")
    return [f"{month} {date.day}, {date.year}", f"{month} {date.day} {date.year}", f"{date.day} {month} {date.year}"]


def build(facts, rng):
    items = []

    def add(kind, page, templates, answers):
        answers = [a for a in dict.fromkeys(a.strip() for a in answers) if a]
        if answers:
            q = rng.choice(templates).format(t=page)
            items.append({"question": q, "answers": answers, "page": page, "kind": kind, "split": split_of(page)})

    for page in facts.pages:
        box, f = facts.boxes[page]
        if box == "EpisodeTabs":
            date = parse_date(plain(f.get("first_aired", "")))
            if date:
                add("air_date", page, EPISODE["air_date"], date_answers(date))
            writers = [plain(f"[[{w}]]") for w in link_targets(f.get("writers", ""))]
            add("writer", page, EPISODE["writer"], writers or [plain(f.get("writers", ""))])
            directors = [plain(f"[[{d}]]") for d in link_targets(f.get("director", ""))]
            add("director", page, EPISODE["director"], directors or [plain(f.get("director", ""))])
            season = plain(f.get("season", ""))
            if season.isdigit():
                add("season", page, EPISODE["season"], [f"season {season}", f"Season {season}"])
        elif box == "CharacterBox":
            voices = link_targets(f.get("voiced_by", ""))[:1]
            add("voice", page, CHARACTER["voice"], [plain(f"[[{v}]]") for v in voices])
            add("full_name", page, CHARACTER["full_name"], [plain(f.get("full_name", "")).split(" (")[0]])
        elif box == "SongBox":
            ep = link_targets(f.get("episode", ""))
            add("song_episode", page, SONG, [plain(f"[[{e}]]") for e in ep[:1]])
        if page in facts.debuts:
            ep, _ = facts.debuts[page]
            add("debut", page, DEBUT, [ep])
    return items


def main():
    facts = Facts(load_raw())
    items = build(facts, random.Random(0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = {}
    for it in items:
        key = (it["split"], it["kind"])
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(*key, counts[key])
    print(f"{len(items)} items -> {OUT}")


if __name__ == "__main__":
    main()
