"""Retrieval for question answering.

quahog/retrieve.py ranks 200-character passages with BM25. That works for continuing text,
but questions like "Who is Peter Griffin?" pull short passages that merely mention
the name. This module adds:

  text:     page text from quahog.scrape.page_text, which includes episode plots and
            episode subpages (Trivia, Goofs, Quotes, References) stored in template
            fields. Subpages fold into their episode.
  facts:    one document per page with its lead paragraph, its infobox fields as
            sentences (episode air dates, writers, voice actors, ...), and the
            character's debut episode, computed from the episode infoboxes.
  linking:  page titles (and short aliases) found in the question. Linked pages
            contribute their facts document and their best-matching passages.
  fallback: BM25 over all passages and facts documents.

Usage:
    from quahog.qa_retrieve import load_retriever
    r = load_retriever()
    for title, text in r.retrieve("When did Joe Swanson first appear?"):
        ...
"""

import collections
import datetime
import json
import pathlib
import re

from quahog.retrieve import Index, chunk_page, tokenize
from quahog.scrape import RAW_DIR, clean, page_text, parse_box

LEAD_CHARS = 600  # lead paragraph budget in a facts document
FIELD_CHARS = 200  # per-field budget, so one long infobox field can't crowd out the lead
MIN_BODY = 200  # same stub threshold as quahog.scrape.load_docs
BOX_TYPES = {"EpisodeTabs", "CharacterBox", "SongBox"}
K = 8  # passages per question
LINK_BONUS = 8.0  # BM25 bonus for passages of pages named in the question
PER_PAGE = 4  # BM25 passages per page (facts documents not counted)
MAX_DOCS = 3  # facts documents of named pages placed first
STOP_ALIASES = {"who", "what", "when", "where", "which", "why", "how", "the", "family guy"}
HATNOTE = re.compile(r"^(see also|for other uses|not to be confused|this article|main article)\b", re.I)
MAJOR_PAGES = 400  # most-linked pages that may be named in lowercase or with a typo (plus every character page)
WORDS = pathlib.Path("/usr/share/dict/words")  # real English words are never typo-corrected into a name

CHARACTER_FIELDS = [
    ("full_name", "Full name"),
    ("voiced_by", "Voiced by"),
    ("age", "Age"),
    ("jobs", "Occupation"),
]
EPISODE_FIELDS = [
    ("first_aired", "First aired"),
    ("writers", "Written by"),
    ("director", "Directed by"),
    ("guest_starring", "Guest starring"),
    ("featuring", "Featuring"),
    ("in_a_sentence", "In a sentence"),
]


def load_raw():
    """All raw pages as {title: wikitext}, merged in the same order as load_docs."""
    pages = {}
    for path in sorted(RAW_DIR.glob("*.json")):
        pages.update(json.loads(path.read_text(encoding="utf-8")))
    return pages


def plain(value, limit=None):
    """Wikitext field value as one line of plain text, optionally cut at a comma."""
    text = clean(value)
    parts = [p.strip(" ,;") for p in text.splitlines()]
    text = ", ".join(p for p in parts if p)
    if limit and len(text) > limit:
        cut = text[:limit]
        text = cut[: cut.rfind(", ")] if ", " in cut else cut
        text += ", etc"
    return text


def link_targets(value):
    """Page names linked in a field value: [[Target|label]] -> Target."""
    return [t.strip() for t in re.findall(r"\[\[([^|\]#]+)", value)]


def alias_key(text, lower=True):
    """Words of a title or question joined by single spaces; punctuation such as commas,
    colons, and quotes is dropped so "Peter, Peter, Pumpkin Cheater" matches either way."""
    words = [w.strip(".'") for w in re.findall(r"[\w.'&!-]+", text.replace("\u2019", "'"))]
    text = " ".join(w for w in words if w)
    return text.lower() if lower else text


def parse_date(text):
    for fmt in ("%B %d, %Y", "%B %Y", "%Y"):
        try:
            return datetime.datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            pass
    return None


def lead(body):
    """First real paragraph (hatnotes skipped), cut at a sentence end within LEAD_CHARS."""
    for para in body.strip().split("\n\n"):
        lines = [l for l in para.splitlines() if l.strip() and not HATNOTE.match(l.strip())]
        para = " ".join(lines).strip()
        if len(para) >= 40:
            break
    else:
        return ""
    if len(para) <= LEAD_CHARS:
        return para
    cut = para[:LEAD_CHARS]
    end = max(cut.rfind(". "), cut.rfind('." '))
    return cut[: end + 1] if end > 0 else cut


def labeled(fields, spec):
    parts = []
    for key, label in spec:
        v = plain(fields.get(key, ""), FIELD_CHARS)
        if v:
            parts.append(f"{label}: {v.rstrip('.')}.")
    return " ".join(parts)


def episode_sentence(title, f):
    s, n, total = (plain(f.get(k, "")) for k in ("season", "episode_number", "total"))
    line = ""
    if s and n:
        line = f'"{title}" is episode {n} of season {s} of Family Guy'
        if total:
            prod = plain(f.get("prod_no", ""))
            line += f" ({total} overall, production code {prod})" if prod else f" ({total} overall)"
        line += "."
    return " ".join(p for p in (line, labeled(f, EPISODE_FIELDS)) if p)


def song_sentence(title, f):
    parts = []
    ep, singers = plain(f.get("episode", "")), plain(f.get("singers", ""), FIELD_CHARS)
    if ep:
        parts.append(f'"{title}" is a song from the episode "{ep}".')
    if singers:
        parts.append(f"Sung by: {singers}.")
    return " ".join(parts)


class Facts:
    """Page texts, facts documents, and title aliases."""

    def __init__(self, raw):
        self.boxes = {t: parse_box(w) for t, w in raw.items()}
        self.bodies = {}
        for t, w in raw.items():
            box, f = self.boxes[t]
            body = page_text(w, f)
            if len(body) >= MIN_BODY or box in BOX_TYPES:
                self.bodies[t] = body
        self.redirects = {}
        for t, w in raw.items():
            m = re.match(r"\s*#redirect\s*\[\[([^\]|#]+)", w, re.I)
            if m:
                self.redirects[t] = m.group(1).strip()
        self.pages = [t for t in self.bodies if self.page_of(t) == t]
        self.short_names = self._short_names()
        self.debuts = self._debuts()
        self.docs = {t: self._doc(t) for t in self.pages}
        self.aliases = self._aliases()
        self.major = self._major(raw)

    def page_of(self, title):
        """Episode subpages ("X/Notes/Trivia") belong to page X."""
        base = title.split("/")[0]
        return base if base != title and base in self.bodies else title

    def resolve(self, name):
        """Link target -> page title, following redirects and short character names."""
        name = name.replace("_", " ").strip()
        name = name[:1].upper() + name[1:]
        for _ in range(3):
            if name in self.bodies:
                return self.page_of(name)
            if name not in self.redirects:
                break
            name = self.redirects[name]
        return self.short_names.get(name.lower())

    def _short_names(self):
        """Character box `name` and first word of the title -> character page."""
        short = {}
        for t in self.pages:
            box, f = self.boxes[t]
            if box != "CharacterBox":
                continue
            name = plain(f.get("name", ""))
            if name:
                short.setdefault(name.lower(), t)
            short.setdefault(t.split()[0].lower(), t)
        return short

    def _debuts(self):
        """Page -> (episode title, air date) of the earliest-aired episode listing it."""
        episodes = []
        for t in self.pages:
            box, f = self.boxes[t]
            if box != "EpisodeTabs":
                continue
            date = parse_date(plain(f.get("first_aired", "")))
            if date is None:
                continue
            listed = []
            for key in ("featuring", "also_appearing", "guest_starring"):
                listed += link_targets(f.get(key, ""))
            episodes.append((date, t, listed))
        episodes.sort()
        debuts = {}
        for date, ep, listed in episodes:
            for name in listed:
                page = self.resolve(name)
                if page and page not in debuts and page != ep:
                    debuts[page] = (ep, date)
        return debuts

    def _doc(self, title):
        box, f = self.boxes[title]
        parts = []
        if box == "EpisodeTabs":
            parts.append(episode_sentence(title, f))
        parts.append(lead(self.bodies[title]))
        if box == "SongBox":
            parts.append(song_sentence(title, f))
        elif box == "CharacterBox":
            parts.append(labeled(f, CHARACTER_FIELDS))
        if title in self.debuts:
            ep, date = self.debuts[title]
            parts.append(
                f'{title} first appeared in the episode "{ep}", which first aired on '
                f"{date.strftime('%B')} {date.day}, {date.year}."
            )
        return " ".join(p for p in parts if p)

    def _aliases(self):
        """Alias key (see alias_key) -> page title. Exact titles win over derived aliases."""
        exact, derived = {}, collections.defaultdict(list)
        for t in self.pages:
            exact[t.lower()] = t
            base = re.sub(r"\s*\(.*?\)\s*$", "", t)
            for a in {base, base.split(",")[0]}:
                if a and a.lower() != t.lower():
                    derived[a.lower()].append(t)
        for t, target in self.redirects.items():
            page = self.resolve(target)
            if page in self.docs:
                derived[t.lower()].append(page)
        for short, t in self.short_names.items():
            derived[short].append(t)

        def rank(t):  # prefer episodes and characters, then longer pages
            return (self.boxes[t][0] in ("EpisodeTabs", "CharacterBox"), len(self.bodies[t]))

        aliases = {alias_key(a): max(ts, key=rank) for a, ts in derived.items()}
        aliases.update({alias_key(a): t for a, t in exact.items()})
        for a in STOP_ALIASES:
            aliases.pop(a, None)
        return {a: t for a, t in aliases.items() if len(a) >= 3}

    def _major(self, raw):
        """Character pages plus the MAJOR_PAGES pages most linked from other pages (Quahog,
        Peter Griffin, the Drunken Clam, ...): names users type casually."""
        inbound = collections.Counter()
        for t, w in raw.items():
            for target in set(link_targets(w)):
                page = self.resolve(target)
                if page and page != t:
                    inbound[page] += 1
        top = {p for p, _ in inbound.most_common(MAJOR_PAGES)}
        return top | {t for t in self.pages if self.boxes[t][0] == "CharacterBox"}


def typo_distance(a, b, limit):
    """Edit distance with transpositions (optimal string alignment), or limit + 1 if larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class Retriever:
    def __init__(self, facts, link_bonus=LINK_BONUS, per_page=PER_PAGE):
        self.facts = facts
        self.link_bonus = link_bonus
        self.per_page = per_page
        passages = [p for t, body in facts.bodies.items() for p in chunk_page(t, body)]
        self.index = Index(passages + list(facts.docs.items()))
        self.n_passages = len(passages)
        self.page_ids = collections.defaultdict(list)  # page -> passage and facts ids
        for i, (title, _) in enumerate(self.index.passages):
            self.page_ids[facts.page_of(title)].append(i)
        self.max_alias_words = max(len(a.split()) for a in facts.aliases)
        # One-word names of major pages, for typo correction ("Clevand" -> Cleveland Brown).
        self.fuzzy_names = {a: t for a, t in facts.aliases.items() if " " not in a and len(a) >= 4 and t in facts.major}
        self.english = {w for w in WORDS.read_text().split() if w.islower()} if WORDS.exists() else None

    def link(self, question):
        """Page titles named in the question, longest match first, without overlaps.

        "Family Guy" is removed before matching so that "Guy" alone cannot link "Guy, Robot",
        but titles of three or more words are matched on the full question first, so
        '"Family Guy" Through the Years' still links."""
        question = question.replace("\u2019", "'")
        full = self._match(alias_key(question, lower=False).split(), min_words=3)
        stripped = re.sub(r"\bfamily guy('s)?\b", " ", question, flags=re.I)
        return list(dict.fromkeys(full + self._match(alias_key(stripped, lower=False).split())))

    def _match(self, words, min_words=1):
        cased = any(w[:1].isupper() for w in words[1:])  # the question uses capitals for names
        found, used = [], set()
        for n in range(min(self.max_alias_words, len(words)), min_words - 1, -1):
            for i in range(len(words) - n + 1):
                span = set(range(i, i + n))
                if span & used:
                    continue
                phrase = " ".join(words[i : i + n])
                title = self.facts.aliases.get(phrase.lower())
                if not title and phrase.lower().endswith("'s"):
                    title = self.facts.aliases.get(phrase.lower()[:-2])
                if not title:
                    continue
                capital = any(w[:1].isupper() for w in words[i : i + n])
                if not capital and (n == 1 or cased) and title not in self.facts.major:
                    continue  # "voice" in "voice actor" is not the show "The Voice"; "quagmire" is Quagmire
                found.append((i, title))
                used |= span
        if min_words == 1:
            for i, w in enumerate(words):
                title = None if i in used else self.correct_typo(w)
                if title:
                    found.append((i, title))
        return [t for _, t in sorted(found)]

    def correct_typo(self, word):
        """The major page a misspelled one-word name means, or None. Only words of 5+ letters
        that are not English words; one edit allowed (two for 7+ letters)."""
        w = word.lower().removesuffix("'s")
        if self.english is None or len(w) < 5 or not w.isalpha() or w in self.english or w in self.facts.aliases:
            return None
        limit = 1 if len(w) < 7 else 2
        best = None
        for name, title in self.fuzzy_names.items():
            d = typo_distance(w, name, limit)
            if d <= limit and (best is None or d < best[0]):
                best = (d, title)
        return best[1] if best else None

    def retrieve(self, question, k=K, max_docs=MAX_DOCS):
        """Up to k (title, text): facts of the first max_docs pages named in the question,
        then passages by BM25 with a bonus for named pages.

        At most per_page passages come from one page, and a passage already contained
        in a chosen facts document is skipped.
        """
        linked = self.link(question)
        out = [(t, self.facts.docs[t]) for t in linked[:max_docs]][:k]
        chosen_docs = [text for _, text in out]
        scores = self.index.scores(question)
        for page in linked:
            for i in self.page_ids[page]:
                scores[i] += self.link_bonus
        taken = collections.Counter()
        for i, _ in sorted(scores.items(), key=lambda item: -item[1]):
            if len(out) >= k:
                break
            title, text = self.index.passages[i]
            page = self.facts.page_of(title)
            if taken[page] >= self.per_page or any(text in d for d in chosen_docs):
                continue
            if i >= self.n_passages:
                chosen_docs.append(text)
            taken[page] += 1
            out.append((title, text))
        return out


def load_retriever():
    return Retriever(Facts(load_raw()))
