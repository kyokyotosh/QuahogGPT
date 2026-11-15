"""Step 3a: download Family Guy wiki pages and turn their wikitext into plain text.

Usage:
  python -m quahog.scrape fetch all              # 400 pages from the whole wiki
  python -m quahog.scrape fetch Characters       # articles in a category tree
  python -m quahog.scrape build                  # merge every cache -> data/corpus.txt

Each `fetch` writes data/raw/<source>.json (title -> raw wikitext), so network
access happens once per source. `build` merges all raw files (deduplicated by
title) and cleans them. Wikitext is the wiki's markup language: [[links]],
{{templates}}, == headings ==, <ref> footnotes, and tables. Models should see prose.
"""

import json
import pathlib
import random
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://familyguy.fandom.com/api.php"
USER_AGENT = "QuahogGPT-learning-project/0.1 (personal learning project)"
MAX_ALL_PAGES = None  # None fetches every main-namespace article
CATEGORY_DEPTH = 2  # how many levels of subcategories to follow
BATCH = 50  # MediaWiki allows up to 50 titles per request for non-bot clients

RAW_DIR = pathlib.Path("data/raw")
CORPUS_PATH = pathlib.Path("data/corpus.txt")
TEXT_FIELDS = ("plot", "contents")  # template fields that hold page text (episode plots, subpages)


def api(**params):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def all_titles(limit=None):
    """Article titles in the main namespace, skipping redirects. limit=None means all."""
    titles, cont = [], {}
    while limit is None or len(titles) < limit:
        data = api(
            action="query",
            list="allpages",
            apnamespace=0,
            apfilterredir="nonredirects",
            aplimit=500,
            **cont,
        )
        titles += [p["title"] for p in data["query"]["allpages"]]
        if "continue" not in data:
            break
        cont = data["continue"]
    return titles if limit is None else titles[:limit]


def category_members(category):
    """Every member of one category (articles and subcategories), following continuation."""
    members, cont = [], {}
    while True:
        data = api(
            action="query",
            list="categorymembers",
            cmtitle=category,
            cmlimit=500,
            **cont,
        )
        members += data["query"]["categorymembers"]
        if "continue" not in data:
            return members
        cont = data["continue"]


def category_titles(root, max_depth=CATEGORY_DEPTH):
    """Article titles under `root` and its subcategories, up to max_depth levels down."""
    seen, titles = set(), set()
    frontier = [f"Category:{root}"]
    for _ in range(max_depth + 1):
        next_frontier = []
        for cat in frontier:
            if cat in seen:
                continue
            seen.add(cat)
            for m in category_members(cat):
                if m["ns"] == 0:
                    titles.add(m["title"])
                elif m["ns"] == 14:
                    next_frontier.append(m["title"])
        frontier = next_frontier
    return sorted(titles)


def fetch_wikitext(titles):
    data = api(
        action="query",
        prop="revisions",
        rvprop="content",
        rvslots="main",
        titles="|".join(titles),
    )
    pages = {}
    for p in data["query"]["pages"].values():
        if "revisions" not in p:
            continue
        slot = p["revisions"][0]["slots"]["main"]
        pages[p["title"]] = slot.get("*") or slot.get("content", "")
    return pages


def fetch(source):
    """Download every page for `source` ("all" or a category name) into data/raw/."""
    if source == "all":
        titles = all_titles(MAX_ALL_PAGES)
    else:
        titles = category_titles(source)
    print(f"{source}: {len(titles)} article titles")
    pages = {}
    for i in range(0, len(titles), BATCH):
        pages.update(fetch_wikitext(titles[i : i + BATCH]))
        print(f"fetched {min(i + BATCH, len(titles))}/{len(titles)}")
        time.sleep(0.5)  # be polite to the wiki's servers
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = source.lower().replace(" ", "-")
    path = RAW_DIR / f"{slug}.json"
    path.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(pages)} pages -> {path}")


def clean(wikitext):
    """Strip wiki markup, keeping the readable text."""
    text = re.sub(r"<!--.*?-->", "", wikitext, flags=re.S)  # HTML comments
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S | re.I)  # footnotes
    while True:  # templates can nest, so remove innermost ones until none remain
        new = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.I)  # images
    text = re.sub(r"\[\[Category:[^\]]*\]\]", "", text, flags=re.I)  # category tags
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)  # [[target|label]] -> label
    text = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", text)  # external links
    text = re.sub(r"'{2,}", "", text)  # bold and italic quotes
    text = re.sub(r"<[^>]+>", "", text)  # leftover HTML tags
    lines = []
    for line in text.splitlines():
        if line.startswith(("{|", "|", "!")):  # table syntax
            continue
        if re.fullmatch(r"\S+\.(?:jpe?g|png|gif|svg)", line.strip(), flags=re.I):  # gallery filenames
            continue
        line = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", line)  # == heading == -> heading
        line = re.sub(r"^[*#:;]+\s*", "", line)  # list bullets
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_box(wikitext):
    """The template at the top of a page as (name, {field: raw value}), or (None, {})."""
    m = re.match(r"\s*\{\{\s*([\w /]+?)\s*(?=[|\n}])", wikitext)
    if not m:
        return None, {}
    depth, i = 0, m.start()
    fields, field_start = [], None
    while i < len(wikitext):
        pair = wikitext[i : i + 2]
        if pair in ("{{", "[["):
            depth += 1
            i += 2
            continue
        if pair in ("}}", "]]"):
            depth -= 1
            i += 2
            if depth == 0:
                if field_start is not None:
                    fields.append(wikitext[field_start : i - 2])
                break
            continue
        if wikitext[i] == "|" and depth == 1:
            if field_start is not None:
                fields.append(wikitext[field_start:i])
            field_start = i + 1
        i += 1
    out = {}
    for f in fields:
        key, sep, value = f.partition("=")
        if sep:
            out[key.strip().lower()] = value.strip()
    return m.group(1).strip(), out


def page_text(wikitext, fields):
    """Readable text of a page, including template text fields that clean() drops."""
    parts = [clean(fields[k]) for k in TEXT_FIELDS if fields.get(k, "").strip()]
    parts.append(clean(wikitext))
    return "\n\n".join(p for p in parts if p)


def load_docs():
    """Unique cleaned pages as (title, body), in a fixed order. Stubs and redirects are dropped."""
    pages = {}
    for path in sorted(RAW_DIR.glob("*.json")):
        found = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path.name}: {len(found)} pages")
        pages.update(found)
    docs = []
    for title, wikitext in pages.items():
        body = page_text(wikitext, parse_box(wikitext)[1])
        if len(body) >= 200:  # skip stubs and redirects
            docs.append((title, body))
    return docs, len(pages)


def build():
    """Merge all raw caches (deduplicated by title), clean them, write the corpus."""
    found, total = load_docs()
    docs = [f"{title}\n{body}" for title, body in found]
    random.Random(0).shuffle(docs)  # mix categories so any contiguous train/val split is representative
    corpus = "\n\n".join(docs)
    CORPUS_PATH.write_text(corpus, encoding="utf-8")
    print(f"kept {len(docs)}/{total} unique pages, {len(corpus):,} characters -> {CORPUS_PATH}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "fetch":
        fetch(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "build":
        build()
    else:
        print(__doc__)
        sys.exit(1)
