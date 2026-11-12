"""Step 3a: download Family Guy wiki pages and turn their wikitext into plain text.

Two stages, so cleaning can be re-run without hitting the wiki again:
  1. fetch  -> data/raw/pages.json   (title -> raw wikitext)
  2. clean  -> data/corpus.txt       (plain text, one blank line between pages)

Wikitext is the wiki's markup language: [[links]], {{templates}}, == headings ==,
<ref> footnotes, and tables. Models should see prose, not markup.
"""

import json
import pathlib
import re
import time
import urllib.parse
import urllib.request

API = "https://familyguy.fandom.com/api.php"
USER_AGENT = "QuahogGPT-learning-project/0.1 (personal learning project)"
MAX_PAGES = 400
BATCH = 50  # MediaWiki allows up to 50 titles per request for non-bot clients

RAW_PATH = pathlib.Path("data/raw/pages.json")
CORPUS_PATH = pathlib.Path("data/corpus.txt")


def api(**params):
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def list_titles(limit):
    """Article titles in the main namespace, skipping redirects."""
    titles, cont = [], {}
    while len(titles) < limit:
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
    return titles[:limit]


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


def fetch_all():
    titles = list_titles(MAX_PAGES)
    print(f"listing done: {len(titles)} titles")
    pages = {}
    for i in range(0, len(titles), BATCH):
        pages.update(fetch_wikitext(titles[i : i + BATCH]))
        print(f"fetched {min(i + BATCH, len(titles))}/{len(titles)}")
        time.sleep(0.5)  # be polite to the wiki's servers
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


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
        line = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", line)  # == heading == -> heading
        line = re.sub(r"^[*#:;]+\s*", "", line)  # list bullets
        lines.append(line.rstrip())
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def build_corpus(pages):
    docs = []
    for title, wikitext in pages.items():
        body = clean(wikitext)
        if len(body) >= 200:  # skip stubs
            docs.append(f"{title}\n{body}")
    corpus = "\n\n".join(docs)
    CORPUS_PATH.write_text(corpus, encoding="utf-8")
    print(f"kept {len(docs)}/{len(pages)} pages, {len(corpus):,} characters -> {CORPUS_PATH}")


if __name__ == "__main__":
    if RAW_PATH.exists():
        pages = json.loads(RAW_PATH.read_text(encoding="utf-8"))
        print(f"using cached {len(pages)} pages from {RAW_PATH}")
    else:
        pages = fetch_all()
    build_corpus(pages)
