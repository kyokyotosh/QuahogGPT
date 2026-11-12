"""Step 5: retrieval over the wiki pages.

Pages are cut into short passages (a few sentences each). BM25 scores passages
against a query, so the model can be shown the wiki text relevant to what it is
writing, instead of guessing from its weights alone.
"""

import collections
import math
import re

from step3_scrape import load_docs

CHUNK_CHARS = 200  # rough passage size; a single long sentence stays whole
K1 = 1.5  # BM25 term-frequency saturation
B = 0.75  # BM25 length normalization

TOKEN = re.compile(r"\w+")


def tokenize(text):
    return TOKEN.findall(text.lower())


def chunk_page(title, body):
    """Group consecutive sentences into passages of at most CHUNK_CHARS."""
    sentences = re.split(r"(?<=[.!?])\s+", body)
    passages, current = [], ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > CHUNK_CHARS:
            passages.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        passages.append(current)
    return [(title, passage) for passage in passages]


class Index:
    """BM25 index over (title, text) passages. The title is indexed with its passage."""

    def __init__(self, passages):
        self.passages = passages
        self.n = len(passages)
        self.doc_len = []
        self.postings = collections.defaultdict(list)  # term -> [(passage_id, term_frequency)]
        for i, (title, text) in enumerate(passages):
            counts = collections.Counter(tokenize(f"{title} {text}"))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.avg_len = sum(self.doc_len) / self.n

    def search(self, query, k=1):
        """Top-k passages for `query`, best first. Empty if no query term appears in the index."""
        scores = collections.defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                norm = K1 * (1 - B + B * self.doc_len[i] / self.avg_len)
                scores[i] += idf * tf * (K1 + 1) / (tf + norm)
        best = sorted(scores.items(), key=lambda item: -item[1])[:k]
        return [self.passages[i] for i, _ in best]


def load_index():
    """Index every passage of every page that the corpus build keeps."""
    docs, _ = load_docs()
    passages = [p for title, body in docs for p in chunk_page(title, body)]
    return Index(passages)
