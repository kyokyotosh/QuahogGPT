"""Answer matching for the QA test sets: lowercase, drop punctuation and articles, then
check whether any accepted answer appears in the prediction as a whole-word phrase."""

import re


def norm(text):
    text = text.lower().replace("\u2019", "'")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return " ".join(text.split())


def correct(prediction, answers):
    p = f" {norm(prediction)} "
    return any(f" {norm(a)} " in p for a in answers if norm(a))
