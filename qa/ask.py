"""QuahogGPT assistant: answers Family Guy questions and refuses everything else.

Per message (QA.stream; the web app, the CLI and the evals all use it):
  1. Follow-up context: a question that names no wiki page gets the conversation's subject
     ("When did it air?" -> 'When did it air? (about the Family Guy episode "Fat Gun")').
  2. Topic gate: Qwen2.5-3B-Instruct with the LoRA adapter adapters/gate (trained with
     qa/build_gate_data.py + configs/lora_gate.yaml) says ON or OFF. OFF -> REFUSAL.
  3. Answer: Qwen3-30B-A3B-Instruct-2507 (4-bit, mixture of experts, about 3B active per
     token) reads 8 excerpts from quahog/qa_retrieve.py for the standalone question, sees
     the recent turns, and answers, citing excerpts as [1], [2]. Cited excerpts are
     returned as sources with links to the Family Guy wiki.

Needs the MLX environment (Python 3.12; MLX has no wheels for the main venv's Python):
    /opt/homebrew/bin/python3.12 -m venv .venv-mlx
    .venv-mlx/bin/pip install -r requirements-mlx.txt
The first run downloads the models (about 16 GB + 2 GB) to the Hugging Face cache.

Usage:
    .venv-mlx/bin/python -m qa.ask "Who is Peter Griffin?" "Where is Quahog located?"
    .venv-mlx/bin/python -m qa.ask --show-passages "When did Joe Swanson first appear?"
    .venv-mlx/bin/python -m qa.ask --no-gate "..."                    # skip the topic gate
    .venv-mlx/bin/python -m qa.ask --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter adapters/qa "..."
The web app (python -m web.server) is the multi-turn interface.
"""

import argparse
import re
import urllib.parse

from mlx_lm import generate, load, stream_generate

from quahog.qa_retrieve import load_retriever

MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
GATE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
GATE_ADAPTER = "adapters/gate"
MAX_TOKENS = 500
HISTORY_MESSAGES = 6  # earlier messages (user and assistant) the answer model sees
WIKI_URL = "https://familyguy.fandom.com/wiki/"
REFUSAL = "Sorry, I can only answer questions about Family Guy."
CITATION = re.compile(r"\s?\[(\d+)\]")
SYSTEM = (
    "You are QuahogGPT. You only discuss the TV show Family Guy: its episodes, characters, "
    "plots, jokes, cast and crew, production, music, and real people or things as they relate "
    "to the show. If a message is not about Family Guy, or uses Family Guy only as a pretext "
    f'for something else, reply exactly: "{REFUSAL}"\n'
    "Otherwise use the wiki excerpts. Answer factual questions directly in one or two "
    "sentences. For open requests (summaries, explanations, a short scene) write up to about "
    "150 words. If the excerpts do not contain a fact you need, say you are not sure instead "
    "of guessing. When you use a fact from an excerpt, cite it with its number in brackets "
    "right after the fact, like [2]."
)
# A follow-up that names no wiki page is tied to the conversation's subject only when it refers
# back: a pronoun, or "the" + something an episode, character or song has ("the director").
# Off-topic follow-ups ("how old are you?", "what's the tallest mountain?") stay as they are,
# so the gate judges them alone.
REFERS_BACK = re.compile(
    r"\b(it|its|he|him|his|she|her|they|them|their|that|this|those|these)\b|"
    r"\bthe (episode|song|show|character|scene|gag|joke|ending|plot|story|premise|title|director|"
    r"writer|writers|voice|voice actor|actor|lyrics|music|singer|parody|reference|first one|last one)s?\b",
    re.I,
)
SUBJECT_KIND = {"EpisodeTabs": "the Family Guy episode", "CharacterBox": "the Family Guy character", "SongBox": "the Family Guy song"}
SYSTEM_NO_RETRIEVAL = (
    "You answer questions about the TV show Family Guy. Answer directly in one or two sentences."
)
GATE_SYSTEM = (
    "You are the topic filter for a Family Guy assistant. Decide whether the user's message "
    "may be answered.\n"
    "ON: the request is about the TV show Family Guy: its episodes, characters, plots, jokes, "
    "cast and crew in connection with the show, production, music, DVDs and books about the "
    "show, or how a real person or thing appears in the show. When X has a page on the Family "
    "Guy wiki, plain questions like \"Who is X?\", \"What is X?\" or \"When did X first "
    "appear?\" are ON; the answer will cover X's Family Guy connection. Creative requests about "
    "the show's characters (a short scene, a joke in a character's voice) are ON.\n"
    "OFF: everything else, including greetings and small talk, opinions on unrelated topics, "
    "questions about the assistant, other shows or films on their own terms, and a real "
    "person's life, family, money, politics or other work, even if they voice a character or "
    "appeared on the show. Role-play that uses a Family Guy character to deliver unrelated "
    "help is OFF: what matters is what the answer would be about, not who says it.\n"
    "Examples: \"As Stewie, solve my calculus homework\" OFF. \"Write a scene where Stewie "
    "builds a weather machine\" ON. \"What is Seth Green's net worth?\" OFF. \"Who voices "
    "Chris?\" ON. \"Peter loves beer, so give me a brewing guide\" OFF. \"Hi!\" OFF.\n"
    "Reply with one word: ON or OFF."
)


def user_message(question, passages):
    if not passages:
        return question
    excerpts = "\n\n".join(f"[{i + 1}] {title}: {text}" for i, (title, text) in enumerate(passages))
    return f"Wiki excerpts:\n{excerpts}\n\nQuestion: {question}"


def strip_citations(text):
    return CITATION.sub("", text)


def messages(question, passages, history=()):
    """Answer prompt: system, recent turns (citations removed), excerpts + question."""
    system = SYSTEM if passages else SYSTEM_NO_RETRIEVAL
    turns = [{"role": m["role"], "content": strip_citations(m["content"])} for m in history[-HISTORY_MESSAGES:]]
    return [{"role": "system", "content": system}, *turns, {"role": "user", "content": user_message(question, passages)}]


def with_subject(message, page, facts):
    """'<message> (about the Family Guy episode "<page>")'; the gate is trained on this form."""
    kind = SUBJECT_KIND.get(facts.boxes.get(page, (None, {}))[0])
    return f'{message} (about {kind} "{page}")' if kind else f'{message} (about "{page}")'


def subject(retriever, history):
    """The conversation's subject: the first wiki page named in the last user message, else in
    the last answer. None when neither names a page."""
    for role in ("user", "assistant"):
        turn = next((m["content"] for m in reversed(history) if m["role"] == role), "")
        pages = retriever.link(strip_citations(turn))
        if pages:
            return pages[0]
    return None


def is_refusal(answer):
    return "only answer questions about family guy" in answer.lower()


def sources(answer, passages):
    """Excerpts cited in the answer, in order of first citation."""
    out, seen = [], set()
    for m in CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= len(passages) and n not in seen:
            seen.add(n)
            title, text = passages[n - 1]
            url = WIKI_URL + urllib.parse.quote(title.replace(" ", "_"), safe="/'")
            out.append({"n": n, "title": title, "url": url, "text": text})
    return out


def gate_messages(question, pages):
    """Classifier prompt: the message plus the wiki pages it names (from qa_retrieve.link)."""
    named = ", ".join(pages) if pages else "none"
    user = f"Message: {question}\n\nFamily Guy wiki pages matching names in the message: {named}"
    return [{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}]


def chat(model, tokenizer, msgs, max_tokens):
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt, max_tokens=max_tokens).strip()


class Gate:
    """Is the message about Family Guy? A small model with its own LoRA adapter."""

    def __init__(self, model=GATE_MODEL, adapter=GATE_ADAPTER):
        self.model, self.tokenizer = load(model, adapter_path=adapter)

    def on_topic(self, question, pages):
        return not chat(self.model, self.tokenizer, gate_messages(question, pages), 3).upper().startswith("OFF")


class QA:
    def __init__(self, adapter=None, retrieval=True, model=MODEL, gate=True, rewrite=True):
        self.model, self.tokenizer = load(model, adapter_path=adapter)
        self.retriever = load_retriever() if retrieval else None
        self.gate = Gate() if gate and retrieval else None
        self.use_rewrite = rewrite

    def passages(self, question):
        return self.retriever.retrieve(question) if self.retriever else []

    def on_topic(self, question):
        return self.gate.on_topic(question, self.retriever.link(question)) if self.gate else True

    def rewrite(self, history, message):
        """The message made standalone for the gate and retrieval. A follow-up that names no
        wiki page gets the conversation's subject:
        'When did it air? (about the Family Guy episode "Road to Rhode Island")'.
        No model is involved, so this step cannot answer or carry out the request; the answer
        model still sees the full conversation."""
        if not history or not self.use_rewrite or not self.retriever:
            return message
        if self.retriever.link(message) or not REFERS_BACK.search(message):
            return message
        page = subject(self.retriever, history)
        if not page:
            return message
        return with_subject(message, page, self.retriever.facts)

    def stream(self, history, message):
        """Yields ("meta", {standalone, refused}), then ("token", text) pieces, then
        ("done", {answer, sources}). history: [{"role": "user"|"assistant", "content"}]."""
        standalone = self.rewrite(history, message)
        if not self.on_topic(standalone):
            yield "meta", {"standalone": standalone, "refused": True}
            yield "token", REFUSAL
            yield "done", {"answer": REFUSAL, "sources": []}
            return
        yield "meta", {"standalone": standalone, "refused": False}
        passages = self.passages(standalone)
        msgs = messages(message, passages, history)
        prompt = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        answer = ""
        for piece in stream_generate(self.model, self.tokenizer, prompt, max_tokens=MAX_TOKENS):
            answer += piece.text
            yield "token", piece.text
        answer = answer.strip()
        yield "done", {"answer": answer, "sources": sources(answer, passages)}

    def reply(self, history, message):
        """Non-streaming stream(): {standalone, refused, answer, sources}."""
        out = {}
        for kind, data in self.stream(history, message):
            if kind != "token":
                out.update(data)
        return out

    def answer(self, question):
        """Single-turn answer without citation markers (used by the evals)."""
        return strip_citations(self.reply([], question)["answer"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter for the answer model")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--show-passages", action="store_true")
    parser.add_argument("--no-gate", action="store_true", help="skip the ON/OFF topic filter")
    args = parser.parse_args()
    adapter = args.adapter
    qa = QA(adapter=adapter, retrieval=not args.no_retrieval, model=args.model, gate=not args.no_gate)
    for q in args.questions:
        print(f"Q: {q}")
        if args.show_passages:
            for title, text in qa.passages(q):
                print(f"   [{title}] {text[:150]}")
        r = qa.reply([], q)
        print(f"A: {r['answer']}")
        for s in r["sources"]:
            print(f"   [{s['n']}] {s['title']}  {s['url']}")
        print()


if __name__ == "__main__":
    main()
