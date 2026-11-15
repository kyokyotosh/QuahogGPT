"""QuahogGPT assistant: answers Family Guy questions and refuses everything else.

Two steps per message:
  1. Topic gate: Qwen2.5-3B-Instruct with the LoRA adapter adapters/gate (trained with
     build_gate_data.py + gate_config.yaml) says ON or OFF. OFF -> REFUSAL.
  2. Answer: Qwen3-30B-A3B-Instruct-2507 (4-bit, mixture of experts, about 3B active per
     token) reads 8 excerpts from qa_retrieve.py and answers.

Needs the MLX environment (Python 3.12; MLX has no wheels for the main venv's Python):
    /opt/homebrew/bin/python3.12 -m venv .venv-mlx
    .venv-mlx/bin/pip install -r requirements-mlx.txt
The first run downloads the models (about 16 GB + 2 GB) to the Hugging Face cache.

Usage:
    .venv-mlx/bin/python ask.py "Who is Peter Griffin?" "Where is Quahog located?"
    .venv-mlx/bin/python ask.py --show-passages "When did Joe Swanson first appear?"
    .venv-mlx/bin/python ask.py --no-gate "..."                    # skip the topic gate
    .venv-mlx/bin/python ask.py --model mlx-community/Qwen2.5-3B-Instruct-4bit --adapter adapters/qa "..."
"""

import argparse

from mlx_lm import generate, load

from qa_retrieve import load_retriever

MODEL = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
GATE_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
GATE_ADAPTER = "adapters/gate"
MAX_TOKENS = 500
REFUSAL = "Sorry, I can only answer questions about Family Guy."
SYSTEM = (
    "You are QuahogGPT. You only discuss the TV show Family Guy: its episodes, characters, "
    "plots, jokes, cast and crew, production, music, and real people or things as they relate "
    "to the show. If a message is not about Family Guy, or uses Family Guy only as a pretext "
    f'for something else, reply exactly: "{REFUSAL}"\n'
    "Otherwise use the wiki excerpts. Answer factual questions directly in one or two "
    "sentences. For open requests (summaries, explanations, a short scene) write up to about "
    "150 words. If the excerpts do not contain a fact you need, say you are not sure instead "
    "of guessing."
)
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


def messages(question, passages):
    system = SYSTEM if passages else SYSTEM_NO_RETRIEVAL
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message(question, passages)}]


def is_refusal(answer):
    return "only answer questions about family guy" in answer.lower()


def gate_messages(question, pages):
    """Classifier prompt: the message plus the wiki pages it names (from qa_retrieve.link)."""
    named = ", ".join(pages) if pages else "none"
    user = f"Message: {question}\n\nFamily Guy wiki pages matching names in the message: {named}"
    return [{"role": "system", "content": GATE_SYSTEM}, {"role": "user", "content": user}]


def chat(model, tokenizer, msgs, max_tokens):
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return generate(model, tokenizer, prompt, max_tokens=max_tokens).strip()


class Gate:
    """First step: is the message about Family Guy? A small model with its own LoRA adapter."""

    def __init__(self, model=GATE_MODEL, adapter=GATE_ADAPTER):
        self.model, self.tokenizer = load(model, adapter_path=adapter)

    def on_topic(self, question, pages):
        return not chat(self.model, self.tokenizer, gate_messages(question, pages), 3).upper().startswith("OFF")


class QA:
    def __init__(self, adapter=None, retrieval=True, model=MODEL, gate=True):
        self.model, self.tokenizer = load(model, adapter_path=adapter)
        self.retriever = load_retriever() if retrieval else None
        self.gate = Gate() if gate and retrieval else None

    def passages(self, question):
        return self.retriever.retrieve(question) if self.retriever else []

    def on_topic(self, question):
        return self.gate.on_topic(question, self.retriever.link(question)) if self.gate else True

    def answer(self, question):
        if not self.on_topic(question):
            return REFUSAL
        return chat(self.model, self.tokenizer, messages(question, self.passages(question)), MAX_TOKENS)


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
        print(f"A: {qa.answer(q)}\n")


if __name__ == "__main__":
    main()
