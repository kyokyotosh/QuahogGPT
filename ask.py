"""Answer Family Guy questions: wiki retrieval + a small instruct model (MLX).

Needs the MLX environment (Python 3.12; MLX has no wheels for the main venv's Python):
    /opt/homebrew/bin/python3.12 -m venv .venv-mlx
    .venv-mlx/bin/pip install -r requirements-mlx.txt

The LoRA adapter (adapters/qa, trained with lora_config.yaml) is used by default.

Usage:
    .venv-mlx/bin/python ask.py "Who is Peter Griffin?" "Where is Quahog located?"
    .venv-mlx/bin/python ask.py --show-passages "When did Joe Swanson first appear?"
    .venv-mlx/bin/python ask.py --no-adapter "Who is Peter Griffin?"    # base Qwen
    .venv-mlx/bin/python ask.py --no-retrieval "Who is Peter Griffin?"
"""

import argparse

from mlx_lm import generate, load

from qa_retrieve import load_retriever

MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER = "adapters/qa"
MAX_TOKENS = 120
SYSTEM = (
    "You answer questions about the TV show Family Guy. Use the wiki excerpts when they "
    "contain the answer. Answer directly in one or two sentences. If the excerpts do not "
    "contain the answer, say you are not sure."
)
SYSTEM_NO_RETRIEVAL = (
    "You answer questions about the TV show Family Guy. Answer directly in one or two sentences."
)


def user_message(question, passages):
    if not passages:
        return question
    excerpts = "\n\n".join(f"[{i + 1}] {title}: {text}" for i, (title, text) in enumerate(passages))
    return f"Wiki excerpts:\n{excerpts}\n\nQuestion: {question}"


def messages(question, passages):
    system = SYSTEM if passages else SYSTEM_NO_RETRIEVAL
    return [{"role": "system", "content": system}, {"role": "user", "content": user_message(question, passages)}]


class QA:
    def __init__(self, adapter=None, retrieval=True, model=MODEL):
        self.model, self.tokenizer = load(model, adapter_path=adapter)
        self.retriever = load_retriever() if retrieval else None

    def passages(self, question):
        return self.retriever.retrieve(question) if self.retriever else []

    def answer(self, question):
        msgs = messages(question, self.passages(question))
        prompt = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tokenizer, prompt, max_tokens=MAX_TOKENS).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--adapter", default=ADAPTER, help="LoRA adapter directory")
    parser.add_argument("--no-adapter", action="store_true", help="use the base model")
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--show-passages", action="store_true")
    args = parser.parse_args()
    qa = QA(adapter=None if args.no_adapter else args.adapter, retrieval=not args.no_retrieval)
    for q in args.questions:
        print(f"Q: {q}")
        if args.show_passages:
            for title, text in qa.passages(q):
                print(f"   [{title}] {text[:150]}")
        print(f"A: {qa.answer(q)}\n")


if __name__ == "__main__":
    main()
