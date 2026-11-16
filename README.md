# QuahogGPT

A Family Guy language-model project, built step by step to learn how LLMs work:

- **QuahogGPT** models trained from scratch on a Family Guy wiki corpus (bigram, tiny GPT, BPE tokenizer, pretraining, and a from-scratch question-answering reader).
- **The assistant** (`qa/ask.py`): answers questions about Family Guy and refuses everything else. A trained topic gate (Qwen2.5-3B + LoRA) decides, then Qwen3-30B-A3B answers from retrieved wiki excerpts and cites them. On held-out tests: 98.7% of factual questions right (93.8% when typed casually, with typos), 99.4% of follow-up questions handled, 100% of off-topic messages refused.
- **The web app** (`web/`): a chat page for the assistant, with follow-up questions, streamed answers and links to the cited wiki pages.

Results, data notes, and history: [HANDOFF.md](HANDOFF.md).

## Layout

```
quahog/     shared library: gpt.py (model), bpe.py (tokenizer), scrape.py (wiki -> corpus),
            retrieve.py (BM25), qa_retrieve.py (QA retrieval), metrics.py
tutorial/   step1_bigram.py, step2_gpt.py
pretrain/   train.py, sample.py, rag_sample.py
qa/         ask.py (assistant), quahog_qa.py + train_quahog_qa.py (from-scratch reader),
            build_qa.py, build_lora_data.py, build_gate_data.py (training data)
evals/      qa.py, guard.py, same_split.py, sampler_query.py, retrieval ablations
configs/    lora_qa.yaml, lora_gate.yaml (mlx_lm LoRA configs)
web/        server.py (standard-library HTTP server), static/ (index.html, style.css, app.js)
data/       qa/ question sets (committed); corpus, tokenizers, raw pages (gitignored)
```

Run every script from the repo root as a module: `python -m <folder>.<script>`.

## Setup (macOS, Apple Silicon)

Two environments: PyTorch for the from-scratch models, MLX for the assistant (MLX needs Python 3.12).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
/opt/homebrew/bin/python3.12 -m venv .venv-mlx && .venv-mlx/bin/pip install -r requirements-mlx.txt
```

On Windows, use `.venv\Scripts\python` in place of `.venv/bin/python`; the MLX parts are macOS only.

## Data

```bash
.venv/bin/python -m quahog.scrape fetch all          # cached per source in data/raw/
.venv/bin/python -m quahog.scrape fetch Characters   # ...and the other categories
.venv/bin/python -m quahog.scrape build              # -> data/corpus.txt
.venv/bin/python -m quahog.bpe --merges 3840 --out data/tokenizer_4k.json
```

## Run

```bash
# Learning steps
.venv/bin/python -m tutorial.step1_bigram

# Pretrain and sample QuahogGPT
.venv/bin/python -m pretrain.train --help
.venv/bin/python -m pretrain.sample

# The assistant
.venv-mlx/bin/python -m qa.ask "Who is Peter Griffin?" "What's the capital of Australia?"

# The web app (then open http://127.0.0.1:8000)
.venv-mlx/bin/python -m web.server

# The from-scratch reader
.venv/bin/python -m qa.quahog_qa "When did Joe Swanson first appear?"

# Evaluations
.venv-mlx/bin/python -m evals.qa --out logs/eval/qa_final.jsonl
.venv-mlx/bin/python -m evals.guard --out logs/eval/guard_final.jsonl
.venv-mlx/bin/python -m evals.chat --out logs/eval/chat.jsonl
.venv/bin/python -m evals.qa --reader quahog --out logs/eval/qa_quahog_v8c.jsonl
```
