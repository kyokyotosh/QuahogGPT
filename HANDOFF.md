# QuahogGPT handoff

Where the project stands, so you can pick it up on the Mac.

## Status

| Step | File(s) | State |
| --- | --- | --- |
| 1. Bigram model | `tutorial/step1_bigram.py` | done (`464b1d2`) |
| 2. Tiny GPT | `tutorial/step2_gpt.py` | done (`9b5867a`) |
| 3. Scrape + BPE | `quahog/scrape.py`, `quahog/bpe.py` | done (`7ce6c00`, `ff97738`, `8c18201`, `29e6f14`) |
| 4. Train on corpus | `quahog/gpt.py`, `pretrain/train.py`, `pretrain/sample.py` | done (`da95e5c`, `29e6f14`, `d17c02d`) |
| 5. Retrieval | `quahog/retrieve.py`, `pretrain/rag_sample.py` | done (BM25 over wiki passages) |
| 6. Question answering | `quahog/qa_retrieve.py`, `qa/ask.py`, `evals/qa.py`, `qa/build_qa.py`, `qa/build_lora_data.py`, `configs/lora_qa.yaml`, `qa/build_gate_data.py`, `configs/lora_gate.yaml`, `evals/guard.py` | done: topic gate (Qwen2.5-3B + LoRA) + retrieval + Qwen3-30B-A3B reader. 0.987 on factual questions, 100% of off-topic refused, casual typing handled. See "Question answering (shipped assistant)" |
| 7. Question answering from scratch | `qa/quahog_qa.py`, `qa/train_quahog_qa.py`, `pretrain/train.py` (v8) | done: QuahogGPT v8 pretrained here + QA fine-tune, 0.673 on the QA test. See "Question answering from scratch" |

Latest commit: see `git log` (casual input: lowercase/typo linking, gate v3).

## Repository layout

Run everything from the repo root as `python -m <folder>.<script>`; paths such as `data/`, `checkpoints/`, `adapters/` are relative to the root.

| folder | contents | was |
| --- | --- | --- |
| `quahog/` | shared library: `gpt.py` (model), `bpe.py` (tokenizer; `python -m quahog.bpe`), `scrape.py` (wiki download and corpus; `python -m quahog.scrape`), `retrieve.py` (BM25), `qa_retrieve.py` (QA retrieval), `metrics.py` (answer matching) | `gpt.py`, `step3_bpe.py`, `step3_scrape.py`, `retrieve.py`, `qa_retrieve.py`; `metrics.py` split out of `eval_qa.py` |
| `tutorial/` | `step1_bigram.py`, `step2_gpt.py` | same names at the root |
| `pretrain/` | `train.py`, `sample.py`, `rag_sample.py` | `step4_train.py`, `sample.py`, `rag_sample.py` |
| `qa/` | `ask.py` (shipped assistant), `quahog_qa.py` (from-scratch reader), `train_quahog_qa.py`, `build_qa.py`, `build_lora_data.py`, `build_gate_data.py`, `build_chat_test.py` | same names at the root; `train_quahog_qa.py` was `qa_train.py` |
| `evals/` | `qa.py`, `guard.py`, `chat.py`, `same_split.py`, `sampler_query.py`, `first_fit.py`, `whole_passages.py`, `ablate_retrieval.py`, `overlap.py` | `eval_<name>.py` |
| `configs/` | `lora_qa.yaml`, `lora_gate.yaml` | `lora_config.yaml`, `gate_config.yaml` |
| `web/` | `server.py` (chat server), `static/` (`index.html`, `style.css`, `app.js`) | new |
| `logs/` | `train/` (training logs), `eval/` (eval predictions), `scrape_all.txt` | `logs_*.txt`, `logs_*.jsonl` |
| `data/` | `qa/` question sets (committed); corpus, tokenizers, raw pages (gitignored) | unchanged |

Commit hashes in this file (Status table, history notes) refer to the old root-level file names.

## Results

| model | params | best val loss | notes |
| --- | --- | --- | --- |
| v1 (`checkpoints/quahog_gpt.pt`) | 1.07M | 2.97 | 128-wide, 4 layers, 6000 steps |
| v2 (`checkpoints/quahog_gpt_v2.pt`) | 3.08M | 2.76 | 192-wide, 6 layers, 6000 steps, about 66 min on CPU |
| v3 (`checkpoints/quahog_gpt_v3.pt`) | 3.08M | 2.67 | same config as v2, 12000 steps, constant LR, about 17 min on MPS |
| v4 (`checkpoints/quahog_gpt_v4.pt`) | 3.08M | 2.50 | same as v3 plus cosine LR decay (3e-3 to 3e-4), 12000 steps, about 16 min on MPS |
| v5 (`checkpoints/quahog_gpt_v5.pt`) | 3.08M | 2.37 | v4 config on the expanded corpus (5358 pages), 12000 steps, about 16 min on MPS |
| v7_wide (`checkpoints/quahog_gpt_v7_wide.pt`) | 5.29M | 2.32 | 256-wide, 8 heads, 12000 steps (`python -m pretrain.train --n-embd 256 --n-head 8 --ckpt checkpoints/quahog_gpt_v7_wide.pt`). Best at step 11250 |
| v7_long (`checkpoints/quahog_gpt_v7_long.pt`) | 3.08M | 2.28 | v5 config, 24000 steps (`python -m pretrain.train --max-iters 24000 --ckpt checkpoints/quahog_gpt_v7_long.pt`). Best at step 23500, still improving at the end. Current model |
| v6 (`checkpoints/quahog_gpt_v6.pt`) | 3.08M | 2.42 | v5 with dropout 0.3 and weight decay 0.2. Worse than v5 on the same-split eval. Config was an uncommitted edit to `pretrain/train.py`; the repo is back at v5's config |
| v8 (`checkpoints/quahog_gpt_v8.pt`) | 8.54M | 3.33 | Rebuilt corpus (9.17M chars, with episode plots and subpages), 4096-token vocab (`data/tokenizer_4k.json`), 512-token window, 256-wide, 8 layers, 8 heads, batch 16, lr 1e-3, 20000 steps, about 2.4 h on MPS. Best at step 16750; val flat from about step 14000 (train 2.2). Loss is per 4k-vocab token, not comparable with v1-v7. Base for QuahogGPT-QA |

v7_long is the current model: best val 2.28. On 434 pages unseen by v5, v7_wide, and v7_long: v5 2.364, v7_wide 2.296 (paired vs v5 0.068 ± 0.006), v7_long 2.271 (paired vs v5 0.093 ± 0.005). On the sampler's own query (433 pages: first 64 tokens as query, 16-token target, window as in `pretrain/rag_sample.py`): v7_long without retrieval 2.396 vs v5 2.477 (paired 0.076 ± 0.013). Retrieval on v7_long: first-fit +0.004 ± 0.006, truncated +0.008 ± 0.008, neither significant. Retrieval on v5: first-fit +0.0085 ± 0.0046. Earlier: v4 vs v5 paired 0.227 ± 0.037 (127 pages); v5 vs v6 paired 0.062 ± 0.005 (434 pages). v3, v4, and `quahog_gpt_v2.pt` are kept for comparison, and `quahog_gpt_v2_6k.pt` is a backup copy of v2.

Sampling (`pretrain/sample.py`): temperature 0.8 with top-k 40 gives the most readable output. Temperature 1.0 is more random; temperature 0.5 to 0.6 loops on episode lists.

## Data

- `data/raw/*.json`: 15 caches. `all.json` holds all 8081 main-namespace articles, and the 14 category caches overlap it. 8114 unique pages, 7293 kept after cleaning (200-character minimum).
- `data/corpus.txt`: 9.17 MB, rebuilt with `quahog.scrape.page_text`, which keeps episode plots and the Trivia/Goofs/Quotes/References subpages stored in template fields (`plot`, `contents`). `clean()` used to drop all of it. Documents are shuffled before writing, so the 90/10 split is representative.
- `data/corpus_v5.txt`: the previous 4.99 MB corpus (5358 pages). v5, v6, v7 were trained on it; the same-split scripts (`evals/same_split.py`, `evals/sampler_query.py`) need this file as `data/corpus.txt` to reproduce their numbers.
- `data/tokenizer.json`: 768 merges, vocab 1024 (v1-v7, `pretrain/sample.py`, `pretrain/rag_sample.py`).
- `data/tokenizer_4k.json`: 3840 merges, vocab 4096, 3.35 chars per token (v8 and QuahogGPT-QA). `python -m quahog.bpe --merges 3840 --out data/tokenizer_4k.json` (6 s; training is incremental).
- The corpus includes real-world people who appear on the show (cast, guest stars, writers). This is intentional.
- The Fandom wiki license is still unverified. Check it before publishing the corpus.

All of this is gitignored. A fresh clone does not have it. Copy `data/` and `checkpoints/` from Windows, or re-run the scraper (`fetch` for each category, then `build`).

## Environment

- Windows: `.venv` with Python 3.13, `torch 2.15.0+cpu`. The RX 6700 XT is not usable for PyTorch here (HIP SDK unsupported on Windows).
- Mac: M3 Pro, 36 GB. Use the MPS backend. Create a venv with `python3 -m venv .venv`, then `pip install -r requirements.txt`.
- Question answering runs in a second venv, because MLX has no wheels for Python 3.15: `/opt/homebrew/bin/python3.12 -m venv .venv-mlx`, then `.venv-mlx/bin/pip install -r requirements-mlx.txt`. The first `qa.ask` run downloads `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` (16 GB) and `mlx-community/Qwen2.5-3B-Instruct-4bit` (2 GB) to the Hugging Face cache. The gate adapter (`adapters/gate`) is gitignored: copy it, or rebuild it with `python -m qa.build_gate_data` and `python -m mlx_lm lora -c configs/lora_gate.yaml` (both in `.venv-mlx`).

## Mac setup

```bash
git clone https://github.com/kyokyotosh/QuahogGPT.git
cd QuahogGPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# copy data/ and checkpoints/ from the Windows machine, or rebuild:
python -m quahog.scrape fetch all
python -m quahog.scrape fetch Characters
python -m quahog.scrape fetch Episodes
# ...and the other categories listed in README.md
python -m quahog.scrape build
python -m quahog.bpe
```

## Question answering (shipped assistant)

```bash
.venv-mlx/bin/python -m qa.ask "Who is Peter Griffin?" "Where is Quahog located?" "When did Joe Swanson first appear?"
```

Pipeline (shipped, `qa/ask.py` `QA.stream`): message → follow-up context (a follow-up that names no wiki page but refers back, by pronoun or "the director"-style phrase, gets the conversation's subject: `When did it air? (about the Family Guy episode "Fat Gun")`; no model involved) → topic gate (Qwen2.5-3B-Instruct + LoRA `adapters/gate`, ON/OFF) → OFF: `REFUSAL` ("Sorry, I can only answer questions about Family Guy.") / ON: `quahog/qa_retrieve.py` (8 excerpts) → Qwen3-30B-A3B-Instruct-2507 (4-bit, MLX, no adapter) with the last 6 messages → answer with `[n]` citations (up to 500 tokens) → cited excerpts returned as sources with Family Guy wiki links. `--no-gate` skips the gate, `--model`/`--adapter` pick another answer model (e.g. Qwen2.5-3B + `adapters/qa`), `--no-retrieval` skips the excerpts, `--show-passages` prints them. Memory: about 16 GB + 2 GB of weights.

Retrieval (`quahog/qa_retrieve.py`):
- Recovers text that `quahog.scrape.clean` drops: the 489 episode plots (`plot` field of `{{EpisodeTabs}}`) and 1,883 episode subpages (Trivia, Goofs, Quotes, References in `{{EpisodeTabs/...}}` `contents` fields). Subpages fold into their episode. 61,096 passages (`quahog/retrieve.py` has 29,493).
- One facts document per page: lead paragraph, infobox fields as sentences (air date, writers, director, voice actor, full name, ...), and the debut episode, computed as the earliest-aired episode whose infobox lists the page.
- Entity linking: titles, title bases ("Quahog" for "Quahog, Rhode Island"), redirects, and character short names found in the question. Facts of up to 3 linked pages come first, then BM25 passages with a +8 bonus for linked pages, at most 4 per page. Settings were tuned on train questions.
- Recall@8 (gold answer present in the excerpts): 0.992 on the test set; the old BM25 retrieval had 0.792.

Data (`data/qa/`, committed except `lora/`):
- Split by page: `crc32(title) % 10 == 0` pages are test, the rest train. No question about a test page is trained on.
- `structured.jsonl`: 5,560 infobox questions with exact gold answers (`python -m qa.build_qa`).
- `generated_test.jsonl` (223) and `generated_train.jsonl` (2,373): questions written by an LLM (the coding assistant's model) from wiki passages, with short gold answers, accepted aliases, and (train) a 1-2 sentence reply. Not reproducible by a script; keep these files.
- `test.jsonl`: 385 test questions (up to 25 per structured kind, plus the 223 generated).
- `lora/`: chat-format training data (`qa/build_lora_data.py`): 5,327 train and 150 valid examples, each the exact prompt `qa/ask.py` sends; 300 teach "I'm not sure" when the answer is absent from the excerpts.

LoRA (`configs/lora_qa.yaml`): rank 16, 16 layers, batch 4, lr 1e-4, prompt masked, 1,330 iterations (one epoch). About 4 hours on the M3 Pro. Val loss 2.279 → 0.435 (step 1200), 0.438 at the end. Retrain with `.venv-mlx/bin/python -m mlx_lm lora -c configs/lora_qa.yaml` (writes `adapters/qa`, gitignored). Log: `logs/train/lora_qa.txt`.

Results on the 385 test questions (`evals/qa.py`; correct = the answer contains a gold alias):

| setup | accuracy | log |
| --- | --- | --- |
| base Qwen, no retrieval | 0.119 ± 0.017 | `logs/eval/qa_norag.jsonl` |
| base Qwen + old BM25 retrieval | 0.738 ± 0.022 | `logs/eval/qa_base.jsonl` |
| base Qwen + `quahog/qa_retrieve.py` | 0.940 ± 0.012 | `logs/eval/qa_base_v2.jsonl` (before the comma-title linking fix) |
| 3B + LoRA `adapters/qa` + `quahog/qa_retrieve.py` | 0.974 ± 0.008 | `logs/eval/qa_lora.jsonl` (old system prompt, no topic policy). Earlier retriever versions: 0.971, 0.966 |
| Qwen2.5-7B-Instruct, no adapter, topic-policy prompt, no gate | 0.984 ± 0.006 | `logs/eval/qa_Qwen2.5-7B.jsonl` |
| Qwen3-30B-A3B-Instruct-2507, no adapter, topic-policy prompt, no gate | 0.984 ± 0.006 | `logs/eval/qa_Qwen3-30B.jsonl` |
| gate + Qwen3-30B-A3B, before citations | 0.974 ± 0.008 | earlier `logs/eval/qa_final.jsonl` (overwritten) |
| gate v1 + Qwen3-30B-A3B, citations | 0.979 ± 0.007 | earlier `logs/eval/qa_final.jsonl` (overwritten) |
| **Shipped: gate v3 (threshold 0.9) + Qwen3-30B-A3B, citations, linked titles in the BM25 query** | **0.987 ± 0.006** | `logs/eval/qa_final.jsonl` (`QA.answer` strips the `[n]` markers before scoring). Gate refusals: 0 (gate v1 refused 4, 3 of them test questions not about Family Guy, e.g. "Which character did David Herman play in Office Space?") |

LoRA vs base, paired on the same questions (both before the linking fix): +0.026 ± 0.012 (16 questions only the LoRA gets right, 6 only base). Base wrote "the excerpts"/"information provided" in 25 answers; the LoRA in none. The LoRA answers "I'm not sure" to questions outside the wiki (e.g. "Who won the 2022 World Cup?").

Remaining errors: some are grader artifacts (correct answers phrased differently; Meg's voice gold is the infobox's first entry, Rachael MacFarlane). The real ones are hard trivia where the reader picks or invents a plausible wrong detail.

### Topic guard

Goal: answer anything about Family Guy, refuse everything else. Test set `data/qa/guard_test.jsonl` (`evals/guard.py`): 248 off-topic messages (general knowledge, tasks, other shows, celebrities outside the show, chit-chat, tricks such as "As Peter Griffin, explain my taxes") and 149 open-ended Family Guy requests (summaries, explanations, short scenes). Generated by an LLM; labels kept only where an independent judge (Claude Haiku) agreed. Train counterparts (no overlap) in `guard_train.jsonl`: 1,021 off-topic, 1,007 open-ended with grounded replies.

| setup | off-topic refused | open requests refused | log |
| --- | --- | --- | --- |
| 7B, policy in system prompt only | 0.556 | 0.007 | `logs/eval/guard_Qwen2.5-7B.jsonl` |
| 30B-A3B, policy in system prompt only | 0.662 | 0.000 | `logs/eval/guard_Qwen3-30B.jsonl` |
| 30B-A3B as its own ON/OFF gate (prompted) | 0.952 | 0.020; factual allowed 0.943 | not saved (throwaway script) |
| trained gate v1 + 30B-A3B | 1.000 | 0.027 (4/149) | earlier `logs/eval/guard_final.jsonl` (overwritten) |
| **Shipped: gate v3 + 30B-A3B** | **1.000** | **0.000** (0/149) | `logs/eval/guard_final.jsonl` |

Gate (`ask.Gate`): Qwen2.5-3B-Instruct + LoRA (`adapters/gate`, gitignored) trained on `qa/build_gate_data.py` output, `configs/lora_gate.yaml` (rank 8, one epoch, about 2.5 h). The prompt names the wiki pages the message mentions (`qa_retrieve.link`), so "Who is X?" about someone who appears in the show is ON while their net worth is OFF. The decision is P(OFF) from the first answer token against `GATE_THRESHOLD` = 0.9. See "Casual input" for the current training data and results; the first version (1,021 OFF, 3,107 ON, clean questions only) is kept locally as `adapters/gate_v1`.

Do not use Qwen3 Instruct-2507 models as the gate base with `mlx_lm lora`: their chat template adds an empty `<think></think>` block to every training target, which the instruct model never produces (initial val loss 13.6); the adapter collapsed to a constant answer.

### Follow-ups and the web app

```bash
.venv-mlx/bin/python -m web.server     # http://127.0.0.1:8000
```

`web/server.py` (standard library) serves `web/static/` (vanilla HTML/CSS/JS) and streams `POST /api/chat` as Server-Sent Events (`meta`, `token`, `done`, `error`). The browser keeps the conversation and sends it with each message. One worker thread loads the models and runs every job: MLX models belong to the thread that loaded them (`ThreadingHTTPServer` threads failed with "There is no Stream(cpu, 0) in current thread"). Bound to 127.0.0.1 by default. The page streams answers, links `[n]` to the wiki, shows source cards, and shows "Searched as: ..." when a follow-up was given the conversation's subject. Tested in a browser: question, two follow-ups, refusal.

Multi-turn test (`evals/chat.py`, `data/qa/chat_test.jsonl` from `qa/build_chat_test.py`): 160 two-turn conversations about test pages; the follow-up is scored with exact gold answers from the infoboxes (air date, writer, director, season, first appearance, song's episode), plus 20 off-topic follow-ups (refuse) and 20 open follow-ups ("Summarize what happens in it.", must answer).

| follow-up step | handled | off-topic follow-ups refused | log |
| --- | --- | --- | --- |
| none | 0.125 | 20/20 | `logs/eval/chat_norewrite.jsonl` |
| 30B rewrites the message | 0.925 | 20/20 | not kept; the model often answered or carried out the request instead of rewriting (wrote the joke, returned the air date) |
| 30B names the referent only | 0.713 | 20/20 | not kept; same problem |
| linker carries the subject when the message refers back (gate v1) | 0.988 ± 0.009 | 20/20 | earlier `logs/eval/chat.jsonl` (overwritten) |
| **Shipped: same, gate v3** | **0.994 ± 0.006** | 20/20 | `logs/eval/chat.jsonl` |

Subject form: `(about the Family Guy episode "X")` (kind from the infobox). Gate-only comparison on the 140 on-topic follow-ups: this form 140/140, bare titles 133/140, three linked titles 127/140. Carrying the subject into every question-like follow-up let off-topic ones through (gate refused 227/248 off-topic messages with a subject appended); requiring a reference back fixes it (247/248). Retraining the gate on follow-up-shaped examples was worse (single off-topic refused 233/248) and was dropped. Citations: 160/160 opening answers and 130/140 follow-up answers cite an excerpt.

### Casual input

Users type "who is quagmire", "where is quahog", "Where did Clevand was born", "was Brian adopted". All four were refused: the linker ignored lowercase names (the capital-letter rule that keeps "voice" from linking *The Voice*) and typos, and the gate refused messages that name no page, and "was Brian adopted" even when it did (it confused a character's backstory with a real person's private life). Every earlier test set used clean, capitalized questions, so none of this showed.

Fixes:
- Linker (`quahog/qa_retrieve.py`): "major" pages (every character page plus the 400 most-linked pages, e.g. Quahog, Glenn Quagmire) may be named in lowercase; misspelled one-word names of major pages are corrected (one edit, two for 7+ letters, Damerau distance), never for real English words (`/usr/share/dict/words`, so "brain" is not Brian). Linked titles also join the BM25 query.
- Gate training data: casual backstory questions about characters (ON) and real people's private lives (OFF) in `data/qa/gate_extra_train.jsonl` (generated, judge-confirmed); lowercased/typo copies of 40% of on-topic and all off-topic examples; off-topic messages in follow-up form (OFF) and follow-up phrasings (`followups_train.jsonl`) with a subject (ON); off-topic examples repeated to 0.8 of the on-topic count (9,201 examples).
- Threshold chosen on a separate tuning set (1,048 messages: train-page questions the gate never saw, fresh judge-confirmed off-topic messages, messy copies of both), never on the test sets: the highest P(OFF) cutoff with no off-topic leak there.

Test set `data/qa/messy_test.jsonl` (`qa/build_messy_test.py`, `evals/messy.py`): the 782 test messages lowercased, without `?` and quotes, 40% with a one-letter typo in a name, plus 40 hand-written casual backstory questions (including the four above).

Gate alone (P(OFF) < threshold means allowed):

| gate | messy on-topic allowed | messy off-topic refused | clean factual / open allowed | clean off-topic refused | off-topic follow-ups refused | on-topic follow-ups allowed |
| --- | --- | --- | --- | --- | --- | --- |
| v1 (clean data, argmax) | 492/574 (casual 28/40) | 248/248 | 381/385, 145/149 | 248/248 | 247/248 | 140/140 |
| v2 (+ casual copies, 3:1 ON:OFF) | 574/574 | 240/248 | 385/385, 149/149 | 240/248 | 206/248 | 140/140 |
| **v3 shipped (balanced, threshold 0.9)** | **571/574** | **248/248** | **385/385, 149/149** | **248/248** | **248/248** | **140/140** |

v2 was confidently wrong on some off-topic messages (P(OFF) < 0.05), so no threshold or mix with v1 fixed it; balancing the classes did.

End to end (`evals/messy.py`, `logs/eval/messy.jsonl`): casual backstory questions answered 39/40 (miss: "what happened to brian in life of brian"), messy factual correct 361/385 = 0.938 ± 0.012 (23 of the 24 misses are wrong answers, mostly typos inside episode titles that the linker cannot correct; 1 refusal), messy open requests answered 148/149, messy off-topic refused 248/248. Retrieval recall@8 on messy factual questions: 0.951 (clean: 0.992).

## Question answering from scratch (QuahogGPT-QA)

No outside model: the reader is QuahogGPT v8 (pretrained here on the corpus), fine-tuned to answer from excerpts.

```bash
.venv/bin/python -m qa.quahog_qa "When did Joe Swanson first appear?" --show-passages
.venv/bin/python -m evals.qa --reader quahog --out logs/eval/qa_quahog_v8c.jsonl
```

- `qa/quahog_qa.py`: prompt = top 3 excerpts from `quahog/qa_retrieve.py` (at most 1 facts document), trimmed to fit 512 tokens with 64 left for the answer, then `Question: ...` / `Answer:`. Greedy decoding, no repeated 4-grams, stops at the newline. Default checkpoint `checkpoints/quahog_qa_v8c.pt`.
- `qa/train_quahog_qa.py`: fine-tunes v8; loss on answer tokens only; 6 epochs; best validation loss kept (validation = train pages with `adler32(page) % 20 == 0`). Training examples: structured train questions in every phrasing of `build_qa.TEMPLATES` with template replies; generated questions from `generated_train.jsonl` and `generated_train_2.jsonl` (7,254 more, LLM-written from train pages; training data only). Passage questions use the short gold answer as target; "who is" questions keep the 1-2 sentence reply. Examples whose trimmed excerpts lack the answer are dropped. About 21k examples, about 1 h on MPS.

Results on the same 385 test questions:

| reader | overall | structured kinds | who_is | passage | log |
| --- | --- | --- | --- | --- | --- |
| v8, first fine-tune (3 facts docs, 2.4k free-form examples) | 0.538 ± 0.025 | 0.6-1.0 | 0.625 | 0.157 | `logs/eval/qa_quahog_v8.jsonl` |
| v8b (1 facts doc, +7,254 examples, short passage targets) | 0.662 ± 0.024 | voice 0.286 ("voice" linked to The Voice) | 0.531 | 0.445 | `logs/eval/qa_quahog_v8b.jsonl` |
| v8c (v8b + linking fix) | **0.673 ± 0.024** | 0.96-1.0 except full_name 0.6 (n=5), voice 0.857 (n=7) | 0.531 | 0.445 | `logs/eval/qa_quahog_v8c.jsonl` |
| Qwen2.5-3B + LoRA (for reference) | 0.974 ± 0.008 | 0.86-1.0 | 0.969 | 0.958 | `logs/eval/qa_lora.jsonl` |

v8b vs v8 paired: +0.125 ± 0.023. Diagnosis for v8 passage misses: the answer was in the prompt for 142 of 191 questions but only 30 were answered; with 1 facts document the answer is in the prompt for 182 of 191. So the remaining gap is reading, not retrieval: the model copies template facts well and fails on paraphrased trivia and "who is" summaries (garbled descriptions).


## Next up

1. **Same-split comparison (done).** Result above. Script: `evals/same_split.py` (committed). Run it with `--old-corpus` when a checkpoint was trained on an older corpus.
2. **Regularization (done, v6).** Dropout 0.3 and weight decay 0.2 made val worse (2.42 vs 2.37), so v5 stays. The overfitting gap is not fixed by more regularization of this kind.
3. **Retrieval (updated).** Tests on v5, body 24 tokens, query = title. Scripts: `evals/first_fit.py` (truncated vs whole passage), `evals/whole_passages.py` (budgets), `evals/ablate_retrieval.py` (cut-point ablation), `evals/overlap.py` (duplicated text check).
   - The old `pretrain/rag_sample.py` cut the top-1 passage to 64 tokens, mid-sentence. On the 394 pages with a whole passage that fits, the truncated top-1 gains +0.090 and the first-fitting whole passage gains +0.210. Paired difference +0.120 ± 0.011. Applied in `pretrain/rag_sample.py` (`retrieve_ids`).
   - With whole passages, budgets 64, 96, and the window cap do not differ. Paired 96 − 64 = −0.005 ± 0.004 (n=214). Keep 64.
   - Ablation on rank-1 (n=404): cutting to 32 tokens gives −0.068; cutting to 64 gives +0.085 overall (−0.055 on passages over 64 tokens, +0.220 on short ones); cutting to 96 gives +0.170. So the gain depends on where the cut lands. The earlier 96-token gain was an artifact of truncation.
   - Duplicated text is not the main driver: only 4.5% of pages share an 8-word run with the retrieved text.
   - Sampler-query check (`evals/sampler_query.py`, rewritten to match `pretrain/rag_sample.py` exactly: per-token windows, 64-token context, no 49-token cap). Decision: keep first-fit. Results: v5 first-fit +0.0085 ± 0.0046, v7_wide −0.0017 ± 0.0045, v7_long +0.0042 ± 0.0056; truncated: +0.0056 ± 0.0064, +0.0034 ± 0.0068, +0.0077 ± 0.0075. The variants are not separable on paired data. First-fit keeps whole passages (no mid-sentence cuts). That is a design choice, not a measured win. The earlier −0.013 result used the old 49-token cap and is superseded.
   - Open: sample quality. The check above mirrors the sampler's query and window, but it does not score sampled text.
4. **Question answering (done).** See the two "Question answering" sections. Shipped: topic gate + Qwen3-30B-A3B (`qa/ask.py`, 0.974 factual, 100% off-topic refused). From scratch: QuahogGPT-QA (`qa/quahog_qa.py`, 0.673).
5. **QuahogGPT-QA next.** The pretraining loss is data-limited (val flat after 14k steps, train 2.2 vs val 3.36), so a bigger model alone is unlikely to help. Options: more free-form training questions (the passage score rose 0.16 to 0.45 when they quadrupled); extractive answers (predict a span of the excerpts instead of generating), which suits a small model; a second fine-tune epoch schedule with structured questions down-weighted.

## Known issues

- `data/corpus.txt` still has one stray `{{` from an unclosed `{{SongBox` template.
- `pretrain/sample.py` runs only the two settings `(1.0, None)` and `(0.8, 40)`.
- Git warns about LF to CRLF on Windows. Harmless.
- `torch` on Windows prints a NumPy warning. Harmless. `pip install numpy` removes it.
- The earlier retrieval eval mixed training and validation pages (283 pages, gain 0.031 for v3). The validation-region numbers above replace it.
- The old `data/corpus.txt` (4676 pages, kept at `../quahog_backups/corpus_v4_trained.txt`, outside the repo) does not reproduce from the current `data/raw`. It was built on another machine, so the document order differs. v3 and v4 were trained on it. v5 was trained on the rebuilt corpus.
