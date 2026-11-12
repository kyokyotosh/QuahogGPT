# QuahogGPT handoff

Where the project stands, so you can pick it up on the Mac.

## Status

| Step | File(s) | State |
| --- | --- | --- |
| 1. Bigram model | `step1_bigram.py` | done (`464b1d2`) |
| 2. Tiny GPT | `step2_gpt.py` | done (`9b5867a`) |
| 3. Scrape + BPE | `step3_scrape.py`, `step3_bpe.py` | done (`7ce6c00`, `ff97738`, `8c18201`, `29e6f14`) |
| 4. Train on corpus | `gpt.py`, `step4_train.py`, `sample.py` | done (`da95e5c`, `29e6f14`, `d17c02d`) |
| 5. Retrieval | `retrieve.py`, `rag_sample.py` | done (BM25 over wiki passages); LoRA fine-tune not started |

Latest commit: see `git log` (v5 promotion and expanded scrape).

## Results

| model | params | best val loss | notes |
| --- | --- | --- | --- |
| v1 (`checkpoints/quahog_gpt.pt`) | 1.07M | 2.97 | 128-wide, 4 layers, 6000 steps |
| v2 (`checkpoints/quahog_gpt_v2.pt`) | 3.08M | 2.76 | 192-wide, 6 layers, 6000 steps, about 66 min on CPU |
| v3 (`checkpoints/quahog_gpt_v3.pt`) | 3.08M | 2.67 | same config as v2, 12000 steps, constant LR, about 17 min on MPS |
| v4 (`checkpoints/quahog_gpt_v4.pt`) | 3.08M | 2.50 | same as v3 plus cosine LR decay (3e-3 to 3e-4), 12000 steps, about 16 min on MPS |
| v5 (`checkpoints/quahog_gpt_v5.pt`) | 3.08M | 2.37 | v4 config on the expanded corpus (5358 pages), 12000 steps, about 16 min on MPS. Current model |
| v6 (`checkpoints/quahog_gpt_v6.pt`) | 3.08M | 2.42 | v5 with dropout 0.3 and weight decay 0.2. Worse than v5 on the same-split eval. Config was an uncommitted edit to `step4_train.py`; the repo is back at v5's config |

v5 is the current model: best val 2.37 (step 11250 of 12000). Train 2.06 vs val 2.37. v4 and v5 val numbers are on different splits. On 127 pages unseen by v4 and v5, v5 scores 2.275 and v4 scores 2.503 on the first 128 tokens (paired difference 0.227 ± 0.037). On 434 pages unseen by v5 and v6, v5 scores 2.364 and v6 scores 2.426 (paired difference 0.062 ± 0.005). v3, v4, and `quahog_gpt_v2.pt` are kept for comparison, and `quahog_gpt_v2_6k.pt` is a backup copy of v2.

Sampling (`sample.py`): temperature 0.8 with top-k 40 gives the most readable output. Temperature 1.0 is more random; temperature 0.5 to 0.6 loops on episode lists.

## Data

- `data/raw/*.json`: 15 caches. `all.json` holds all 8081 main-namespace articles, and the 14 category caches overlap it. 8114 unique pages, 5358 kept after cleaning (200-character minimum).
- `data/corpus.txt`: 5.0 MB. Token count is printed at the top of `logs_v5_train.txt`. Documents are shuffled before writing, so the 90/10 split is representative.
- `data/tokenizer.json`: 768 merges, vocab 1024.
- The corpus includes real-world people who appear on the show (cast, guest stars, writers). This is intentional.
- The Fandom wiki license is still unverified. Check it before publishing the corpus.

All of this is gitignored. A fresh clone does not have it. Copy `data/` and `checkpoints/` from Windows, or re-run the scraper (`fetch` for each category, then `build`).

## Environment

- Windows: `.venv` with Python 3.13, `torch 2.15.0+cpu`. The RX 6700 XT is not usable for PyTorch here (HIP SDK unsupported on Windows).
- Mac: M3 Pro. Use the MPS backend. Create a venv with `python3 -m venv .venv`, then `pip install -r requirements.txt`.

## Mac setup

```bash
git clone https://github.com/kyokyotosh/QuahogGPT.git
cd QuahogGPT
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# copy data/ and checkpoints/ from the Windows machine, or rebuild:
python step3_scrape.py fetch all
python step3_scrape.py fetch Characters
python step3_scrape.py fetch Episodes
# ...and the other categories listed in README.md
python step3_scrape.py build
python step3_bpe.py
```

## Next up

1. **Same-split comparison (done).** Result above. Script: `eval_same_split.py` (committed). Run it with `--old-corpus` when a checkpoint was trained on an older corpus.
2. **Regularization (done, v6).** Dropout 0.3 and weight decay 0.2 made val worse (2.42 vs 2.37), so v5 stays. The overfitting gap is not fixed by more regularization of this kind.
3. **Retrieval.** `rag_sample.py` retrieves one passage every 16 tokens, with a 64-token passage budget (the window is 128 tokens). On 434 pages unseen by v5 and v6, the gain is clearly positive: v5 0.033 ± 0.006, v6 0.038 ± 0.006 (s.e.). An earlier claim that it was not significant came from a 125-page set and was underpowered. Ideas: a bigger passage budget or block size (needs retraining), reranking.
4. **Fine-tune (LoRA).** Not started. There is no pretrained base model.

## Known issues

- `data/corpus.txt` still has one stray `{{` from an unclosed `{{SongBox` template.
- `sample.py` runs only the two settings `(1.0, None)` and `(0.8, 40)`.
- Git warns about LF to CRLF on Windows. Harmless.
- `torch` on Windows prints a NumPy warning. Harmless. `pip install numpy` removes it.
- The earlier retrieval eval mixed training and validation pages (283 pages, gain 0.031 for v3). The validation-region numbers above replace it.
- The old `data/corpus.txt` (4676 pages, kept at `../quahog_backups/corpus_v4_trained.txt`, outside the repo) does not reproduce from the current `data/raw`. It was built on another machine, so the document order differs. v3 and v4 were trained on it. v5 was trained on the rebuilt corpus.
