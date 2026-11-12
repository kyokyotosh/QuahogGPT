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

Latest commit: see `git log` (v7_long promotion, sampler-faithful eval).

## Results

| model | params | best val loss | notes |
| --- | --- | --- | --- |
| v1 (`checkpoints/quahog_gpt.pt`) | 1.07M | 2.97 | 128-wide, 4 layers, 6000 steps |
| v2 (`checkpoints/quahog_gpt_v2.pt`) | 3.08M | 2.76 | 192-wide, 6 layers, 6000 steps, about 66 min on CPU |
| v3 (`checkpoints/quahog_gpt_v3.pt`) | 3.08M | 2.67 | same config as v2, 12000 steps, constant LR, about 17 min on MPS |
| v4 (`checkpoints/quahog_gpt_v4.pt`) | 3.08M | 2.50 | same as v3 plus cosine LR decay (3e-3 to 3e-4), 12000 steps, about 16 min on MPS |
| v5 (`checkpoints/quahog_gpt_v5.pt`) | 3.08M | 2.37 | v4 config on the expanded corpus (5358 pages), 12000 steps, about 16 min on MPS |
| v7_wide (`checkpoints/quahog_gpt_v7_wide.pt`) | 5.29M | 2.32 | 256-wide, 8 heads, 12000 steps (`python3 step4_train.py --n-embd 256 --n-head 8 --ckpt checkpoints/quahog_gpt_v7_wide.pt`). Best at step 11250 |
| v7_long (`checkpoints/quahog_gpt_v7_long.pt`) | 3.08M | 2.28 | v5 config, 24000 steps (`python3 step4_train.py --max-iters 24000 --ckpt checkpoints/quahog_gpt_v7_long.pt`). Best at step 23500, still improving at the end. Current model |
| v6 (`checkpoints/quahog_gpt_v6.pt`) | 3.08M | 2.42 | v5 with dropout 0.3 and weight decay 0.2. Worse than v5 on the same-split eval. Config was an uncommitted edit to `step4_train.py`; the repo is back at v5's config |

v7_long is the current model: best val 2.28. On 434 pages unseen by v5, v7_wide, and v7_long: v5 2.364, v7_wide 2.296 (paired vs v5 0.068 ± 0.006), v7_long 2.271 (paired vs v5 0.093 ± 0.005). On the sampler's own query (433 pages: first 64 tokens as query, 16-token target, window as in `rag_sample.py`): v7_long without retrieval 2.396 vs v5 2.477 (paired 0.076 ± 0.013). Retrieval on v7_long: first-fit +0.004 ± 0.006, truncated +0.008 ± 0.008, neither significant. Retrieval on v5: first-fit +0.0085 ± 0.0046. Earlier: v4 vs v5 paired 0.227 ± 0.037 (127 pages); v5 vs v6 paired 0.062 ± 0.005 (434 pages). v3, v4, and `quahog_gpt_v2.pt` are kept for comparison, and `quahog_gpt_v2_6k.pt` is a backup copy of v2.

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
3. **Retrieval (updated).** Tests on v5, body 24 tokens, query = title. Scripts: `eval_first_fit.py` (truncated vs whole passage), `eval_whole_passages.py` (budgets), `eval_ablate_retrieval.py` (cut-point ablation), `eval_overlap.py` (duplicated text check).
   - The old `rag_sample.py` cut the top-1 passage to 64 tokens, mid-sentence. On the 394 pages with a whole passage that fits, the truncated top-1 gains +0.090 and the first-fitting whole passage gains +0.210. Paired difference +0.120 ± 0.011. Applied in `rag_sample.py` (`retrieve_ids`).
   - With whole passages, budgets 64, 96, and the window cap do not differ. Paired 96 − 64 = −0.005 ± 0.004 (n=214). Keep 64.
   - Ablation on rank-1 (n=404): cutting to 32 tokens gives −0.068; cutting to 64 gives +0.085 overall (−0.055 on passages over 64 tokens, +0.220 on short ones); cutting to 96 gives +0.170. So the gain depends on where the cut lands. The earlier 96-token gain was an artifact of truncation.
   - Duplicated text is not the main driver: only 4.5% of pages share an 8-word run with the retrieved text.
   - Sampler-query check (`eval_sampler_query.py`, rewritten to match `rag_sample.py` exactly: per-token windows, 64-token context, no 49-token cap). Decision: keep first-fit. Results: v5 first-fit +0.0085 ± 0.0046, v7_wide −0.0017 ± 0.0045, v7_long +0.0042 ± 0.0056; truncated: +0.0056 ± 0.0064, +0.0034 ± 0.0068, +0.0077 ± 0.0075. The variants are not separable on paired data. First-fit keeps whole passages (no mid-sentence cuts). That is a design choice, not a measured win. The earlier −0.013 result used the old 49-token cap and is superseded.
   - Open: sample quality. The check above mirrors the sampler's query and window, but it does not score sampled text.
4. **Fine-tune (LoRA).** Not started. There is no pretrained base model.

## Known issues

- `data/corpus.txt` still has one stray `{{` from an unclosed `{{SongBox` template.
- `sample.py` runs only the two settings `(1.0, None)` and `(0.8, 40)`.
- Git warns about LF to CRLF on Windows. Harmless.
- `torch` on Windows prints a NumPy warning. Harmless. `pip install numpy` removes it.
- The earlier retrieval eval mixed training and validation pages (283 pages, gain 0.031 for v3). The validation-region numbers above replace it.
- The old `data/corpus.txt` (4676 pages, kept at `../quahog_backups/corpus_v4_trained.txt`, outside the repo) does not reproduce from the current `data/raw`. It was built on another machine, so the document order differs. v3 and v4 were trained on it. v5 was trained on the rebuilt corpus.
