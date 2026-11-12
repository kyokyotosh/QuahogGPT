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

Latest commit: `d17c02d`. Uncommitted: v3 promotion, step 5 retrieval, and the `load_docs()` refactor in `step3_scrape.py`.

## Results

| model | params | best val loss | notes |
| --- | --- | --- | --- |
| v1 (`checkpoints/quahog_gpt.pt`) | 1.07M | 2.97 | 128-wide, 4 layers, 6000 steps |
| v2 (`checkpoints/quahog_gpt_v2.pt`) | 3.08M | 2.76 | 192-wide, 6 layers, 6000 steps, about 66 min on CPU |
| v3 (`checkpoints/quahog_gpt_v3.pt`) | 3.08M | 2.67 | same config as v2, 12000 steps, about 17 min on MPS |

v3 is the current model. Train 2.31 vs val 2.67 at step 12000. Val was still improving at the last checkpoint. `quahog_gpt_v2.pt` is kept for comparison, and `quahog_gpt_v2_6k.pt` is a backup copy of it.

Sampling (`sample.py`): temperature 0.8 with top-k 40 gives the most readable output. Temperature 1.0 is more random; temperature 0.5 to 0.6 loops on episode lists.

## Data

- `data/raw/*.json`: 14 category caches, 13 MB, 7312 unique pages, 4676 kept after cleaning.
- `data/corpus.txt`: 4.2 MB, 1,705,388 BPE tokens. Documents are shuffled before writing, so the 90/10 split is representative.
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

1. **Longer or larger run.** v3 was still improving at step 12000. The learning rate is constant (3e-3, no decay); decay is not tried yet.
2. **More data.** The train/val gap (2.31 vs 2.67) says the model is data-limited. Add wiki categories, after the Fandom license is checked.
3. **Retrieval.** `rag_sample.py` retrieves one passage every 16 tokens, with a 64-token passage budget (the window is 128 tokens). Measured on 283 held-out pages: plain loss 2.273, grounded 2.242, gain 0.031 ± 0.007 (s.e.). The first retrieved passages match the prompts. Ideas: a bigger passage budget or block size (needs retraining), reranking.
4. **Fine-tune (LoRA).** Not started. There is no pretrained base model.

## Known issues

- `data/corpus.txt` still has two stray lines from an unclosed `{{SongBox` template.
- `sample.py` runs only the two settings `(1.0, None)` and `(0.8, 40)`.
- Git warns about LF to CRLF on Windows. Harmless.
- `torch` on Windows prints a NumPy warning. Harmless. `pip install numpy` removes it.
- `data/corpus.txt` on this Mac does not reproduce from `data/raw` with the current build code. Same 4676 pages and 4,240,663 characters, different document order. Retrieval reads `data/raw` directly, so it is unaffected. v3 trained on the on-disk corpus.
