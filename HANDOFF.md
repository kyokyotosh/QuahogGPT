# QuahogGPT handoff

Where the project stands, so you can pick it up on the Mac.

## Status

| Step | File(s) | State |
| --- | --- | --- |
| 1. Bigram model | `step1_bigram.py` | done (`464b1d2`) |
| 2. Tiny GPT | `step2_gpt.py` | done (`9b5867a`) |
| 3. Scrape + BPE | `step3_scrape.py`, `step3_bpe.py` | done (`7ce6c00`, `ff97738`, `8c18201`, `29e6f14`) |
| 4. Train on corpus | `gpt.py`, `step4_train.py`, `sample.py` | done (`da95e5c`, `29e6f14`, `d17c02d`) |
| 5. Fine-tune + retrieval | none | not started |

Latest commit: `d17c02d`.

## Results

| model | params | best val loss | notes |
| --- | --- | --- | --- |
| v1 (`checkpoints/quahog_gpt.pt`) | 1.07M | 2.97 | 128-wide, 4 layers, 6000 steps |
| v2 (`checkpoints/quahog_gpt_v2.pt`) | 3.08M | 2.76 | 192-wide, 6 layers, 6000 steps, about 66 min on CPU |

v2 is the current model. Train 2.42 vs val 2.76, so overfitting has started but val was still falling at the end.

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

1. **Device change (not written yet).** `step4_train.py` and `sample.py` run everything on CPU. Add `device = "mps" if torch.backends.mps.is_available() else "cpu"`, move the model and batches to `device`, and use `map_location=device` on `torch.load`.
2. **Timing test on the Mac.** Run about 50 steps, compare with the CPU rate (v2 took about 40 s per 250 steps on Windows).
3. **Decide the next experiment.** Options: a longer or larger run (v2 was still improving), more data (more wiki categories), or step 5 (retrieval, so the model can look up facts instead of guessing).

## Known issues

- `data/corpus.txt` still has two stray lines from an unclosed `{{SongBox` template.
- `sample.py` runs only the two settings `(1.0, None)` and `(0.8, 40)`.
- Git warns about LF to CRLF on Windows. Harmless.
- `torch` on Windows prints a NumPy warning. Harmless. `pip install numpy` removes it.
