# QuahogGPT

A small language model that talks about Family Guy lore. Built step by step to learn how LLMs work.

## Roadmap

1. **Bigram model** (`step1_bigram.py`): character tokenizer, embedding table, cross-entropy loss, sampling.
2. **Tiny GPT from scratch**: self-attention, positional embeddings, transformer blocks, training loop.
3. **Real data**: scrape and clean a Family Guy wiki corpus, train a byte-level BPE tokenizer.
4. **Scale up training**: bigger model, GPU, checkpoints, eval loss.
5. **Fine-tune + retrieval**: LoRA on a pretrained small model, RAG over the wiki so facts stay accurate.

## Setup

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
python step1_bigram.py
```

Scrape and build the corpus (fetch is cached per source in `data/raw/`):

```powershell
python step3_scrape.py fetch all
python step3_scrape.py fetch Characters
python step3_scrape.py fetch Episodes
python step3_scrape.py build
python step3_bpe.py
python step4_train.py
```
