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
