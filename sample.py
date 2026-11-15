"""Sample from the saved best checkpoint under different decoding settings.

Training already finished, so this only changes how tokens are picked:
no retraining, same weights. Compare temperature and top-k side by side.
"""

import json
import pathlib

import torch

from quahog.gpt import GPT, GPTConfig
from quahog.bpe import build_vocab, decode, encode

CKPT_PATH = pathlib.Path("checkpoints/quahog_gpt_v7_long.pt")
TOKENIZER_PATH = pathlib.Path("data/tokenizer.json")
PROMPTS = ["Peter", "Stewie", "The Griffins live"]
SETTINGS = [
    (1.0, None),
    (0.8, 40),
]
NEW_TOKENS = 150

device = "mps" if torch.backends.mps.is_available() else "cpu"

merges = {(a, b): idx for a, b, idx in json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))}
vocab = build_vocab(merges)

ckpt = torch.load(CKPT_PATH, map_location=device)
cfg = GPTConfig(**ckpt["config"])
model = GPT(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

torch.manual_seed(0)  # same random draws for each setting, so differences come from decoding
for temperature, top_k in SETTINGS:
    print(f"\n==================== temperature={temperature}  top_k={top_k} ====================")
    for prompt in PROMPTS:
        ids = torch.tensor([encode(prompt, merges)], dtype=torch.long, device=device)
        out = model.generate(ids, NEW_TOKENS, temperature=temperature, top_k=top_k)
        print(f"\n--- {prompt!r} ---")
        print(decode(out[0].tolist(), vocab))
