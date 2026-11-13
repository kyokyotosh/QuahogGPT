"""The GPT model, shared by training scripts. Architecture is the same as step 2,
with sizes moved into a config object so each experiment can choose its own.

    tokens -> embeddings (token + position) -> [attention, MLP] x N -> logits
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128  # context length in tokens
    n_embd: int = 128  # width of the residual stream
    n_head: int = 4  # attention heads; each works on n_embd // n_head dimensions
    n_layer: int = 4  # transformer blocks
    dropout: float = 0.2


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        # Lower-triangular mask: position t may attend to positions <= t only.
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        k = self.key(x)  # (B, T, head_size): what each position offers
        q = self.query(x)  # (B, T, head_size): what each position is looking for
        v = self.value(x)  # (B, T, head_size): what each position contributes
        # softmax(q @ k^T / sqrt(head_size)), future positions masked, dropout, then @ v.
        # The fused kernel computes exactly that, faster and with less memory at long T.
        p = self.dropout.p if self.training else 0.0
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel, concatenated, then projected back to n_embd."""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embd // cfg.n_head
        self.heads = nn.ModuleList([Head(cfg, head_size) for _ in range(cfg.n_head)])
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-position MLP: each position processes what attention gathered."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communicate (attention), then compute (MLP). Residuals keep gradients flowing."""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # one learned vector per position
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Sample tokens one at a time.

        temperature < 1 sharpens the distribution (safer, more repetitive);
        temperature > 1 flattens it (more random). top_k keeps only the k most
        likely tokens before sampling, which blocks the long tail of junk tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]  # only the last block_size tokens are visible
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]  # smallest logit still allowed
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
