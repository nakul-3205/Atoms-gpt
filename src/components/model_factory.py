import torch
import torch.nn as nn
import math
from src.logger import logger


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, context_length, dropout):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Causal mask — model can only look at past tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, context_length, dropout):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, context_length, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # Residual connection
        x = x + self.ff(self.ln2(x))     # Residual connection
        return x


class AtomsGPT(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        cfg = config['model']
        self.vocab_size = cfg['vocab_size']
        self.context_length = cfg['context_length']

        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['embed_dim'])
        self.drop = nn.Dropout(cfg['dropout'])

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                cfg['embed_dim'], cfg['n_heads'],
                cfg['context_length'], cfg['dropout']
            )
            for _ in range(cfg['n_layers'])
        ])

        self.ln_f = nn.LayerNorm(cfg['embed_dim'])
        self.head = nn.Linear(cfg['embed_dim'], cfg['vocab_size'], bias=False)

        # Weight tying (standard GPT trick — saves params)
        self.token_emb.weight = self.head.weight

        self._init_weights()
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Atoms-GPT initialized | Parameters: {total/1e6:.2f}M")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=50256  # pad token id
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx