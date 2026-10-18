"""Zero-output-initialized cross-attention sidecar. State keys match historical Z3."""

import torch
from torch import nn
from torch.nn import functional as F


class StyleAdapter(nn.Module):
    def __init__(self, dim=256, width=256, tokens=8, layers=4, hidden=6144):
        super().__init__()
        if width < 64 or width % 64 or min(dim, tokens, layers, hidden) < 1:
            raise ValueError("width must be a positive multiple of 64; dimensions must be positive")
        self.width, self.tokens, self.heads = width, tokens, width // 64
        self.projector = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, tokens * width)
        )
        self.norm = nn.LayerNorm(width)
        self.key, self.value = nn.Linear(width, width), nn.Linear(width, width)
        self.query = nn.ModuleList([nn.Linear(hidden, width) for _ in range(layers)])
        self.out = nn.ModuleList([nn.Linear(width, hidden, bias=False) for _ in range(layers)])
        for layer in self.out:
            nn.init.zeros_(layer.weight)

    def encode(self, style):
        return self.norm(self.projector(style).reshape(style.shape[0], self.tokens, self.width))

    def forward(self, x, style_tokens, index, image_tokens, active, scale=1.0):
        im = x[:, :image_tokens]
        batch = im.shape[0]
        q = self.query[index](im).reshape(batch, -1, self.heads, 64).transpose(1, 2)
        k = self.key(style_tokens).reshape(batch, -1, self.heads, 64).transpose(1, 2)
        v = self.value(style_tokens).reshape(batch, -1, self.heads, 64).transpose(1, 2)
        delta = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, -1, self.width)
        delta = self.out[index](delta) * active[:, None, None] * scale
        return torch.cat((im + delta.to(im.dtype), x[:, image_tokens:]), dim=1)
