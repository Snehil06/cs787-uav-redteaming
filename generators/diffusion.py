"""Conditional DDPM over scenario parameter vectors, with classifier-free guidance."""
import math

import torch
from torch import nn

from generators.base import C_DIM, DEVICE, X_DIM, Generator, ResBlock


def cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    betas = 1 - ab[1:] / ab[:-1]
    return betas.clamp(1e-5, 0.999)


class Denoiser(nn.Module):
    def __init__(self, width: int = 512, depth: int = 4, emb: int = 256):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(128, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.c_mlp = nn.Sequential(nn.Linear(C_DIM, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Linear(X_DIM, width)
        self.blocks = nn.ModuleList([ResBlock(width, emb) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, X_DIM))

    @staticmethod
    def t_embed(t: torch.Tensor) -> torch.Tensor:
        half = 64
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def forward(self, x, t, c):
        e = self.t_mlp(self.t_embed(t)) + self.c_mlp(c)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, e)
        return self.out(h)


class DiffusionGenerator(Generator):
    name = "diffusion"

    def __init__(self, T: int = 200, seed: int = 0, **net_kw):
        super().__init__(seed)
        self.T = T
        self.net = Denoiser(**net_kw)
        betas = cosine_schedule(T)
        self.betas = betas.to(DEVICE)
        self.alphas = (1 - betas).to(DEVICE)
        self.ab = torch.cumprod(1 - betas, 0).to(DEVICE)

    def loss(self, x0, c):
        t = torch.randint(0, self.T, (len(x0),), device=DEVICE)
        eps = torch.randn_like(x0)
        ab = self.ab[t][:, None]
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return ((self.net(xt, t, c) - eps) ** 2).mean()

    def _sample(self, n, c, guidance: float = 2.0, clip_x0: bool = True):
        net = self.ema
        x = torch.randn(n, X_DIM, device=DEVICE)
        c_null = torch.zeros_like(c)
        for i in reversed(range(self.T)):
            t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
            eps_c = net(x, t, c)
            if guidance == 1.0:
                eps = eps_c
            else:
                eps_u = net(x, t, c_null)
                eps = eps_u + guidance * (eps_c - eps_u)
            ab, a, b = self.ab[i], self.alphas[i], self.betas[i]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            if clip_x0:
                x0 = x0.clamp(-1, 1)
            ab_prev = self.ab[i - 1] if i > 0 else torch.tensor(1.0, device=DEVICE)
            mean = (ab_prev.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                var = b * (1 - ab_prev) / (1 - ab)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
