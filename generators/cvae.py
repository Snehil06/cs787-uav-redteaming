"""Conditional VAE baseline: same data, same conditioning, same training budget as the diffusion model."""
import torch
from torch import nn

from generators.base import C_DIM, DEVICE, X_DIM, Generator, ResBlock


class _MLP(nn.Module):
    def __init__(self, d_in, d_out, width=512, depth=3, emb=256):
        super().__init__()
        self.c_mlp = nn.Sequential(nn.Linear(C_DIM, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([ResBlock(width, emb) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, d_out))

    def forward(self, x, c):
        e = self.c_mlp(c)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, e)
        return self.out(h)


class CVAENet(nn.Module):
    def __init__(self, latent: int = 16, **kw):
        super().__init__()
        self.latent = latent
        self.enc = _MLP(X_DIM, 2 * latent, **kw)
        self.dec = _MLP(latent, X_DIM, **kw)


class CVAEGenerator(Generator):
    name = "cvae"

    def __init__(self, latent: int = 32, beta: float = 0.3, warmup: int = 2000, free_bits: float = 0.1,
                 seed: int = 0, **net_kw):
        """beta < 1, KL warm-up and free bits all counter posterior collapse (decoder ignoring z)."""
        super().__init__(seed)
        self.beta, self.warmup, self.free_bits = beta, warmup, free_bits
        self.net = CVAENet(latent, **net_kw)
        self._step = 0

    def loss(self, x, c):
        mu, logvar = self.net.enc(x, c).chunk(2, dim=-1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        recon = self.net.dec(z, c)
        rec = ((recon - x) ** 2).sum(-1).mean()
        kl_dim = (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).mean(0)
        kl = torch.clamp(kl_dim, min=self.free_bits).sum()
        self._step += 1
        ramp = min(1.0, self._step / max(1, self.warmup))
        return (rec + ramp * self.beta * kl) / X_DIM

    def _sample(self, n, c, temperature: float = 1.0):
        z = torch.randn(n, self.net.latent, device=DEVICE) * temperature
        return self.ema.dec(z, c)
