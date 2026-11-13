"""Shared training / sampling interface for the learned generators (diffusion, cVAE)."""
import copy
import os

import numpy as np
import torch
from torch import nn

from generators.codec import (COND_DIM, cond_from_spec, cond_tensor_values, decode, encode,
                              random_cond_mask, scenario_conditions)
from scenarios.spec import VECTOR_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResBlock(nn.Module):
    """Pre-norm residual MLP block, modulated (FiLM) by a conditioning embedding."""

    def __init__(self, width: int, emb: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(emb, 2 * width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, x, e):
        scale, shift = self.film(e).chunk(2, dim=-1)
        return x + self.mlp(self.norm(x) * (1 + scale) + shift)


class Generator:
    """Subclasses define self.net, loss(x, c) and _sample(n, c, **kw) in model space."""
    name = "generator"

    def __init__(self, seed: int = 0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net: nn.Module
        self.ema: nn.Module | None = None

    # ---- data
    def prepare(self, vecs: np.ndarray, difficulty: np.ndarray):
        """Raw scenario vectors + measured F -> model-space tensors."""
        z = encode(vecs)
        cvals = scenario_conditions(vecs, difficulty)
        return z, cvals

    # ---- training
    def fit(self, z: np.ndarray, cvals: np.ndarray, steps: int = 4000, batch: int = 512, lr: float = 1e-3,
            log_every: int = 500, ema_decay: float = 0.999) -> list[float]:
        self.net.to(DEVICE).train()
        if self.ema is None:
            self.ema = copy.deepcopy(self.net).eval()
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        zt = torch.tensor(z, device=DEVICE)
        losses = []
        for step in range(steps):
            idx = self.rng.integers(0, len(z), batch)
            mask = random_cond_mask(self.rng, batch)
            c = torch.tensor(cond_tensor_values(cvals[idx], mask), device=DEVICE)
            loss = self.loss(zt[idx], c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                    pe.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
            losses.append(loss.item())
            if log_every and (step + 1) % log_every == 0:
                print(f"  [{self.name}] step {step + 1}/{steps}  loss {np.mean(losses[-log_every:]):.4f}")
        return losses

    # ---- sampling
    @torch.no_grad()
    def sample(self, n: int, spec: dict | None = None, clip: bool = False, **kw) -> np.ndarray:
        """Raw scenario vectors (n, 46) for the condition spec, e.g. {'difficulty': 0.8}."""
        c = torch.tensor(cond_from_spec(spec, n), device=DEVICE)
        z = self._sample(n, c, **kw).cpu().numpy()
        return decode(z, clip=clip)

    # ---- persistence
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"net": self.net.state_dict(), "ema": self.ema.state_dict()}, path)

    def load(self, path: str):
        state = torch.load(path, map_location=DEVICE)
        self.net.load_state_dict(state["net"])
        self.ema = copy.deepcopy(self.net)
        self.ema.load_state_dict(state["ema"])
        self.net.to(DEVICE)
        self.ema.to(DEVICE).eval()
        return self


X_DIM, C_DIM = VECTOR_DIM, COND_DIM
