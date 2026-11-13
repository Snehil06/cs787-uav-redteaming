"""DiffScene-style baseline: a diffusion prior steered by the gradient of a learned adversarial objective.

Following DiffScene (Xu et al. 2023; 2025), at every reverse step the predicted clean sample x0 is moved by a few
gradient-ascent steps on an objective J, then re-noised:

    x0  <- x0 + alpha * grad_x0 [ J_phi(x0) - lam * ||x0 - x0_prior||^2 ]

DiffScene's objectives (distance to the ego vehicle, route blocking) are differentiable functions of a trajectory.
Our failure score F comes from a black-box simulator with a sampling-based planner, so J_phi is a learned
surrogate critic of the simulator (the "learned critic" variant of DiffScene). The proximal term plays the role
of DiffScene's constraint objective: it keeps the guided sample near what the prior predicted.

Tuning options (all off by default = plain DiffScene):
  target="fail"     critic is a classifier of P(failure) instead of a regressor of F
  n_critics, k_unc  ensemble of critics; ascend mean - k_unc * std (pessimistic, harder to exploit)
  norm_grad         fixed-length gradient steps per sample
  t_frac            guide only the last t_frac of the reverse steps (prior shapes the sample first)
  validity, w_val   add w_val * log P(valid) from a learned validity classifier to the objective
  cfg_spec, cfg_w   hybrid: classifier-free guided prior (e.g. difficulty = 1, w = 2) instead of unconditional
"""
import copy

import numpy as np
import torch
from torch import nn

from generators.base import DEVICE, X_DIM
from generators.codec import cond_from_spec, decode, encode


class Critic(nn.Module):
    """Model-space scenario -> score in [0, 1] (predicted F, P(failure) or P(valid))."""

    def __init__(self, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(X_DIM, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def logit(self, z):
        return self.net(z).squeeze(-1)

    def forward(self, z):
        return torch.sigmoid(self.logit(z))


def train_critic(vecs: np.ndarray, y: np.ndarray, critic: Critic | None = None, steps: int = 3000,
                 lr: float = 1e-3, seed: int = 0, batch: int = 512, loss: str = "mse", z: np.ndarray | None = None,
                 bootstrap: bool = False) -> Critic:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    critic = critic or Critic()
    critic.to(DEVICE).train()
    zt = torch.tensor(encode(vecs) if z is None else z, device=DEVICE)
    yt = torch.tensor(np.clip(y, 0, 1), dtype=torch.float32, device=DEVICE)
    pool = rng.integers(0, len(zt), len(zt)) if bootstrap else np.arange(len(zt))
    opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.as_tensor(pool[rng.integers(0, len(pool), batch)], device=DEVICE)
        if loss == "bce":
            l = nn.functional.binary_cross_entropy_with_logits(critic.logit(zt[idx]), yt[idx])
        else:
            l = ((critic(zt[idx]) - yt[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        l.backward()
        opt.step()
        sched.step()
    return critic.eval()


def train_ensemble(vecs, y, n: int, loss: str, seed: int = 0, steps: int = 3000):
    return [train_critic(vecs, y, steps=steps, seed=seed + 101 * k, loss=loss, bootstrap=n > 1) for k in range(n)]


@torch.no_grad()
def predict(critics, vecs: np.ndarray) -> np.ndarray:
    z = torch.tensor(encode(vecs), device=DEVICE)
    return torch.stack([c(z) for c in critics]).mean(0).cpu().numpy()


class DiffSceneGenerator:
    """Wraps a trained DiffusionGenerator (the prior) and critic(s); samples with gradient guidance."""
    name = "diffscene"

    def __init__(self, prior, critics, alpha: float = 3.0, inner_steps: int = 3, lam: float = 1.0, seed: int = 0,
                 k_unc: float = 0.0, norm_grad: bool = False, t_frac: float = 1.0, validity: Critic | None = None,
                 w_val: float = 0.0, cfg_spec: dict | None = None, cfg_w: float = 1.0, target: str = "F"):
        self.prior = prior
        self.critics = critics if isinstance(critics, list) else [critics]
        self.alpha, self.inner_steps, self.lam = alpha, inner_steps, lam
        self.k_unc, self.norm_grad, self.t_frac = k_unc, norm_grad, t_frac
        self.validity, self.w_val = validity, w_val
        self.cfg_spec, self.cfg_w, self.target = cfg_spec, cfg_w, target
        torch.manual_seed(seed)

    def objective(self, z):
        preds = torch.stack([c(z) for c in self.critics])            # (n_critics, batch)
        J = preds.mean(0)
        if self.k_unc and len(self.critics) > 1:
            J = J - self.k_unc * preds.std(0)
        if self.validity is not None and self.w_val:
            J = J + self.w_val * nn.functional.logsigmoid(self.validity.logit(z))
        return J

    def sample(self, n: int, spec: dict | None = None, clip: bool = False, **_) -> np.ndarray:
        p = self.prior
        net = p.ema
        c_null = torch.tensor(cond_from_spec(None, n), device=DEVICE)
        c_cfg = torch.tensor(cond_from_spec(self.cfg_spec, n), device=DEVICE) if self.cfg_spec else None
        x = torch.randn(n, X_DIM, device=DEVICE)
        t_guide = int(round(self.t_frac * p.T))
        for i in reversed(range(p.T)):
            t = torch.full((n,), i, device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                eps = net(x, t, c_null)
                if c_cfg is not None:
                    eps = eps + self.cfg_w * (net(x, t, c_cfg) - eps)
            ab, a, b = p.ab[i], p.alphas[i], p.betas[i]
            x0_prior = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
            x0 = x0_prior
            if self.alpha > 0 and i < t_guide:
                g_x0 = x0_prior.clone()
                for _k in range(self.inner_steps):
                    g_x0.requires_grad_(True)
                    with torch.enable_grad():
                        J = self.objective(g_x0).sum() - self.lam * ((g_x0 - x0_prior) ** 2).sum()
                        (g,) = torch.autograd.grad(J, g_x0)
                    if self.norm_grad:
                        g = g / (g.norm(dim=1, keepdim=True) + 1e-8)
                    g_x0 = (g_x0 + self.alpha * g).clamp(-1, 1).detach()
                x0 = g_x0
            ab_prev = p.ab[i - 1] if i > 0 else torch.tensor(1.0, device=DEVICE)
            mean = (ab_prev.sqrt() * b / (1 - ab)) * x0 + (a.sqrt() * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                var = b * (1 - ab_prev) / (1 - ab)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return decode(x.detach().cpu().numpy(), clip=clip)

    def refit_critic(self, vecs: np.ndarray, F: np.ndarray, outcome: np.ndarray | None = None, steps: int = 1000,
                     seed: int = 0):
        """Feedback variant: continue training the critic(s) on everything flown so far."""
        y, loss = ((outcome == "failure").astype(float), "bce") if self.target == "fail" else (F, "mse")
        self.critics = [train_critic(vecs, y, critic=copy.deepcopy(c), steps=steps, lr=3e-4, seed=seed + k,
                                     loss=loss, bootstrap=len(self.critics) > 1)
                        for k, c in enumerate(self.critics)]
