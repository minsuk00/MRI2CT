"""Compact, correct conditional 1D DDPM for the bone toy.

Tests the GENERATIVE lane: a model that SAMPLES CT|MR instead of regressing its
mean. Predictions:
  - A single sample is SHARP and full-amplitude (fixes blur), unlike the regressor.
  - The POSTERIOR MEAN (average of many samples) reproduces the regressor's blur
    /undershoot -> proves "generative mean == regression", i.e. sampling buys
    realism, not per-voxel accuracy.
  - Per-bone SCATTER is bounded by the information limit (alpha): at alpha=0 samples
    are sharp but their heights are drawn from the prior -> distribution-calibrated
    but per-voxel inaccurate. No objective beats the information ceiling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sinusoidal(t, dim):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class EpsCNN(nn.Module):
    def __init__(self, cond_ch=1, width=96, tdim=64):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(tdim, width), nn.GELU(), nn.Linear(width, width))
        self.tdim = tdim
        self.in_conv = nn.Conv1d(1 + cond_ch, width, 5, padding=2)
        dils = [1, 2, 4, 8, 1]
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 5, padding=2 * d, dilation=d) for d in dils])
        self.out = nn.Conv1d(width, 1, 1)

    def forward(self, x_t, cond, t):
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        h = h + self.tmlp(sinusoidal(t, self.tdim))[:, :, None]
        for c in self.convs:
            h = F.gelu(c(h)) + h
        return self.out(h)


class DDPM:
    def __init__(self, T=200, width=96):
        self.T = T
        betas = torch.linspace(1e-4, 0.02, T, device=DEVICE)
        self.betas = betas
        self.alphas = 1 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        self.model = EpsCNN(width=width).to(DEVICE)

    def _norm(self, ct):
        return (ct - self.mu) / self.sigma

    def _denorm(self, x):
        return x * self.sigma + self.mu

    def fit(self, mr, ct, epochs=600, bs=64, lr=2e-3, seed=0, verbose=False):
        torch.manual_seed(seed)
        self.mu = float(ct.mean()); self.sigma = float(ct.std())
        S = mr.shape[0]
        MR = torch.from_numpy(mr[:, None, :].astype(np.float32)).to(DEVICE)
        X0 = torch.from_numpy(self._norm(ct)[:, None, :].astype(np.float32)).to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        idx = np.arange(S)
        for ep in range(epochs):
            np.random.default_rng(ep).shuffle(idx)
            self.model.train()
            for b0 in range(0, S, bs):
                bi = idx[b0:b0 + bs]
                x0 = X0[bi]; cond = MR[bi]
                t = torch.randint(0, self.T, (len(bi),), device=DEVICE)
                ab = self.abar[t][:, None, None]
                eps = torch.randn_like(x0)
                x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
                pred = self.model(x_t, cond, t)
                loss = F.mse_loss(pred, eps)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if verbose and ep % 150 == 0:
                print(f"    diff ep{ep} loss {loss.item():.4f}")

    @torch.no_grad()
    def sample(self, mr, n_per=1, seed=0):
        """Return [S, n_per, N] samples (denormalized to CT scale)."""
        self.model.eval()
        S = mr.shape[0]
        MR = torch.from_numpy(mr[:, None, :].astype(np.float32)).to(DEVICE)
        outs = []
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        for k in range(n_per):
            x = torch.randn(S, 1, MR.shape[-1], device=DEVICE, generator=g)
            for ti in reversed(range(self.T)):
                t = torch.full((S,), ti, device=DEVICE, dtype=torch.long)
                eps = self.model(x, MR, t)
                a = self.alphas[ti]; ab = self.abar[ti]; b = self.betas[ti]
                mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
                if ti > 0:
                    z = torch.randn(S, 1, MR.shape[-1], device=DEVICE, generator=g)
                    x = mean + b.sqrt() * z
                else:
                    x = mean
            outs.append(self._denorm(x).cpu().numpy()[:, 0, :])
        return np.stack(outs, axis=1)
