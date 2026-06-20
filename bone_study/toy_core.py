"""
Core: models, training, and metrics for the 1D bone toy.

Metrics target the two failure modes the reports identified:
  amp_ratio    : recovered bone PEAK amplitude / true peak (1.0 = no undershoot)
  edge_sharp   : recovered edge gradient / true edge gradient (1.0 = no blur)
  bone_mae     : |pred-ct| over bone voxels  (per-voxel accuracy)
  soft_mae     : |pred-ct| over soft voxels  (PSNR / metric proxy)
All metrics are averaged over the test set; amp/edge are averaged per-bone.
"""
import numpy as np
import torch
import torch.nn as nn

from toy_data import N, _bump

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class CNN1D(nn.Module):
    """Plain 1D conv regressor. Large receptive field via dilations.

    in_ch>1 lets us append extra input channels (e.g. a retrieved template).
    Predicts a residual added to a learned constant base, range-free (no sigmoid
    cap in the toy: we want to test the OBJECTIVE, not clipping)."""

    def __init__(self, in_ch=1, width=64):
        super().__init__()
        dils = [1, 1, 2, 4, 8, 1]
        layers = []
        c = in_ch
        for i, d in enumerate(dils):
            layers += [nn.Conv1d(c, width, 5, padding=2 * d, dilation=d), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        return self.head(self.body(x))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def to_t(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(DEVICE)


def train_regressor(mr, ct, in_extra=None, loss="l1", epochs=400, bs=64,
                    lr=2e-3, width=64, seed=0, weight_bone=None, mask=None,
                    verbose=False):
    """mr, ct: [S, N]. in_extra: optional [S, K, N] extra input channels.
    weight_bone: if set, multiply per-voxel loss by this factor inside `mask`."""
    torch.manual_seed(seed)
    S = mr.shape[0]
    X = mr[:, None, :]
    if in_extra is not None:
        X = np.concatenate([X, in_extra], axis=1)
    Xt, Yt = to_t(X), to_t(ct[:, None, :])
    Wt = None
    if weight_bone is not None and mask is not None:
        w = np.ones_like(ct)
        w[mask] = weight_bone
        Wt = to_t(w[:, None, :])

    model = CNN1D(in_ch=X.shape[1], width=width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossfn = nn.L1Loss(reduction="none") if loss == "l1" else nn.MSELoss(reduction="none")

    idx = np.arange(S)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(idx)
        model.train()
        tot = 0.0
        for b0 in range(0, S, bs):
            bi = idx[b0:b0 + bs]
            pred = model(Xt[bi])
            l = lossfn(pred, Yt[bi])
            if Wt is not None:
                l = l * Wt[bi]
            l = l.mean()
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item() * len(bi)
        sched.step()
        if verbose and (ep % 100 == 0 or ep == epochs - 1):
            print(f"    ep{ep} loss {tot / S:.4f}")
    return model


@torch.no_grad()
def predict(model, mr, in_extra=None):
    model.eval()
    X = mr[:, None, :]
    if in_extra is not None:
        X = np.concatenate([X, in_extra], axis=1)
    return model(to_t(X)).cpu().numpy()[:, 0, :]


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _bone_support(b, n=N):
    prof = _bump(b["loc"], b["width"])
    idx = np.where(prof > 0.5)[0]
    return idx


def eval_metrics(pred, ct, mask, bones_list):
    """pred, ct, mask: [S, N]; bones_list: list (len S) of bone dicts.

    Amplitude error is split into BIAS (systematic undershoot, signed) and
    SCATTER (per-bone |error|, the information-limited part). At alpha=0 a model
    that predicts the mean height has ~0 bias but large scatter."""
    amp_bias, amp_scatter, edge_ratios = [], [], []
    for s in range(pred.shape[0]):
        p, c = pred[s], ct[s]
        for b in bones_list[s]:
            idx = _bone_support(b)
            if len(idx) < 2:
                continue
            lo, hi = idx.min(), idx.max()
            ringL = slice(max(0, lo - 6), max(1, lo - 2))
            ringR = slice(min(N - 1, hi + 3), min(N, hi + 7))
            base_c = np.concatenate([c[ringL], c[ringR]]).mean()
            base_p = np.concatenate([p[ringL], p[ringR]]).mean()
            true_amp = c[idx].max() - base_c
            # peak HU recovery: max over the support (+/-2 voxel shift tolerance) so a
            # SHARP sample shifted by 1-2 voxels still gets credit for reaching the
            # height. Per-voxel position error is captured separately by bone_mae.
            ps, pe = max(0, lo - 2), min(N, hi + 3)
            pred_amp = p[ps:pe].max() - base_p
            if true_amp > 0.5:
                amp_bias.append(pred_amp - true_amp)        # <0 => undershoot
                amp_scatter.append(abs(pred_amp - true_amp))
            w0, w1 = max(1, lo - 2), min(N, hi + 3)
            gc = np.abs(np.diff(c[w0:w1])).max()
            gp = np.abs(np.diff(p[w0:w1])).max()
            if gc > 0.3:
                edge_ratios.append(np.clip(gp / gc, 0, 2.0))
    bm = mask
    return dict(
        amp_bias=float(np.mean(amp_bias)),
        amp_scatter=float(np.mean(amp_scatter)),
        edge_sharp=float(np.mean(edge_ratios)),
        bone_mae=float(np.abs(pred - ct)[bm].mean()),
        soft_mae=float(np.abs(pred - ct)[~bm].mean()),
    )


def fmt(m):
    return (f"amp_bias={m['amp_bias']:+.2f}  amp_scat={m['amp_scatter']:.2f}  "
            f"edge={m['edge_sharp']:.2f}  bone_mae={m['bone_mae']:.3f}  soft_mae={m['soft_mae']:.3f}")
