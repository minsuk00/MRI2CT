"""Experiment 4: cheap objective/post-hoc tweaks (the moves we argued won't
fundamentally work). Confirms they give only small bias gains and never beat the
information ceiling.

  - L2 regression (mean, not median): slightly different bias, same scatter.
  - bone-weighted L1 (x5, x20): capacity reallocation -> small bias gain, scatter floor unchanged.
  - post-hoc recalibration: a learned bone gain fixes BULK bias but not scatter or blur.
"""
import numpy as np
from toy_data import make_dataset
from toy_core import train_regressor, predict, eval_metrics, fmt

NTR, NTE = 1500, 400
JIT = 1.5


def recalibrate(pred, mask, gain):
    """Apply a scalar gain to the bone residual above a local baseline.
    Models a post-hoc HU recalibration restricted to bone regions."""
    out = pred.copy()
    for s in range(pred.shape[0]):
        m = mask[s]
        if not m.any():
            continue
        base = np.median(pred[s][~m]) if (~m).any() else 0.0
        out[s][m] = base + gain * (pred[s][m] - base)
    return out


def fit_gain(pred, ct, mask):
    """Choose the bone gain minimizing bone MAE on the calibration (train) set."""
    best, bg = 1e9, 1.0
    for g in np.linspace(1.0, 2.5, 16):
        e = np.abs(recalibrate(pred, mask, g) - ct)[mask].mean()
        if e < best:
            best, bg = e, g
    return bg


results = {}
for alpha in [0.0, 1.0]:
    print(f"\n=== alpha_info={alpha}, jitter={JIT} ===")
    mr_tr, ct_tr, m_tr, b_tr = make_dataset(NTR, alpha, JIT, seed=1)
    mr_te, ct_te, m_te, b_te = make_dataset(NTE, alpha, JIT, seed=999)
    res = {}

    reg = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0)
    pred = predict(reg, mr_te)
    res["L1 (baseline)"] = eval_metrics(pred, ct_te, m_te, b_te)
    print(f"  {'L1 (baseline)':>18} | {fmt(res['L1 (baseline)'])}")

    reg2 = train_regressor(mr_tr, ct_tr, loss="l2", epochs=400, seed=0)
    res["L2"] = eval_metrics(predict(reg2, mr_te), ct_te, m_te, b_te)
    print(f"  {'L2':>18} | {fmt(res['L2'])}")

    for wb in [5.0, 20.0]:
        rw = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0,
                             weight_bone=wb, mask=m_tr)
        res[f"bone-weighted L1 x{int(wb)}"] = eval_metrics(predict(rw, mr_te), ct_te, m_te, b_te)
        print(f"  {f'bone-weighted L1 x{int(wb)}':>18} | {fmt(res[f'bone-weighted L1 x{int(wb)}'])}")

    pred_tr = predict(reg, mr_tr)
    g = fit_gain(pred_tr, ct_tr, m_tr)
    rec = recalibrate(pred, m_te, g)
    res[f"recalibrate (g={g:.2f})"] = eval_metrics(rec, ct_te, m_te, b_te)
    print(f"  {f'recalibrate (g={g:.2f})':>18} | {fmt(res[f'recalibrate (g={g:.2f})'])}")
    results[alpha] = res

np.save("/home/minsukc/MRI2CT/bone_study/exp4_results.npy", results, allow_pickle=True)

print("\nExpect: all stay near the L1 baseline's bone_mae; recalibration fixes")
print("amp_bias (less negative) but amp_scatter/edge are unchanged -> not a solution.")
