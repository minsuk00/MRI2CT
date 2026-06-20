"""Experiment 5 (synthesis): the INFORMATION CEILING.

Sweep alpha_info and plot bone error for each method. Thesis:
  - regressor, realistic-retrieval, and single-sample diffusion all sit on the
    SAME descending ceiling -> per-voxel bone accuracy is set by the MR's bone
    information (alpha), not the method.
  - Only the ORACLE template (true side information) breaks the ceiling.
  - But the methods differ on edge_sharp: diffusion and a sharp template restore
    sharpness (the clinically-relevant half) even on the ceiling.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_data import make_dataset
from toy_core import train_regressor, predict, eval_metrics
from toy_diffusion import DDPM
from exp2_retrieval import build_templates

NTR, NTE = 1500, 400
JIT = 1.5
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

rows = {k: {"bone_mae": [], "edge_sharp": [], "amp_scatter": []}
        for k in ["regressor", "retrieved", "oracle_tmpl", "diffusion"]}

for alpha in ALPHAS:
    print(f"\n--- alpha={alpha} ---")
    mr_tr, ct_tr, m_tr, b_tr = make_dataset(NTR, alpha, JIT, seed=1)
    mr_te, ct_te, m_te, b_te = make_dataset(NTE, alpha, JIT, seed=999)

    reg = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0)
    met = eval_metrics(predict(reg, mr_te), ct_te, m_te, b_te)
    for k in rows["regressor"]:
        rows["regressor"][k].append(met[k])
    print(f"  regressor   bone_mae={met['bone_mae']:.3f} edge={met['edge_sharp']:.2f}")

    for kind, key, rj in [("retrieved", "retrieved", 0.8), ("oracle", "oracle_tmpl", 0.0)]:
        tr_t = build_templates(ct_tr, b_tr, kind, alpha, rj, seed=1)
        te_t = build_templates(ct_te, b_te, kind, alpha, rj, seed=999)
        mdl = train_regressor(mr_tr, ct_tr, in_extra=tr_t, loss="l1", epochs=400, seed=0)
        met = eval_metrics(predict(mdl, mr_te, in_extra=te_t), ct_te, m_te, b_te)
        for k in rows[key]:
            rows[key][k].append(met[k])
        print(f"  {key:11} bone_mae={met['bone_mae']:.3f} edge={met['edge_sharp']:.2f}")

    ddpm = DDPM(T=200, width=128)
    ddpm.fit(mr_tr, ct_tr, epochs=900, seed=0)
    one = ddpm.sample(mr_te, n_per=1, seed=7)[:, 0, :]
    met = eval_metrics(one, ct_te, m_te, b_te)
    for k in rows["diffusion"]:
        rows["diffusion"][k].append(met[k])
    print(f"  diffusion   bone_mae={met['bone_mae']:.3f} edge={met['edge_sharp']:.2f}")

np.save("/home/minsukc/MRI2CT/bone_study/exp5_results.npy",
        {"alphas": ALPHAS, "rows": rows}, allow_pickle=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
styles = {"regressor": ("C0", "o"), "retrieved": ("C1", "s"),
          "oracle_tmpl": ("C2", "^"), "diffusion": ("C3", "D")}
for key, (col, mk) in styles.items():
    axes[0].plot(ALPHAS, rows[key]["bone_mae"], col, marker=mk, label=key)
    axes[1].plot(ALPHAS, rows[key]["edge_sharp"], col, marker=mk, label=key)
axes[0].set_xlabel("alpha_info (MR bone-intensity content)"); axes[0].set_ylabel("bone MAE (lower=better)")
axes[0].set_title("Per-voxel bone error: one descending CEILING\n(only oracle side-info breaks it)")
axes[1].set_xlabel("alpha_info"); axes[1].set_ylabel("edge sharpness (1=real)")
axes[1].set_title("Edge sharpness: diffusion / sharp template\nrestore it even on the ceiling")
axes[1].axhline(1.0, color="k", ls=":", lw=1)
for ax in axes:
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/home/minsukc/MRI2CT/bone_study/figs/05_ceiling.png", dpi=95)
print("\nsaved 05_ceiling.png")
