"""Experiment 3: generative (conditional DDPM) vs regression.

Shows: (1) a single diffusion sample is SHARP (fixes blur) where the regressor is
blurred; (2) the posterior MEAN of many samples reproduces the regressor's
undershoot/blur (generative mean == regression); (3) per-bone scatter is bounded
by alpha (information limit), so sampling buys realism, not per-voxel accuracy."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_data import make_dataset
from toy_core import train_regressor, predict, eval_metrics, fmt
from toy_diffusion import DDPM

NTR, NTE = 1500, 400
JIT = 1.5
KMEAN = 16  # samples to average for the posterior mean

results = {}
for alpha in [0.0, 1.0]:
    print(f"\n=== alpha_info={alpha}, jitter={JIT} ===")
    mr_tr, ct_tr, m_tr, b_tr = make_dataset(NTR, alpha, JIT, seed=1)
    mr_te, ct_te, m_te, b_te = make_dataset(NTE, alpha, JIT, seed=999)

    # regressor baseline
    reg = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0)
    pred_reg = predict(reg, mr_te)
    m_reg = eval_metrics(pred_reg, ct_te, m_te, b_te)
    print(f"  {'regressor(L1)':>16} | {fmt(m_reg)}")

    # diffusion
    ddpm = DDPM(T=200, width=128)
    ddpm.fit(mr_tr, ct_tr, epochs=900, seed=0, verbose=True)
    samples = ddpm.sample(mr_te, n_per=KMEAN, seed=7)   # [S, K, N]
    one = samples[:, 0, :]
    post_mean = samples.mean(axis=1)
    m_one = eval_metrics(one, ct_te, m_te, b_te)
    m_pm = eval_metrics(post_mean, ct_te, m_te, b_te)
    print(f"  {'diffusion(1 sample)':>16} | {fmt(m_one)}")
    print(f"  {'diffusion(post-mean)':>16} | {fmt(m_pm)}")

    results[alpha] = dict(regressor=m_reg, diff_one=m_one, diff_postmean=m_pm)

    # figure: a couple of test examples
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, s in zip(axes, [0, 3]):
        ax.plot(ct_te[s], "k", lw=2.0, label="CT (target)")
        ax.plot(pred_reg[s], "C0", lw=1.4, label="regressor (L1)")
        ax.plot(one[s], "C3", lw=1.2, alpha=0.9, label="diffusion (1 sample)")
        ax.plot(post_mean[s], "C2--", lw=1.2, label="diffusion (posterior mean)")
        ax.set_ylim(-2, 7); ax.legend(fontsize=8, ncol=4)
    axes[0].set_title(f"alpha_info={alpha}: regressor blurs; a single diffusion sample is sharp; "
                      f"averaging samples (posterior mean) re-blurs")
    plt.tight_layout()
    plt.savefig(f"/home/minsukc/MRI2CT/bone_study/figs/03_generative_alpha{alpha}.png", dpi=90)
    plt.close()

np.save("/home/minsukc/MRI2CT/bone_study/exp3_results.npy", results, allow_pickle=True)
print("\nDone exp3.")
