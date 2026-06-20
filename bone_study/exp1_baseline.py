"""Experiment 1: does the L1 regressor reproduce bone undershoot + blur,
and how does it depend on alpha_info (MR bone-intensity content) and jitter
(edge-location uncertainty)? This validates the toy before testing solutions."""
import numpy as np
from toy_data import make_dataset
from toy_core import train_regressor, predict, eval_metrics, fmt

NTR, NTE = 1500, 400

print("=== Baseline L1 regressor across regimes ===")
print(f"{'alpha':>6} {'jitter':>7} | metrics (amp_ratio, edge_sharp: 1.0 = perfect)")
results = {}
for alpha in [0.0, 0.5, 1.0]:
    for jitter in [0.0, 1.5]:
        mr_tr, ct_tr, m_tr, b_tr = make_dataset(NTR, alpha, jitter, seed=1)
        mr_te, ct_te, m_te, b_te = make_dataset(NTE, alpha, jitter, seed=999)
        model = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0)
        pred = predict(model, mr_te)
        met = eval_metrics(pred, ct_te, m_te, b_te)
        results[(alpha, jitter)] = met
        print(f"{alpha:>6.1f} {jitter:>7.1f} | {fmt(met)}")

np.save("/home/minsukc/MRI2CT/bone_study/exp1_results.npy", results, allow_pickle=True)
print("\nInterpretation:")
print(" - amp_ratio < 1  => bulk UNDERSHOOT;  edge_sharp < 1 => BLUR.")
print(" - alpha controls recoverable INTENSITY; jitter controls BLUR floor.")
