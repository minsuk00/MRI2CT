"""Self-audit of every headline number in the U-Net failure analysis. Re-derives each
identity two independent ways, labels macro (mean over subjects) vs micro (pooled
sum/sum) explicitly, and asserts it closes. Run this before trusting the reports.
"""
import os
import numpy as np
import pandas as pd

RUN = os.path.join("/home/minsukc/MRI2CT", "evaluation_results/unet_failure_20260619")
EVAL = os.path.join("/home/minsukc/MRI2CT", "evaluation_results/full_eval_20260617")
ps = pd.read_csv(os.path.join(RUN, "per_subject.csv"))
pl = pd.read_csv(os.path.join(RUN, "per_label.csv"))
rel = pd.read_csv(os.path.join(EVAL, "metrics/per_subject.csv"))
rel = rel[rel.model == "unet"].copy()
m = ps.merge(rel[["subj_id", "body_mae_hu", "synthrad_mae", "body_psnr"]], on="subj_id")

checks = []
def chk(name, a, b, tol):
    ok = abs(a - b) < tol
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {a:.4f} vs {b:.4f}  (|d|={abs(a-b):.2e}, tol={tol})")

print("=== 1. tissue partition is exact (per subject) ===")
nres = (ps.n_air + ps.n_soft + ps.n_bone - ps.n_body).abs().max()
print(f"  [{'PASS' if nres == 0 else 'FAIL'}] max |n_air+n_soft+n_bone - n_body| = {nres}")
checks.append(nres == 0)

print("=== 2. reconciliation to released metrics (per subject, max abs err) ===")
chk("MACRO body_mae_hu vs released",
    (m.mae_clip * m.n_body / m.n_total).mean(), m.body_mae_hu.mean(), 1e-2)
chk("MICRO synthrad_mae vs released (mae_raw)",
    (m.mae_raw - m.synthrad_mae).abs().max(), 0.0, 1e-2)

print("=== 3. MACRO body_mae_hu = air+soft+bone contributions (macro) ===")
contrib = {t: (ps[f"aerr_sum_{t}_clip"] / ps.n_total).mean() for t in ["air", "soft", "bone"]}
chk("sum of macro tissue contributions = macro body_mae_hu",
    sum(contrib.values()), m.body_mae_hu.mean(), 1e-6)
print(f"      air {contrib['air']:.2f} + soft {contrib['soft']:.2f} + bone {contrib['bone']:.2f}")

print("=== 4. MICRO body-voxel-mean = air+soft+bone (micro), and != per-label avg ===")
micro_body = ps.total_aerr_sum_clip.sum() / ps.n_body.sum()
micro_tissue = sum(ps[f"aerr_sum_{t}_clip"].sum() for t in ["air", "soft", "bone"]) / ps.n_body.sum()
chk("micro body-voxel-mean = micro tissue sum", micro_body, micro_tissue, 1e-3)
lab_avg = (pl.mae_clip * pl.n).sum() / pl.n.sum()
coverage = 100 * ps.n_labeled.sum() / ps.n_body.sum()
print(f"      per-label voxel-avg (labeled only) = {lab_avg:.2f}  != body-voxel-mean {micro_body:.2f} (labels cover {coverage:.1f}%)")
checks.append(abs(lab_avg - micro_body) > 1.0)  # they should DIFFER
print(f"  [{'PASS' if abs(lab_avg-micro_body)>1.0 else 'FAIL'}] per-label avg differs from body MAE (expected, coverage<100%)")

print("=== 5. per-voxel MAE x prevalence (micro) = body-voxel-mean ===")
# micro per-voxel MAE per tissue, weighted by micro prevalence, must sum to micro body-voxel-mean
pv = {t: ps[f"aerr_sum_{t}_clip"].sum() / ps[f"n_{t}"].sum() for t in ["air", "soft", "bone"]}
prev = {t: ps[f"n_{t}"].sum() / ps.n_body.sum() for t in ["air", "soft", "bone"]}
chk("sum pv_mae*prevalence = micro body-voxel-mean",
    sum(pv[t] * prev[t] for t in pv), micro_body, 1e-3)
print(f"      per-voxel MAE: air {pv['air']:.1f}, soft {pv['soft']:.1f}, bone {pv['bone']:.1f}")

print("=== 6. macro contributions == oracle gains (consistency with report 06) ===")
# oracle gain for tissue t (macro body-MAE reduction) should equal its macro contribution
orc = pd.read_csv(os.path.join(RUN, "bone_oracle.csv"), index_col=0)
for t in ["air", "soft", "bone"]:
    chk(f"oracle ΔbodyMAE[{t}] == macro contribution[{t}]",
        float(orc.loc[t, "dBodyMAE"]), contrib[t], 5e-2)

print("\n" + ("ALL CHECKS PASS" if all(checks) else f"*** {checks.count(False)} CHECK(S) FAILED ***"))
