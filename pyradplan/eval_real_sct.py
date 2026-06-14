"""Dose evaluation on a REAL model sCT: optimize plan on gtCT, deliver on sCT,
compute MAE_dose / DVH / gamma. Uses UNet outputs from the eval dump.

REQUIRES THE DEDICATED `pyradplan` ENV:
    micromamba run -n pyradplan python pyradplan/eval_real_sct.py [SUBJECT_ID] [MODEL]

sCT  = <EVALROOT>/<model>/<subj>/sample.nii.gz   (model prediction)
gtCT = <EVALROOT>/<model>/<subj>/target.nii.gz   (ground truth, same grid)
body mask + totalseg come from the GPFS subject dir (same grid, verified).
Target = synthetic sphere PTV at body centroid (no clinical PTV available).
OARs   = up to 3 largest TotalSeg organs near the target (for DVH).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import plan_transfer_example as pt
import dose_metrics as dm
from pyRadPlan import fluence_optimization
from pyRadPlan.core import xp_utils as xp

EVALROOT = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260609/volumes"
GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
OUT = os.path.join(os.path.dirname(__file__), "out", "real_sct")
RX = 60.0
GANTRY = [0, 72, 144, 216, 288]

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "1ABB002"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "unet"


def main():
    xp.PREFER_GPU = False
    os.makedirs(OUT, exist_ok=True)

    sdir = f"{EVALROOT}/{MODEL}/{SUBJ}"
    gt = nib.load(f"{sdir}/target.nii.gz").get_fdata().astype(np.float32)   # ground-truth CT
    sct = nib.load(f"{sdir}/sample.nii.gz").get_fdata().astype(np.float32)  # model sCT
    body = nib.load(f"{GPFS}/{SUBJ}/mask.nii").get_fdata().astype(np.uint8)
    seg = nib.load(f"{GPFS}/{SUBJ}/ct_totalseg.nii.gz").get_fdata().astype(int)
    assert gt.shape == body.shape == sct.shape, (gt.shape, body.shape, sct.shape)
    print("SUBJ %s model=%s shape=%s gtHU=[%.0f,%.0f] sctHU=[%.0f,%.0f]"
          % (SUBJ, MODEL, gt.shape, gt.min(), gt.max(), sct.min(), sct.max()))

    # synthetic spherical PTV at body centroid
    c = [int(v.mean()) for v in np.nonzero(body > 0)]
    X, Y, Z = np.indices(body.shape)
    target = ((((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= 20**2).astype(np.uint8)) & body

    # 1. optimize plan on gtCT
    ct_gt, cst, pln, stf, dij_gt = pt.build_problem(gt, body, target)
    w = np.asarray(fluence_optimization(ct_gt, cst, stf, dij_gt, pln)).ravel()
    dose_gt = pt.dose_of(dij_gt, w)
    # 2. deliver SAME plan on the real sCT
    _, _, _, _, dij_sct = pt.build_problem(sct, body, target)
    dose_sct = pt.dose_of(dij_sct, w)
    print("nbeamlets=%d dmax_gt=%.2f dmax_sct=%.2f" % (w.size, dose_gt.max(), dose_sct.max()))

    # masks in (z,y,x) to match dose cubes
    tt = np.transpose(target, (2, 1, 0)).astype(bool)
    bb = np.transpose(body, (2, 1, 0)).astype(bool)
    seg_t = np.transpose(seg, (2, 1, 0))

    # all non-zero TotalSeg organs as OAR candidates; dvh_metric ranks them by
    # (D5+Dmean)/2 on the GT dose and uses the top 3 (matches SynthRAD code).
    labs = [int(l) for l in np.unique(seg_t[bb]) if l > 0]
    oar_masks = {f"organ{l}": (seg_t == l) for l in labs}

    # 3. metrics (functionally matching SynthRAD: MAE eq.5, DVH composite eqs.6-9, gamma 2%/2mm)
    mae = dm.mae_dose(dose_gt, dose_sct, RX, threshold=0.9)
    dvh_val, dvh = dm.dvh_metric(dose_gt, dose_sct, tt, oar_masks, RX, n_oars=3)
    gamma = dm.gamma_pass_rate(dose_gt, dose_sct, spacing_mm=1.5, dose_pct=2.0, dist_mm=2.0)

    print("=== METRICS (%s %s) ===" % (SUBJ, MODEL))
    print("MAE_dose (eq.5, /Rx)      = %.4f" % mae)
    print("DVH_metric (composite)    = %.4f  (target=%.4f oar=%.4f)" % (dvh_val, dvh["target_term"], dvh["oar_term"]))
    print("gamma 2%%/2mm pass rate    = %.2f%%" % gamma)
    print("  PTV D98: gt=%.2f sct=%.2f Gy | V95: gt=%.1f sct=%.1f %%"
          % (dvh["PTV_D98_gt"], dvh["PTV_D98_sct"], dvh["PTV_V95_gt"], dvh["PTV_V95_sct"]))
    print("  OARs used (3 most-irradiated):", dvh["oars_used"])
    for nm, d in dvh["per_oar"].items():
        print("    %s: D2 gt=%.2f sct=%.2f | Dmean gt=%.2f sct=%.2f Gy"
              % (nm, d["D2_gt"], d["D2_sct"], d["Dmean_gt"], d["Dmean_sct"]))

    # 4. figure (5-panel) + DVH overlay
    diff = dose_sct - dose_gt
    dmax = float(dose_gt.max())
    dlim = max(float(np.percentile(np.abs(diff[bb]), 99)), 1e-3)
    sl = c[2]
    gt_t = np.transpose(gt, (2, 1, 0)); sct_t = np.transpose(sct, (2, 1, 0))
    fig, ax = plt.subplots(1, 6, figsize=(26, 4.6))
    ax[0].imshow(gt_t[sl], cmap="gray", vmin=-1000, vmax=1000); ax[0].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[0].set_title("gtCT (HU)"); ax[0].axis("off")
    ax[1].imshow(sct_t[sl], cmap="gray", vmin=-1000, vmax=1000); ax[1].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[1].set_title("UNet sCT (HU)"); ax[1].axis("off")
    for a, d, ttl in [(ax[2], dose_gt, "dose on gtCT"), (ax[3], dose_sct, "dose on sCT (same plan)")]:
        a.imshow(gt_t[sl], cmap="gray", vmin=-1000, vmax=1000)
        im = a.imshow(np.ma.masked_less(d[sl], 0.05*dmax), cmap="jet", alpha=0.6, vmin=0, vmax=dmax)
        a.contour(tt[sl], colors="lime", linewidths=0.8); a.set_title(ttl); a.axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="Gy")
    dd = np.ma.masked_where(~bb[sl], diff[sl])
    im2 = ax[4].imshow(dd, cmap="bwr", vmin=-dlim, vmax=dlim); ax[4].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[4].set_title("dose diff +/-%.2f Gy" % dlim); ax[4].axis("off")
    fig.colorbar(im2, ax=ax[4], fraction=0.046, label="Gy")
    # DVH overlay: PTV + the top-ranked OAR used by the metric
    dvh_curves = [("PTV", tt, "tab:red")]
    if dvh["oars_used"]:
        top_oar = dvh["oars_used"][0]
        dvh_curves.append((top_oar, oar_masks[top_oar], "tab:gray"))
    for nm, msk, col in dvh_curves:
        x = np.linspace(0, dmax * 1.05, 200)
        ygt = [(dose_gt[msk] >= t).mean() * 100 for t in x]
        ysct = [(dose_sct[msk] >= t).mean() * 100 for t in x]
        ax[5].plot(x, ygt, col, label=f"{nm} gt"); ax[5].plot(x, ysct, col, ls="--", label=f"{nm} sCT")
    ax[5].set_xlabel("Dose (Gy)"); ax[5].set_ylabel("Vol %"); ax[5].set_title("DVH (solid=gt, dash=sCT)")
    ax[5].legend(fontsize=7); ax[5].grid(alpha=0.3)
    fig.suptitle("%s %s | MAE_dose=%.4f  DVH=%.4f  gamma2%%/2mm=%.1f%%  PTV D98 gt=%.1f sct=%.1f"
                 % (SUBJ, MODEL, mae, dvh_val, gamma, dvh["PTV_D98_gt"], dvh["PTV_D98_sct"]))
    fig.tight_layout()
    out = os.path.join(OUT, f"{SUBJ}_{MODEL}_dose_eval.png")
    fig.savefig(out, dpi=110); print("SAVED", out); print("PASS")


if __name__ == "__main__":
    main()
