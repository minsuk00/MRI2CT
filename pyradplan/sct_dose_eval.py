"""End-to-end sCT-vs-gtCT dose evaluation with pyRadPlan.

REQUIRES THE DEDICATED `pyradplan` ENV (see README.md):
    micromamba run -n pyradplan python pyradplan/sct_dose_eval.py [SUBJECT_ID]

Pipeline:
  0. HU-sensitivity sanity check (same fluence on water-block vs bone-block must differ).
  1. Build gtCT (HU -> density via pyRadPlan's default HLUT, auto-computed in create_ct).
  2. Plant a synthetic spherical PTV (TotalSeg has no tumor) + body EXTERNAL.
  3. Optimize a 5-beam photon IMRT plan on gtCT -> fluence weights w.
  4. Dose on gtCT with w (and an open/uniform-fluence field too).
  5. Build a PROXY sCT (no real sCT on disk yet): bias bone HU by -250 -> sCT.
  6. Dose on the proxy sCT with the SAME w.
  7. Metrics: body MAE, PTV Dmean/D95 shift, 3%-dose-difference pass rate, DVH.
  8. Visualization -> pyradplan/out/<subj>_sct_dose_eval.png

API (pyRadPlan 0.4.0): fluence_optimization returns the weight vector (ndarray);
dose cubes come from dij.compute_result_ct_grid(weights)["physical_dose"].
The photon SVD engine breaks on the torch-GPU backend, so dose calc is forced to CPU.
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "1THB008"
ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
OUT = os.path.join(os.path.dirname(__file__), "out")
RX = 60.0  # prescription Gy


def to_sitk(arr_xyz, spacing=1.5, cast=np.float32):
    a = np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0)))  # (x,y,z) -> sitk (z,y,x)
    img = sitk.GetImageFromArray(a.astype(cast))
    img.SetSpacing((spacing,) * 3)
    return img


def dvh(dose, mask, n=200):
    d = dose[mask]
    x = np.linspace(0, max(float(d.max()), 1e-6) * 1.05, n)
    y = np.array([(d >= t).mean() * 100.0 for t in x])
    return x, y


def main():
    from pyRadPlan import (PhotonPlan, generate_stf, calc_dose_influence, fluence_optimization)
    from pyRadPlan.ct import create_ct
    from pyRadPlan.cst import create_voi, create_cst
    from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing
    from pyRadPlan.core import xp_utils as xp

    # 0.4.0 photon SVD engine breaks on the torch-GPU backend ("tensors on cuda:0 and
    # cpu"); GPU is detected but the dose path is broken -> force numpy/CPU (fast).
    xp.PREFER_GPU = False
    os.makedirs(OUT, exist_ok=True)
    print("BACKEND active_ns=%s (torch_gpu_available=%s, forced CPU for dose calc)"
          % (getattr(xp.choose_array_api_namespace(), "__name__", "?"), xp.pytorch_gpu_available()))

    def plan_pieces(hu_arr, body_arr, target_arr, spacing, gantry):
        ct = create_ct(cube_hu=to_sitk(hu_arr, spacing))
        cst = create_cst([
            create_voi(voi_type="EXTERNAL", name="Body", ct_image=ct,
                       mask=to_sitk(body_arr, spacing, np.uint8),
                       objectives=[SquaredOverdosing(parameters=[30.0], priority=10.0)]),
            create_voi(voi_type="TARGET", name="PTV", ct_image=ct,
                       mask=to_sitk(target_arr, spacing, np.uint8),
                       objectives=[SquaredDeviation(parameters=[RX], priority=800.0)]),
        ], ct=ct)
        pln = PhotonPlan(machine="Generic")
        pln.prop_stf = {"gantry_angles": gantry, "couch_angles": [0] * len(gantry)}
        stf = generate_stf(ct, cst, pln)
        dij = calc_dose_influence(ct, cst, stf, pln)
        return ct, cst, pln, stf, dij

    def dose_on(dij, w):
        # compute_result_ct_grid returns {name: sitk.Image}; GetArrayFromImage -> (z,y,x),
        # matching our np.transpose(.,(2,1,0)) masks.
        img = dij.compute_result_ct_grid(np.asarray(w, np.float32).ravel())["physical_dose"]
        return sitk.GetArrayFromImage(img).astype(np.float32)

    # ---- 0. HU-sensitivity sanity (same fluence, water-block vs bone-block) --
    try:
        cube_w = np.full((40, 40, 40), -1000.0, np.float32); cube_w[8:32, 8:32, 8:32] = 0.0
        cube_b = cube_w.copy(); cube_b[8:32, 8:32, 8:32] = 1000.0
        bm = (cube_w > -500).astype(np.uint8)
        tg = np.zeros_like(bm); tg[16:24, 16:24, 16:24] = 1; tg &= bm
        _, cw, pw, sw, dijw = plan_pieces(cube_w, bm, tg, 3.0, [0])
        w_ph = np.asarray(fluence_optimization(_, cw, sw, dijw, pw)).ravel()
        d_w = dose_on(dijw, w_ph)
        _, cb, pb, sb, dijb = plan_pieces(cube_b, bm, tg, 3.0, [0])
        d_b = dose_on(dijb, w_ph)                          # SAME fluence, bone block
        sens = float(np.abs(d_b - d_w).mean())
        print("HU_SENSITIVITY same-fluence water-vs-bone: MAE=%.5f Gy maxdiff=%.4f Gy -> %s"
              % (sens, float(np.abs(d_b - d_w).max()),
                 "HU-SENSITIVE (good)" if sens > 1e-4 else "HU-INSENSITIVE!"))
    except Exception as e:
        sens = float("nan")
        import traceback; traceback.print_exc()
        print("HU_SENSITIVITY skipped: %s: %s" % (type(e).__name__, str(e)[:120]))

    # ---- 1-3. gtCT, structures, optimize -----------------------------------
    p = f"{ROOT}/{SUBJ}/"
    hu = nib.load(p + "ct.nii").get_fdata().astype(np.float32)
    body = nib.load(p + "mask.nii").get_fdata().astype(np.uint8)
    c = [int(v.mean()) for v in np.nonzero(body > 0)]
    X, Y, Z = np.indices(body.shape)
    target = ((((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= 15**2).astype(np.uint8)) & body
    print("SUBJ %s shape=%s HU=[%.0f,%.0f] PTV_vox=%d" % (SUBJ, hu.shape, hu.min(), hu.max(), int(target.sum())))

    gant = [0, 72, 144, 216, 288]
    ct_gt, cst, pln, stf, dij = plan_pieces(hu, body, target, 1.5, gant)
    t = time.time(); w = np.asarray(fluence_optimization(ct_gt, cst, stf, dij, pln)).ravel(); print("opt %.1fs nbixels=%d" % (time.time() - t, w.size))

    # ---- 4. dose on gtCT (optimized + uniform/open field) ------------------
    d_gt = dose_on(dij, w)
    d_uniform = dose_on(dij, np.ones_like(w))

    # ---- 5-6. proxy sCT (bone HU bias) + dose with SAME w -----------------
    sct = hu.copy(); sct[(hu > 150) & (body > 0)] -= 250.0
    _, _, _, _, dij_s = plan_pieces(sct, body, target, 1.5, gant)
    d_sct = dose_on(dij_s, w)

    # ---- 7. metrics --------------------------------------------------------
    tt = np.transpose(target, (2, 1, 0)).astype(bool)
    bb = np.transpose(body, (2, 1, 0)).astype(bool)
    # guard against any grid mismatch
    if d_gt.shape != tt.shape:
        print("WARN dose shape %s != mask shape %s" % (d_gt.shape, tt.shape))
    dmax = float(d_gt.max()); diff = d_sct - d_gt
    hi = bb & (d_gt > 0.1 * dmax)
    passrate = 100.0 * float((np.abs(diff[hi]) <= 0.03 * dmax).mean())
    print("PTV Dmean gt=%.2f sct=%.2f Gy (delta %.2f%% Rx) | D95 gt=%.2f sct=%.2f"
          % (d_gt[tt].mean(), d_sct[tt].mean(), 100 * diff[tt].mean() / RX,
             np.percentile(d_gt[tt], 5), np.percentile(d_sct[tt], 5)))
    print("body MAE=%.4f Gy max|delta|=%.3f Gy (%.1f%% Dmax) | 3%%-dose-diff pass=%.2f%%"
          % (np.abs(diff[bb]).mean(), np.abs(diff[bb]).max(), 100 * np.abs(diff[bb]).max() / dmax, passrate))

    # ---- 8. visualization --------------------------------------------------
    sl = int(np.clip(c[2], 0, d_gt.shape[0] - 1))
    hu_s = np.transpose(hu, (2, 1, 0))[sl]
    sct_s = np.transpose(sct, (2, 1, 0))[sl]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(hu_s, cmap="gray", vmin=-1000, vmax=1000); ax[0, 0].set_title("gtCT (HU)")
    ax[0, 1].imshow(hu_s, cmap="gray", vmin=-1000, vmax=1000)
    im1 = ax[0, 1].imshow(np.ma.masked_less(d_gt[sl], 0.05 * dmax), cmap="jet", alpha=0.6, vmin=0, vmax=dmax)
    ax[0, 1].set_title("dose on gtCT (optimized)"); fig.colorbar(im1, ax=ax[0, 1], fraction=0.046)
    ax[0, 2].imshow(hu_s, cmap="gray", vmin=-1000, vmax=1000)
    im2 = ax[0, 2].imshow(np.ma.masked_less(d_uniform[sl], 0.05 * max(d_uniform.max(), 1e-6)), cmap="jet", alpha=0.6)
    ax[0, 2].set_title("dose on gtCT (uniform/open field)"); fig.colorbar(im2, ax=ax[0, 2], fraction=0.046)
    ax[1, 0].imshow(sct_s, cmap="gray", vmin=-1000, vmax=1000)
    im3 = ax[1, 0].imshow(np.ma.masked_less(d_sct[sl], 0.05 * dmax), cmap="jet", alpha=0.6, vmin=0, vmax=dmax)
    ax[1, 0].set_title("dose on proxy sCT (same fluence)"); fig.colorbar(im3, ax=ax[1, 0], fraction=0.046)
    dd = np.ma.masked_where(~bb[sl], diff[sl])
    im4 = ax[1, 1].imshow(dd, cmap="bwr", vmin=-0.05 * dmax, vmax=0.05 * dmax)
    ax[1, 1].set_title("dose diff (sCT - gtCT)"); fig.colorbar(im4, ax=ax[1, 1], fraction=0.046)
    for nm, msk, col in [("PTV", tt, "tab:red"), ("Body", bb, "tab:gray")]:
        x, y = dvh(d_gt, msk); ax[1, 2].plot(x, y, col, label=f"{nm} gtCT")
        x, y = dvh(d_sct, msk); ax[1, 2].plot(x, y, col, ls="--", label=f"{nm} sCT")
    ax[1, 2].set_xlabel("Dose (Gy)"); ax[1, 2].set_ylabel("Volume (%)")
    ax[1, 2].set_title("DVH: gtCT (solid) vs sCT (dashed)"); ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)
    for a in ax.flat[:5]:
        a.axis("off")
    fig.suptitle("%s  pyRadPlan sCT-vs-gtCT dose eval  |  HU-sens MAE=%.4f  body MAE=%.4f Gy  3%%pass=%.1f%%"
                 % (SUBJ, sens, float(np.abs(diff[bb]).mean()), passrate))
    fig.tight_layout()
    outpng = os.path.join(OUT, f"{SUBJ}_sct_dose_eval.png")
    fig.savefig(outpng, dpi=110)
    print("SAVED_FIG", outpng)
    print("PASS: end-to-end sCT-vs-gtCT dose eval complete")


if __name__ == "__main__":
    main()
