"""Uniform-fluence transfer: same UNIFORM beams on gtCT and proxy sCT, compare dose.

REQUIRES THE DEDICATED `pyradplan` ENV (see README.md):
    micromamba run -n pyradplan python pyradplan/uniform_transfer.py [SUBJECT_ID]

Like plan_transfer_example.py but with NO optimization: a synthetic sphere target in
the middle, uniform fluence (w = ones), many beams (every 20 deg = 18 beams). Build the
dose-influence matrix D from each CT's HU, apply the SAME uniform w to gtCT and to the
proxy sCT, and compare -> the difference is purely the sCT's HU error (no objectives, no
plan). Writes a 5-panel figure to pyradplan/out/<subj>_uniform_transfer.png.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import openfield_dose as of   # to_sitk, uniform_dose, ROOT

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "1THB008"
OUT = os.path.join(os.path.dirname(__file__), "out")
GANTRY = list(range(0, 360, 20))   # every 20 deg -> 18 beams


def main():
    from pyRadPlan.core import xp_utils as xp
    xp.PREFER_GPU = False
    os.makedirs(OUT, exist_ok=True)

    p = f"{of.ROOT}/{SUBJ}/"
    hu = nib.load(p + "ct.nii").get_fdata().astype(np.float32)
    body = nib.load(p + "mask.nii").get_fdata().astype(np.uint8)
    c = [int(v.mean()) for v in np.nonzero(body > 0)]
    X, Y, Z = np.indices(body.shape)
    sphere = ((((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= 20**2).astype(np.uint8)) & body
    print("SUBJ %s shape=%s sphere_vox=%d beams=%d (every 20deg)" % (SUBJ, hu.shape, int(sphere.sum()), len(GANTRY)))

    # uniform dose on gtCT and on proxy sCT, SAME uniform fluence, SAME sphere+body
    dose_gt, nb = of.uniform_dose(hu, sphere, GANTRY, body=body)
    sct = hu.copy(); sct[(hu > 150) & (body > 0)] -= 250.0     # proxy sCT (bone -250 HU)
    dose_sct, _ = of.uniform_dose(sct, sphere, GANTRY, body=body)
    print("nbeamlets=%d  dmax_gt=%.3f  dmax_sct=%.3f" % (nb, dose_gt.max(), dose_sct.max()))

    # metrics
    tt = np.transpose(sphere, (2, 1, 0)).astype(bool)
    bb = np.transpose(body, (2, 1, 0)).astype(bool)
    diff = dose_sct - dose_gt
    dmax = float(dose_gt.max())
    dlim = max(float(np.percentile(np.abs(diff[bb]), 99)), 1e-3)
    mae = float(np.abs(diff[bb]).mean())
    print("target mean: gt=%.3f sct=%.3f Gy | body MAE=%.4f Gy | diff-range +/-%.3f Gy"
          % (dose_gt[tt].mean(), dose_sct[tt].mean(), mae, dlim))

    # 5-panel figure
    sl = c[2]
    hu_t = np.transpose(hu, (2, 1, 0)); sct_t = np.transpose(sct, (2, 1, 0))
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    ax[0].imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000); ax[0].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[0].set_title("gtCT (HU)"); ax[0].axis("off")
    ax[1].imshow(sct_t[sl], cmap="gray", vmin=-1000, vmax=1000); ax[1].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[1].set_title("proxy sCT (HU)"); ax[1].axis("off")
    for a, d, ttl in [(ax[2], dose_gt, "uniform dose on gtCT"), (ax[3], dose_sct, "uniform dose on sCT")]:
        a.imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000)
        im = a.imshow(np.ma.masked_less(d[sl], 0.05*dmax), cmap="jet", alpha=0.6, vmin=0, vmax=dmax)
        a.contour(tt[sl], colors="lime", linewidths=0.8); a.set_title(ttl); a.axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="dose (Gy, arb. unit fluence)")
    dd = np.ma.masked_where(~bb[sl], diff[sl])
    im2 = ax[4].imshow(dd, cmap="bwr", vmin=-dlim, vmax=dlim)
    ax[4].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[4].set_title("dose diff (sCT-gtCT), +/-%.2f Gy" % dlim); ax[4].axis("off")
    fig.colorbar(im2, ax=ax[4], fraction=0.046, label="Gy")
    fig.suptitle("%s  UNIFORM fluence transfer (sphere target, %d beams every 20deg)  |  body MAE=%.4f Gy"
                 % (SUBJ, len(GANTRY), mae))
    fig.tight_layout()
    outpng = os.path.join(OUT, f"{SUBJ}_uniform_transfer.png")
    fig.savefig(outpng, dpi=115); print("SAVED", outpng); print("PASS")


if __name__ == "__main__":
    main()
