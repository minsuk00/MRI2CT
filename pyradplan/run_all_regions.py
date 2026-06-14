"""Run uniform-fluence + plan-transfer dose calc on one subject per region.

REQUIRES THE DEDICATED `pyradplan` ENV (see README.md):
    micromamba run -n pyradplan python pyradplan/run_all_regions.py

For each region (abdomen/thorax/head&neck/pelvis/brain) it picks one subject and writes
into pyradplan/out/<region>/:
  - <subj>_openfield_sphere.png   : uniform fluence (no plan), synthetic sphere PTV
  - <subj>_plan_transfer.png      : optimize on gtCT, deliver SAME plan on proxy sCT (5 panels)
Reuses the functions from openfield_dose.py and plan_transfer_example.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import openfield_dose as of
import plan_transfer_example as pt
from pyRadPlan import fluence_optimization
from pyRadPlan.core import xp_utils as xp

ROOT = of.ROOT
OUT = os.path.join(os.path.dirname(__file__), "out")
GANTRY = [0, 72, 144, 216, 288]
RX = 60.0

# one subject per region (2nd char of ID: A=abdomen, T=thorax, H=head&neck, P=pelvis, B=brain)
REGIONS = {
    "abdomen": "1ABA005",
    "thorax": "1THA001",
    "head_and_neck": "1HNA001",
    "pelvis": "1PA001",
    "brain": "1BA001",
}


def load(subj):
    p = f"{ROOT}/{subj}/"
    hu = nib.load(p + "ct.nii").get_fdata().astype(np.float32)
    body = nib.load(p + "mask.nii").get_fdata().astype(np.uint8)
    c = [int(v.mean()) for v in np.nonzero(body > 0)]
    X, Y, Z = np.indices(body.shape)
    sphere = ((((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= 20**2).astype(np.uint8)) & body
    return hu, body, sphere, c


def run_uniform(subj, hu, body, sphere, outdir):
    dose, nb = of.uniform_dose(hu, sphere, GANTRY, body=body)   # body VOI -> corridors computed
    out = os.path.join(outdir, f"{subj}_openfield_sphere.png")
    of.plot_dose(hu, sphere, dose,
                 f"{subj}  UNIFORM fluence, target=SPHERE(r=20)  beams={GANTRY}  nbeamlets={nb}", out)
    print("  uniform: nbeamlets=%d dmax=%.2f -> %s" % (nb, float(dose.max()), os.path.basename(out)))


def run_plan_transfer(subj, hu, body, target, c, outdir):
    # 1. optimize on gtCT -> plan weights w
    ct_gt, cst, pln, stf, dij_gt = pt.build_problem(hu, body, target)
    w = np.asarray(fluence_optimization(ct_gt, cst, stf, dij_gt, pln)).ravel()
    # 2. deliver SAME w on gtCT and proxy sCT (bone -250 HU)
    dose_gt = pt.dose_of(dij_gt, w)
    sct = hu.copy(); sct[(hu > 150) & (body > 0)] -= 250.0
    _, _, _, _, dij_sct = pt.build_problem(sct, body, target)
    dose_sct = pt.dose_of(dij_sct, w)
    # 3. metrics
    tt = np.transpose(target, (2, 1, 0)).astype(bool)
    bb = np.transpose(body, (2, 1, 0)).astype(bool)
    diff = dose_sct - dose_gt
    dmax = float(dose_gt.max())
    dlim = max(float(np.percentile(np.abs(diff[bb]), 99)), 1e-3)
    mae = float(np.abs(diff[bb]).mean())
    # 4. 5-panel figure
    sl = c[2]
    hu_t = np.transpose(hu, (2, 1, 0)); sct_t = np.transpose(sct, (2, 1, 0))
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    ax[0].imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000); ax[0].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[0].set_title("gtCT (HU)"); ax[0].axis("off")
    ax[1].imshow(sct_t[sl], cmap="gray", vmin=-1000, vmax=1000); ax[1].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[1].set_title("proxy sCT (HU)"); ax[1].axis("off")
    for a, d, ttl in [(ax[2], dose_gt, "dose on gtCT"), (ax[3], dose_sct, "dose on sCT (same plan)")]:
        a.imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000)
        im = a.imshow(np.ma.masked_less(d[sl], 0.05*dmax), cmap="jet", alpha=0.6, vmin=0, vmax=dmax)
        a.contour(tt[sl], colors="lime", linewidths=0.8); a.set_title(ttl); a.axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="dose (Gy)")
    dd = np.ma.masked_where(~bb[sl], diff[sl])
    im2 = ax[4].imshow(dd, cmap="bwr", vmin=-dlim, vmax=dlim)
    ax[4].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[4].set_title("dose diff (sCT-gtCT), +/-%.2f Gy" % dlim); ax[4].axis("off")
    fig.colorbar(im2, ax=ax[4], fraction=0.046, label="Gy")
    fig.suptitle("%s  optimize-on-gtCT, deliver-same-plan-on-sCT  |  body MAE=%.4f Gy" % (subj, mae))
    fig.tight_layout()
    out = os.path.join(outdir, f"{subj}_plan_transfer.png")
    fig.savefig(out, dpi=115); plt.close(fig)
    print("  plan_transfer: nbeamlets=%d body MAE=%.4f Gy diff-range +/-%.3f -> %s"
          % (w.size, mae, dlim, os.path.basename(out)))


def main():
    xp.PREFER_GPU = False
    for region, subj in REGIONS.items():
        outdir = os.path.join(OUT, region)
        os.makedirs(outdir, exist_ok=True)
        print(f"=== {region}: {subj} ===")
        hu, body, sphere, c = load(subj)
        print("  shape=%s HU=[%.0f,%.0f] body_vox=%d sphere_vox=%d"
              % (hu.shape, hu.min(), hu.max(), int(body.sum()), int(sphere.sum())))
        run_uniform(subj, hu, body, sphere, outdir)
        run_plan_transfer(subj, hu, body, sphere, c, outdir)
    print("ALL DONE")


if __name__ == "__main__":
    main()
