"""Minimal example: optimize a plan on the gtCT, deliver the SAME plan on the sCT.

REQUIRES THE DEDICATED `pyradplan` ENV (see README.md):
    micromamba run -n pyradplan python pyradplan/plan_transfer_example.py [SUBJECT_ID]

The "plan" = the optimized beamlet weights w (how hard each beamlet fires), computed
ONCE on the ground-truth CT against a prescription (PTV -> RX Gy). We then deliver that
SAME w on both CTs -- rebuilding the dose-influence matrix D from each CT's HU, because
the radiation physics depends on tissue density. dose_sct - dose_gt is therefore purely
the effect of the sCT's HU error on the delivered dose.

No real sCT on disk yet -> we fake one (bone HU biased -250) as a stand-in. Replace the
`sct = ...` line with `nib.load(<your_sct>).get_fdata()` to use a real prediction.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "1THB008"
ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
OUT = os.path.join(os.path.dirname(__file__), "out")
SPACING = 1.5
GANTRY = [0, 72, 144, 216, 288]   # 5 beams
RX = 60.0                          # prescription dose to the PTV (Gy)


def to_sitk(arr_xyz, cast=np.float32):
    a = np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0))).astype(cast)
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((SPACING,) * 3)
    return img


def build_problem(hu, body, target):
    """CT + structures(+objectives) + beams -> (ct, cst, pln, stf, dij) for this HU volume."""
    from pyRadPlan import PhotonPlan, generate_stf, calc_dose_influence
    from pyRadPlan.ct import create_ct
    from pyRadPlan.cst import create_voi, create_cst
    from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

    ct = create_ct(cube_hu=to_sitk(hu))
    cst = create_cst([
        create_voi(voi_type="EXTERNAL", name="Body", ct_image=ct, mask=to_sitk(body, np.uint8),
                   objectives=[SquaredOverdosing(parameters=[30.0], priority=10.0)]),
        create_voi(voi_type="TARGET", name="PTV", ct_image=ct, mask=to_sitk(target, np.uint8),
                   objectives=[SquaredDeviation(parameters=[RX], priority=800.0)]),
    ], ct=ct)
    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {"gantry_angles": GANTRY, "couch_angles": [0] * len(GANTRY)}
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    return ct, cst, pln, stf, dij


def dose_of(dij, w):
    return sitk.GetArrayFromImage(dij.compute_result_ct_grid(np.asarray(w, np.float32).ravel())["physical_dose"])


def main():
    from pyRadPlan import fluence_optimization
    from pyRadPlan.core import xp_utils as xp
    xp.PREFER_GPU = False
    os.makedirs(OUT, exist_ok=True)

    p = f"{ROOT}/{SUBJ}/"
    hu = nib.load(p + "ct.nii").get_fdata().astype(np.float32)
    body = nib.load(p + "mask.nii").get_fdata().astype(np.uint8)

    # synthetic spherical PTV at the body centroid (TotalSeg has no tumor target)
    c = [int(v.mean()) for v in np.nonzero(body > 0)]
    X, Y, Z = np.indices(body.shape)
    target = ((((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= 20**2).astype(np.uint8)) & body
    print("SUBJ %s shape=%s PTV_vox=%d beams=%s RX=%.0f Gy" % (SUBJ, hu.shape, int(target.sum()), GANTRY, RX))

    # ---- 1. OPTIMIZE on the ground-truth CT -> the plan (weights w) --------
    ct_gt, cst, pln, stf, dij_gt = build_problem(hu, body, target)
    w = np.asarray(fluence_optimization(ct_gt, cst, stf, dij_gt, pln)).ravel()
    print("PLAN optimized on gtCT: %d beamlets" % w.size)

    # ---- 2. deliver the SAME w on gtCT and on the (proxy) sCT --------------
    dose_gt = dose_of(dij_gt, w)

    sct = hu.copy(); sct[(hu > 150) & (body > 0)] -= 250.0     # <-- replace with real sCT
    _, _, _, _, dij_sct = build_problem(sct, body, target)
    dose_sct = dose_of(dij_sct, w)                              # SAME plan, sCT physics

    # ---- 3. compare --------------------------------------------------------
    tt = np.transpose(target, (2, 1, 0)).astype(bool)
    bb = np.transpose(body, (2, 1, 0)).astype(bool)
    diff = dose_sct - dose_gt
    dmax = float(dose_gt.max())
    print("PTV  Dmean gt=%.2f sct=%.2f Gy (delta %.2f%% Rx)" % (dose_gt[tt].mean(), dose_sct[tt].mean(), 100*diff[tt].mean()/RX))
    print("body MAE=%.4f Gy  max|delta|=%.3f Gy (%.1f%% Dmax)" % (np.abs(diff[bb]).mean(), np.abs(diff[bb]).max(), 100*np.abs(diff[bb]).max()/dmax))

    # ---- 4. figure: gtCT | sCT | dose on gtCT | dose on sCT | dose diff -----
    sl = c[2]
    hu_t = np.transpose(hu, (2, 1, 0))
    sct_t = np.transpose(sct, (2, 1, 0))
    # tight diff range: 99th-pct of |diff| inside the body (ignores rare outliers)
    dlim = float(np.percentile(np.abs(diff[bb]), 99))
    dlim = max(dlim, 1e-3)
    print("dose-diff colour range = +/- %.3f Gy (99th pct of |diff| in body)" % dlim)

    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    # 1-2: raw CTs, no dose
    ax[0].imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000)
    ax[0].contour(tt[sl], colors="lime", linewidths=0.8); ax[0].set_title("gtCT (HU)"); ax[0].axis("off")
    ax[1].imshow(sct_t[sl], cmap="gray", vmin=-1000, vmax=1000)
    ax[1].contour(tt[sl], colors="lime", linewidths=0.8); ax[1].set_title("proxy sCT (HU)"); ax[1].axis("off")
    # 3-4: dose on each CT
    for a, d, ttl in [(ax[2], dose_gt, "dose on gtCT"), (ax[3], dose_sct, "dose on sCT (same plan)")]:
        a.imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000)
        im = a.imshow(np.ma.masked_less(d[sl], 0.05*dmax), cmap="jet", alpha=0.6, vmin=0, vmax=dmax)
        a.contour(tt[sl], colors="lime", linewidths=0.8); a.set_title(ttl); a.axis("off")
    fig.colorbar(im, ax=ax[3], fraction=0.046, label="dose (Gy)")
    # 5: dose diff with tight range
    dd = np.ma.masked_where(~bb[sl], diff[sl])
    im2 = ax[4].imshow(dd, cmap="bwr", vmin=-dlim, vmax=dlim)
    ax[4].contour(tt[sl], colors="lime", linewidths=0.8)
    ax[4].set_title("dose diff (sCT - gtCT), +/-%.2f Gy" % dlim); ax[4].axis("off")
    fig.colorbar(im2, ax=ax[4], fraction=0.046, label="Gy")
    fig.suptitle("%s  optimize-on-gtCT, deliver-same-plan-on-sCT  |  body MAE=%.4f Gy" % (SUBJ, float(np.abs(diff[bb]).mean())))
    fig.tight_layout()
    outpng = os.path.join(OUT, f"{SUBJ}_plan_transfer.png")
    fig.savefig(outpng, dpi=115); print("SAVED", outpng); print("PASS")


if __name__ == "__main__":
    main()
