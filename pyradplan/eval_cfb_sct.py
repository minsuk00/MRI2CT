"""Dose eval on CFB-GBM with the REAL GTV as target (not a synthetic sphere).
Optimize a photon plan on gtCT -> deliver SAME plan on each sCT -> MAE_dose/gamma/PTV-DVH.

    micromamba run -n pyradplan python pyradplan/eval_cfb_sct.py <unet|amix> [SEQ ...]

unet -> reads sct_<seq>.nii.gz ; amix -> reads sct_amix_<seq>.nii.gz (all 1.5mm RAS).
Figure slice = GTV centroid (where the target+dose actually are). CT window [-100,100] HU.
"""
import os, sys, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import nibabel as nib, numpy as np
import plan_transfer_example as pt
import dose_metrics as dm
from pyRadPlan import fluence_optimization
from pyRadPlan.core import xp_utils as xp

D = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/external_inference/cfb_gbm/001_t0"
RX = 60.0
MODEL = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("unet", "amix") else "unet"
PFX = "sct_amix_" if MODEL == "amix" else "sct_"
arg_seqs = [a for a in sys.argv[1:] if a not in ("unet", "amix")]
SEQS = arg_seqs or sorted(os.path.basename(f)[len(PFX):-7] for f in glob.glob(f"{D}/{PFX}*.nii.gz")
                          if "amix" not in os.path.basename(f) or MODEL == "amix")


def main():
    xp.PREFER_GPU = False
    gt   = nib.load(f"{D}/ct.nii").get_fdata().astype(np.float32)
    body = nib.load(f"{D}/mask.nii").get_fdata().astype(np.uint8)
    gtv  = (nib.load(f"{D}/gtv.nii").get_fdata() > 0.5).astype(np.uint8) & body
    print("MODEL=%s seqs=%s gtv=%.1fcm3" % (MODEL, SEQS, gtv.sum()*1.5**3/1000))

    ct_gt, cst, pln, stf, dij_gt = pt.build_problem(gt, body, gtv)
    w = np.asarray(fluence_optimization(ct_gt, cst, stf, dij_gt, pln)).ravel()
    dose_gt = pt.dose_of(dij_gt, w)
    print("plan: %d beamlets dmax_gt=%.2f Gy" % (w.size, dose_gt.max()))

    tt = np.transpose(gtv, (2, 1, 0)).astype(bool)
    bb = np.transpose(body, (2, 1, 0)).astype(bool)
    rows = []
    for s in SEQS:
        f = f"{D}/{PFX}{s}.nii.gz"
        if not os.path.exists(f): print("  [skip]", f); continue
        sct = nib.load(f).get_fdata().astype(np.float32)
        _, _, _, _, dij_sct = pt.build_problem(sct, body, gtv)
        dose_sct = pt.dose_of(dij_sct, w)
        mae = dm.mae_dose(dose_gt, dose_sct, RX, threshold=0.9)
        gamma = dm.gamma_pass_rate(dose_gt, dose_sct, spacing_mm=1.5, dose_pct=2.0, dist_mm=2.0)
        d98g, d98s = np.percentile(dose_gt[tt], 2), np.percentile(dose_sct[tt], 2)
        v95g = (dose_gt[tt] >= 0.95*RX).mean()*100; v95s = (dose_sct[tt] >= 0.95*RX).mean()*100
        rows.append((s, mae, gamma, d98g, d98s, v95g, v95s, dose_sct))
        print("=== %s/%s ===  MAE_dose=%.4f  gamma2%%/2mm=%.1f%%  PTV D98 gt=%.1f sct=%.1f Gy  V95 gt=%.0f sct=%.0f %%"
              % (MODEL, s, mae, gamma, d98g, d98s, v95g, v95s))

    if not rows: print("no sCT evaluated"); return
    # slice = GTV centroid (target + dose live here)
    sl = int(round(np.nonzero(tt)[0].mean()))
    gt_t = np.transpose(gt, (2,1,0)); dmax = float(dose_gt.max())
    n = len(rows); fig, ax = plt.subplots(n+1, 3, figsize=(11, 3.4*(n+1)))
    ax = np.atleast_2d(ax)
    for a in ax.ravel(): a.axis("off")
    def dose_ov(a, d, under):
        a.imshow(under[sl], cmap="gray", vmin=-100, vmax=100)
        return a.imshow(np.ma.masked_less(d[sl], 0.05*dmax), cmap="jet", alpha=.55, vmin=0, vmax=dmax)
    ax[0,0].imshow(gt_t[sl], cmap="gray", vmin=-100, vmax=100); ax[0,0].contour(tt[sl],colors="lime",lw=1.0); ax[0,0].set_title("gtCT + GTV")
    im=dose_ov(ax[0,1], dose_gt, gt_t); ax[0,1].contour(tt[sl],colors="lime",lw=1.0); ax[0,1].set_title("dose on gtCT")
    fig.colorbar(im, ax=ax[0,1], fraction=.046, label="Gy")
    ax[0,2].set_title(f"slice z={sl} (GTV centroid)")
    for i,(s,mae,gamma,d98g,d98s,v95g,v95s,dose_sct) in enumerate(rows):
        r=i+1; sct_t=np.transpose(nib.load(f"{D}/{PFX}{s}.nii.gz").get_fdata(),(2,1,0))
        ax[r,0].imshow(sct_t[sl],cmap="gray",vmin=-100,vmax=100); ax[r,0].contour(tt[sl],colors="lime",lw=1.0); ax[r,0].set_title(f"sCT[{s}]")
        im=dose_ov(ax[r,1], dose_sct, sct_t); ax[r,1].contour(tt[sl],colors="lime",lw=1.0); ax[r,1].set_title(f"dose on sCT[{s}]")
        diff=dose_sct-dose_gt; dlim=max(float(np.percentile(np.abs(diff[bb]),99)),1e-3)
        im2=ax[r,2].imshow(np.ma.masked_where(~bb[sl],diff[sl]),cmap="bwr",vmin=-dlim,vmax=dlim); ax[r,2].contour(tt[sl],colors="lime",lw=1.0)
        ax[r,2].set_title(f"diff +/-{dlim:.1f}Gy  MAE={mae:.3f} gamma={gamma:.0f}%")
        fig.colorbar(im2,ax=ax[r,2],fraction=.046,label="Gy")
    fig.suptitle(f"CFB-GBM 001 t0 — {MODEL} sCT — real GTV, optimize-on-gtCT / deliver-on-sCT (OOD model)")
    fig.tight_layout(); p=f"{D}/cfb_001_dose_{MODEL}.png"; fig.savefig(p,dpi=110); print("SAVED",p,"\nPASS")


if __name__ == "__main__":
    main()
