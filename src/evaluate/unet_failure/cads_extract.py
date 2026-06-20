"""Report 10 extraction: decompose the U-Net sCT error using the GT CADS 35-label
segmentation ONLY (no HU thresholds for tissue, no babyseg). Everything is
full-range raw HU, error = sCT - GT, restricted to the body mask, and accumulated
as MICRO sums so per-label / per-group MAE reconstructs the body-voxel MAE exactly.

Per subject we also audit the body mask: how much of "body" is unlabeled (CADS=0),
how much is air-HU, and how much sits in the eroded rim (proxy for loose-mask
external air) vs the interior (internal gas).

Outputs (to RUN/):
  cads_per_label.csv  - one row per (subject,label): n, sum|err|, sum err, sum GT, sum pred, n(GT>1024)
  cads_subject.csv    - per subject: body MAE + mask-audit counts
  cads_calib.npz      - pooled GT-vs-pred 2D histograms for bone / air-organ / soft / unlabeled
"""
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import binary_erosion

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
NL = 35
BONE = [7, 27, 28, 29, 30]
AIRORG = [9, 13]
CLASS_NAMES = [
    "Background", "Brain - other", "CSF", "Eyes & optic pathway", "Face & oral soft tissue",
    "Gray matter", "Head & neck glands", "Skull", "White matter", "Airway", "Breast",
    "Esophagus", "Heart", "Lungs", "Thoracic cavity", "Abdominal cavity", "Adrenals",
    "Bowel", "Gallbladder", "Kidneys", "Liver", "Pancreas", "Spleen", "Stomach", "Bladder",
    "Prostate & seminal vesicle", "Blood vessels", "Bone - other", "Limb & girdle bones",
    "Spine", "Thoracic cage", "Gland - other", "Muscle", "Spinal cord", "Subcutaneous tissue",
]
EDG = np.linspace(-1024, 3000, 202)            # GT axis (full range)
EDP = np.linspace(-1024, 1100, 108)            # pred axis (sigmoid-bounded)


def reg(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def process(s):
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        bodyv = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception as e:
        return None, None, None, f"{s}: {e}"

    rim_mask = bodyv & ~binary_erosion(bodyv, iterations=3)   # outer shell of the mask
    g, p, sg = gt[bodyv], sct[bodyv], seg[bodyv]
    rim = rim_mask[bodyv]
    err = p - g
    ae = np.abs(err)

    rows = []
    for l in range(NL):
        m = sg == l
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "subj": s, "region": reg(s), "label": l, "name": CLASS_NAMES[l],
            "is_bone": l in BONE, "n": n,
            "sabs": float(ae[m].sum()), "serr": float(err[m].sum()),
            "sgt": float(g[m].sum()), "spred": float(p[m].sum()),
            "n_gt1024": int((g[m] > 1024).sum()),
        })

    # body-mask audit
    air = g < -300
    lab0 = sg == 0
    sub = {
        "subj": s, "region": reg(s), "n_body": int(len(g)),
        "body_mae": float(ae.mean()),
        "n_air": int(air.sum()),                       # all air-HU in body
        "n_lab0": int(lab0.sum()),
        "n_lab0_air": int((lab0 & air).sum()),
        "n_rim": int(rim.sum()),
        "n_lab0_air_rim": int((lab0 & air & rim).sum()),
        "n_lab0_air_int": int((lab0 & air & ~rim).sum()),
        "sabs_lab0_air_rim": float(ae[lab0 & air & rim].sum()),
        "sabs_lab0_air_int": float(ae[lab0 & air & ~rim].sum()),
        "sabs_lab0": float(ae[lab0].sum()),
    }

    # 2D calibration hists per group
    grp = np.zeros(len(g), np.int8)                    # 0 unlabeled,1 soft,2 airorg,3 bone
    grp[np.isin(sg, list(range(1, NL)))] = 1
    grp[np.isin(sg, AIRORG)] = 2
    grp[np.isin(sg, BONE)] = 3
    H = {}
    for gi in range(4):
        mm = grp == gi
        H[gi] = np.histogram2d(g[mm], p[mm], bins=[EDG, EDP])[0] if mm.any() else np.zeros((len(EDG) - 1, len(EDP) - 1))
    return rows, sub, H, None


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    print(f"[cads_extract] {len(subs)} subjects", flush=True)
    L, S, errs = [], [], []
    Hsum = {gi: np.zeros((len(EDG) - 1, len(EDP) - 1)) for gi in range(4)}
    with Pool(8) as pool:
        for i, (rows, sub, H, e) in enumerate(pool.imap_unordered(process, subs)):
            if e:
                errs.append(e)
                continue
            L.extend(rows)
            S.append(sub)
            for gi in range(4):
                Hsum[gi] += H[gi]
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(subs)}", flush=True)
    pd.DataFrame(L).to_csv(os.path.join(RUN, "cads_per_label.csv"), index=False)
    pd.DataFrame(S).to_csv(os.path.join(RUN, "cads_subject.csv"), index=False)
    np.savez(os.path.join(RUN, "cads_calib.npz"), gt_edges=EDG, pred_edges=EDP,
             unlabeled=Hsum[0], soft=Hsum[1], airorg=Hsum[2], bone=Hsum[3])
    print(f"[cads_extract] done, {len(S)} ok, {len(errs)} err", flush=True)
    for e in errs[:5]:
        print("  ", e)


if __name__ == "__main__":
    main()
