"""Report 11 per-model heavy extraction pass (one pass over all 207 subjects).

Fuses, for a single model, the per-volume passes that report 10 split across
cads_extract.py, cads_mask_split.py, important/cads_bone_hu_split.py and
verify_density.py. Every accumulation rule is copied verbatim from those scripts
so the `unet` outputs reproduce report 10 exactly.

  python mm_extract.py --model <unet|amix|maisi|cwdm|mcddpm|koalAI>

Writes to OUTROOT/<model>/:
  cads_per_label.csv   per (subj,label): n, sum|err|, sum err, sum GT, sum pred, n(GT>1024)
  cads_subject.csv     per subject: body MAE + rim audit + tight-body external/internal split
  cads_calib.npz       pooled GT-vs-pred 2D histograms (unlabeled/soft/airorg/bone)
  cads_bone_hu_split.csv  within-bone error by GT-HU density band
  verify_density.csv   bone localization AUC + HU decomposition + threshold-recovery Dice
"""
import os
import zlib
import json
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.ndimage import binary_erosion, binary_fill_holes, label
from sklearn.metrics import roc_auc_score

import mm_common as C

MODEL = "unet"          # set in main() before the Pool is forked
TS = np.arange(-100, 700, 10)   # bone-threshold sweep (verify_density)


def tight_body(gt, body):
    """largest connected non-air blob with internal gas filled = the real patient."""
    t = (gt > -300) & body
    lab, n = label(t)
    if n == 0:
        return t
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    t = lab == big
    for z in range(t.shape[2]):
        t[:, :, z] = binary_fill_holes(t[:, :, z])
    return t


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def process(s):
    try:
        gt = C.canon(f"{C.DATA}/{s}/ct.nii")
        sct = C.canon(C.sct_path(MODEL, s))
        bodyv = C.canon(f"{C.DATA}/{s}/mask.nii") > 0
        seg = C.canon(f"{C.DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception as e:
        return None, None, None, None, None, f"{s}: {e}"

    rim_mask = bodyv & ~binary_erosion(bodyv, iterations=3)
    tb = tight_body(gt, bodyv)

    g, p, sg = gt[bodyv], sct[bodyv], seg[bodyv]
    rim = rim_mask[bodyv]
    tbm = tb[bodyv]
    err = p - g
    ae = np.abs(err)

    # ---- per-label micro sums (cads_extract) ----
    rows = []
    for l in range(C.NL):
        m = sg == l
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "subj": s, "region": C.reg(s), "label": l, "name": C.CLASS_NAMES[l],
            "is_bone": l in C.BONE, "n": n,
            "sabs": float(ae[m].sum()), "serr": float(err[m].sum()),
            "sgt": float(g[m].sum()), "spred": float(p[m].sum()),
            "n_gt1024": int((g[m] > 1024).sum()),
        })

    # ---- body-mask audit (cads_extract) + tight-body external/internal (cads_mask_split) ----
    air = g < -300
    lab0 = sg == 0
    ext = lab0 & ~tbm          # label-0 outside the patient (loose mask)
    intr = lab0 & tbm          # label-0 inside the patient (internal gas)
    sub = {
        "subj": s, "region": C.reg(s), "n_body": int(len(g)),
        "body_mae": float(ae.mean()),
        "n_air": int(air.sum()), "n_lab0": int(lab0.sum()),
        "n_lab0_air": int((lab0 & air).sum()),
        "n_rim": int(rim.sum()),
        "n_lab0_air_rim": int((lab0 & air & rim).sum()),
        "n_lab0_air_int": int((lab0 & air & ~rim).sum()),
        "sabs_lab0_air_rim": float(ae[lab0 & air & rim].sum()),
        "sabs_lab0_air_int": float(ae[lab0 & air & ~rim].sum()),
        "sabs_lab0": float(ae[lab0].sum()),
        # tight-body split
        "sabs_body": float(ae.sum()),
        "n_ext": int(ext.sum()), "n_int": int(intr.sum()),
        "sabs_ext": float(ae[ext].sum()), "sabs_int": float(ae[intr].sum()),
    }

    # ---- 2D calibration hists per group (cads_extract) ----
    grp = np.zeros(len(g), np.int8)
    grp[np.isin(sg, list(range(1, C.NL)))] = 1
    grp[np.isin(sg, C.AIRORG)] = 2
    grp[np.isin(sg, C.BONE)] = 3
    H = {}
    for gi in range(4):
        mm = grp == gi
        H[gi] = (np.histogram2d(g[mm], p[mm], bins=[C.EDG, C.EDP])[0]
                 if mm.any() else np.zeros((len(C.EDG) - 1, len(C.EDP) - 1)))

    # ---- within-bone HU-band split (cads_bone_hu_split) ----
    bone = np.isin(sg, C.BONE)
    gtb, aeb, errb = g[bone], ae[bone], err[bone]
    band = np.digitize(gtb, C.HU_EDGES[1:-1])
    bn = np.bincount(band, minlength=C.NHU)
    bsabs = np.bincount(band, weights=aeb, minlength=C.NHU)
    bserr = np.bincount(band, weights=errb, minlength=C.NHU)
    bb = np.concatenate([[len(g), float(ae.sum())], bn, bsabs, bserr])

    # ---- verify_density: localization AUC + decomposition + threshold recovery ----
    vd = None
    if bone.sum() >= 200 and (~bone).sum() >= 200:
        rng = np.random.RandomState(zlib.crc32(s.encode()) & 0x7fffffff)
        bi = np.where(bone)[0]
        ni = np.where(~bone)[0]
        bi = rng.choice(bi, min(40000, len(bi)), replace=False)
        ni = rng.choice(ni, min(40000, len(ni)), replace=False)
        y = np.concatenate([np.ones(len(bi)), np.zeros(len(ni))])
        auc_sct = roc_auc_score(y, np.concatenate([p[bi], p[ni]]))
        auc_gt = roc_auc_score(y, np.concatenate([g[bi], g[ni]]))
        pb = p[bone]
        d_real = dice(g > 150, bone)
        d_sct150 = dice(p > 150, bone)
        dices = [dice(p > t, bone) for t in TS]
        bi_best = int(np.nanargmax(dices))
        vd = {
            "subj": s, "region": C.reg(s), "n_bone": int(bone.sum()),
            "auc_sct": auc_sct, "auc_gt": auc_gt,
            "frac_rendered_soft": float((pb < 50).mean()),
            "frac_undershoot": float(((pb >= 50) & (pb < 150)).mean()),
            "frac_correct": float((pb >= 150).mean()),
            "dice_real_t150": d_real, "dice_sct_t150": d_sct150,
            "dice_sct_best": float(dices[bi_best]), "t_best": float(TS[bi_best]),
        }

    return rows, sub, H, bb, vd, None


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.MODELS)
    args = ap.parse_args()
    MODEL = args.model
    C.ensure(MODEL)
    RUN = C.run_dir(MODEL)

    subs = C.subjects()
    print(f"[mm_extract:{MODEL}] {len(subs)} subjects", flush=True)
    L, S, V, errs = [], [], [], []
    Hsum = {gi: np.zeros((len(C.EDG) - 1, len(C.EDP) - 1)) for gi in range(4)}
    BB = np.zeros(2 + 3 * C.NHU)
    with Pool(8) as pool:
        for i, (rows, sub, H, bb, vd, e) in enumerate(pool.imap_unordered(process, subs)):
            if e:
                errs.append(e)
                continue
            L.extend(rows)
            S.append(sub)
            if vd:
                V.append(vd)
            for gi in range(4):
                Hsum[gi] += H[gi]
            BB += bb
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(subs)}", flush=True)

    pd.DataFrame(L).to_csv(os.path.join(RUN, "cads_per_label.csv"), index=False)
    pd.DataFrame(S).to_csv(os.path.join(RUN, "cads_subject.csv"), index=False)
    pd.DataFrame(V).to_csv(os.path.join(RUN, "verify_density.csv"), index=False)
    np.savez(os.path.join(RUN, "cads_calib.npz"), gt_edges=C.EDG, pred_edges=C.EDP,
             unlabeled=Hsum[0], soft=Hsum[1], airorg=Hsum[2], bone=Hsum[3])

    # bone HU-band table
    body_n, body_sabs = BB[0], BB[1]
    n = BB[2:2 + C.NHU]
    sabs = BB[2 + C.NHU:2 + 2 * C.NHU]
    serr = BB[2 + 2 * C.NHU:2 + 3 * C.NHU]
    bone_n, bone_sabs = n.sum(), sabs.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        bone_tbl = pd.DataFrame({
            "GT-HU band": C.HU_BANDS,
            "pct_body_vox": 100 * n / body_n,
            "pct_bone_vox": 100 * n / bone_n,
            "micro_mae": sabs / n,
            "bias": serr / n,
            "pct_body_error": 100 * sabs / body_sabs,
            "pct_bone_error": 100 * sabs / bone_sabs,
        })
    bone_tbl.to_csv(os.path.join(RUN, "cads_bone_hu_split.csv"), index=False)
    json.dump({"bone_pct_body_vox": float(100 * bone_n / body_n),
               "bone_pct_body_error": float(100 * bone_sabs / body_sabs)},
              open(os.path.join(RUN, "cads_bone_hu_meta.json"), "w"), indent=2)

    print(f"[mm_extract:{MODEL}] done: {len(S)} ok, {len(errs)} err, {len(V)} vd", flush=True)
    for e in errs[:5]:
        print("  ", e)


if __name__ == "__main__":
    main()
