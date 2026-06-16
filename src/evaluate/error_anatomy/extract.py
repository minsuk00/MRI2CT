"""Per-subject error-anatomy extraction for MR->CT models.

For each (model, subject): load the RAW ground-truth CT (full HU, up to ~3000),
the 35-label CADS segmentation, the body mask, and the saved model prediction
(`sample.nii.gz`, HU). Compute a battery of decompositions:
  - body MAE (raw GT and clipped-to-1024 GT for a sanity check vs per_subject.csv)
  - tissue split by GT HU: air (<-300), soft (-300..200), bone (>200)
  - bone breakdown: mid-bone (200..1024) vs cortical (>1024), MAE + signed bias
    (undershoot), MAE-mass contribution, predicted-HU distribution, cap pileup
  - bone interior (eroded) vs boundary shell -> localization-vs-magnitude proxy
  - per 35-label structure: MAE, signed bias, voxel count, GT/pred mean HU, frac>1024

Outputs two CSVs under the run dir: `summary.csv` (model x subject) and
`structures.csv` (model x subject x label).

Single GT for every model = raw dataset ct.nii (NOT each model's clipped target).
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA_ROOT = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
VOL_ROOT = os.path.join(REPO, "evaluation_results/full_eval_20260609/volumes")
PER_SUBJECT_CSV = os.path.join(REPO, "evaluation_results/full_eval_20260609/metrics/per_subject.csv")
MODELS = ["unet", "amix", "maisi", "mcddpm"]

BONE_LABELS = {7: "skull", 27: "bone_other", 28: "limb_girdle", 29: "spine", 30: "thoracic_cage"}
LABEL_NAMES = {
    0: "background", 1: "brain_other", 2: "csf", 3: "eyes", 4: "face_oral", 5: "gray_matter",
    6: "hn_glands", 7: "skull", 8: "white_matter", 9: "airway", 10: "breast", 11: "esophagus",
    12: "heart", 13: "lungs", 14: "thoracic_cavity", 15: "abdominal_cavity", 16: "adrenals",
    17: "bowel", 18: "gallbladder", 19: "kidneys", 20: "liver", 21: "pancreas", 22: "spleen",
    23: "stomach", 24: "bladder", 25: "prostate_sv", 26: "blood_vessels", 27: "bone_other",
    28: "limb_girdle", 29: "spine", 30: "thoracic_cage", 31: "gland_other", 32: "muscle",
    33: "spinal_cord", 34: "subcutaneous",
}

AIR_HI = -300.0    # HU < -300  -> air
SOFT_HI = 200.0    # -300..200  -> soft tissue ; >200 -> bone
CAP = 1024.0       # cortical bone threshold / regressor sigmoid cap


def _load(p, dtype=np.float32):
    # Eval predictions are saved in canonical RAS; the raw dataset CT/seg/mask are in
    # the original (often LPS-like) orientation. as_closest_canonical applies a pure
    # axis flip/permute (no interpolation -> HU and integer labels preserved) so every
    # volume lands on the same RAS grid as the predictions. No-op for already-RAS preds.
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dtype)


def _safe_mean(a):
    return float(a.mean()) if a.size else np.nan


def process_subject(args):
    subj, region = args
    sdir = os.path.join(DATA_ROOT, subj)
    try:
        gt = _load(os.path.join(sdir, "ct.nii"))
        seg = _load(os.path.join(sdir, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        body = _load(os.path.join(sdir, "mask.nii")) > 0
    except Exception as e:
        return [], [], f"{subj}: GT load fail {e}"

    if not (gt.shape == seg.shape == body.shape):
        return [], [], f"{subj}: shape mismatch gt{gt.shape} seg{seg.shape} mask{body.shape}"

    gtb = gt[body]
    air_m = gtb < AIR_HI
    soft_m = (gtb >= AIR_HI) & (gtb <= SOFT_HI)
    bone_m = gtb > SOFT_HI
    cort_m = gtb > CAP
    mid_m = (gtb > SOFT_HI) & (gtb <= CAP)

    # bone interior vs boundary (localization vs magnitude proxy), computed on the full grid
    bone_full = body & (gt > SOFT_HI)
    interior_full = ndimage.binary_erosion(bone_full, iterations=1)
    boundary_full = bone_full & ~interior_full
    int_b = interior_full[body]
    bnd_b = boundary_full[body]

    # present structures (within body)
    segb = seg[body]
    present = [int(v) for v in np.unique(segb) if v != 0]

    summ_rows, struct_rows = [], []
    for model in MODELS:
        pp = os.path.join(VOL_ROOT, model, subj, "sample.nii.gz")
        if not os.path.exists(pp):
            continue
        try:
            pred = _load(pp)
        except Exception as e:
            struct_rows  # noqa
            summ_rows.append({"model": model, "subj_id": subj, "region": region, "error": f"pred load {e}"})
            continue
        if pred.shape != gt.shape:
            summ_rows.append({"model": model, "subj_id": subj, "region": region,
                              "error": f"pred shape {pred.shape}"})
            continue

        pb = pred[body]
        err = pb - gtb
        aerr = np.abs(err)

        row = {
            "model": model, "subj_id": subj, "region": region,
            "n_body": int(body.sum()), "n_total": int(gt.size),
            "mae_raw": _safe_mean(aerr),
            "mae_clip": _safe_mean(np.abs(np.clip(pb, -1024, 1024) - np.clip(gtb, -1024, 1024))),
            # tissue split
            "mae_air": _safe_mean(aerr[air_m]), "bias_air": _safe_mean(err[air_m]), "n_air": int(air_m.sum()),
            "mae_soft": _safe_mean(aerr[soft_m]), "bias_soft": _safe_mean(err[soft_m]), "n_soft": int(soft_m.sum()),
            "mae_bone": _safe_mean(aerr[bone_m]), "bias_bone": _safe_mean(err[bone_m]), "n_bone": int(bone_m.sum()),
            # bone magnitude breakdown
            "mae_midbone": _safe_mean(aerr[mid_m]), "bias_midbone": _safe_mean(err[mid_m]), "n_midbone": int(mid_m.sum()),
            "mae_cortical": _safe_mean(aerr[cort_m]), "bias_cortical": _safe_mean(err[cort_m]), "n_cortical": int(cort_m.sum()),
            # mass contribution of cortical (>1024) to total bone error
            "bone_err_sum": float(aerr[bone_m].sum()),
            "cortical_err_sum": float(aerr[cort_m].sum()),
            # predicted-HU distribution inside true bone
            "pred_bone_mean": _safe_mean(pb[bone_m]),
            "pred_bone_p95": float(np.percentile(pb[bone_m], 95)) if bone_m.any() else np.nan,
            "pred_bone_max": float(pb[bone_m].max()) if bone_m.any() else np.nan,
            "gt_bone_mean": _safe_mean(gtb[bone_m]),
            "gt_cortical_frac": float(cort_m.sum()) / max(int(bone_m.sum()), 1),
            # cap pileup: fraction of bone voxels where pred is pinned at the +1024 ceiling
            "pred_capped_frac": float((pb[bone_m] >= 1023).mean()) if bone_m.any() else np.nan,
            # localization proxy
            "mae_bone_interior": _safe_mean(aerr[int_b]), "n_bone_interior": int(int_b.sum()),
            "mae_bone_boundary": _safe_mean(aerr[bnd_b]), "n_bone_boundary": int(bnd_b.sum()),
        }
        summ_rows.append(row)

        for lab in present:
            m = segb == lab
            n = int(m.sum())
            if n < 50:
                continue
            struct_rows.append({
                "model": model, "subj_id": subj, "region": region,
                "label": lab, "name": LABEL_NAMES.get(lab, str(lab)),
                "is_bone": lab in BONE_LABELS, "n": n,
                "mae": _safe_mean(aerr[m]), "bias": _safe_mean(err[m]),
                "gt_mean": _safe_mean(gtb[m]), "pred_mean": _safe_mean(pb[m]),
                "gt_frac_gt1024": float((gtb[m] > CAP).mean()),
            })

    return summ_rows, struct_rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "evaluation_results/unet_error_analysis_20260616"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N subjects")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(PER_SUBJECT_CSV)
    subs = (df[df.model == "unet"][["subj_id", "region"]]
            .drop_duplicates().values.tolist())
    if args.limit:
        subs = subs[:args.limit]
    print(f"[extract] {len(subs)} subjects x {len(MODELS)} models, {args.workers} workers", flush=True)

    summ_all, struct_all, errs = [], [], []
    with Pool(args.workers) as pool:
        for i, (sr, st, err) in enumerate(pool.imap_unordered(process_subject, subs)):
            if err:
                errs.append(err)
            summ_all.extend(sr)
            struct_all.extend(st)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(subs)} subjects done", flush=True)

    summ = pd.DataFrame(summ_all)
    struct = pd.DataFrame(struct_all)
    summ.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    struct.to_csv(os.path.join(args.out, "structures.csv"), index=False)
    print(f"[extract] wrote summary.csv ({len(summ)} rows), structures.csv ({len(struct)} rows)", flush=True)
    if errs:
        print(f"[extract] {len(errs)} errors:")
        for e in errs[:20]:
            print("   ", e)

    # ---- correctness gate ----
    # mae_raw is the body-voxel mean over ~raw GT == CSV synthrad_mae (body-voxels-only, [-1024,3000]).
    # body_mae_hu is the full-volume zeroed mean: body_mae_hu == mae_clip * n_body/n_total.
    valid = summ[summ.get("mae_raw").notna()] if "mae_raw" in summ.columns else summ.iloc[:0]
    if len(valid):
        m = valid.merge(df[["model", "subj_id", "synthrad_mae", "body_mae_hu"]], on=["model", "subj_id"], how="left")
        m["d_raw"] = (m["mae_raw"] - m["synthrad_mae"]).abs()
        m["body_recompute"] = m["mae_clip"] * m["n_body"] / m["n_total"]
        m["d_body"] = (m["body_recompute"] - m["body_mae_hu"]).abs()
        print("[gate] |mae_raw - synthrad_mae| by model (should be ~0):")
        print(m.groupby("model")["d_raw"].agg(["mean", "max"]).to_string())
        print("[gate] |recomputed body_mae - csv body_mae_hu| by model (should be ~0):")
        print(m.groupby("model")["d_body"].agg(["mean", "max"]).to_string())


if __name__ == "__main__":
    main()
