"""Uniform Track-A + Hard-Dice scoring for the UNet perceptual-loss ablation.

A trimmed sibling of score_all_models.py for the 3 UNet variants (no perceptual,
perceptual, perceptual+dice). Reuses the SAME metric functions as full_eval so
the numbers are directly comparable — in particular Hard Dice (compute_dice_hard),
NOT validate.py's soft compute_dice. Track B (SynthRAD-native) is intentionally
dropped: all three variants are the same [-1024,1024]-clipped UNet, so Track B
adds nothing and would only drag in the koalAI nnunetv2 fork.

Per (variant, subject):
  Track A — amix-clip [-1024,1024] (hu_range 2048), GPU fused_ssim3d:
    mae_hu / psnr / ssim                  full volume
    body_mae_hu / body_psnr / body_ssim   zero-out body mask
  Hard Dice — Baby-UNet teacher on the prediction, argmax labels vs GT ct_seg:
    dice_score_all / dice_score_bone      (compute_dice_hard, bone_idx=5)

GT CT, body mask and seg all come from the dataset (DATA_ROOT), exactly as in
score_all_models.py, so GT handling is identical. Prediction is read from
<raw_dir>/<tag>/<subj>/sample.nii.gz (validate.py output, HU).

Emits <out_dir>/{per_subject.csv, by_region.csv, overall.csv}.

Usage (GPU node):
    python src/evaluate/score_perc_ablation.py \
        --raw_dir /gpfs/.../perc_ablation_20260603/raw \
        --tags 9xmodnhn_ep400 ye820cq0_ep400 06e850ny_ep400 \
        --out_dir /gpfs/.../perc_ablation_20260603/metrics
"""
import argparse
import csv
import glob
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import get_region_key, get_split_subjects  # noqa: E402
from common.utils import compute_metrics, compute_metrics_body  # noqa: E402
from common.eval_utils import compute_dice_hard, load_teacher_model, run_teacher_sw  # noqa: E402

DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
TEACHER = "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth"
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]
METRIC_COLS = ["mae_hu", "psnr", "ssim", "body_mae_hu", "body_psnr", "body_ssim",
               "dice_score_all", "dice_score_bone"]


def load_ras(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj, dtype=np.float32).squeeze()


def to_t(arr, device):
    return torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None].to(device)


def hu_to_amix01(hu):
    return (np.clip(hu, -1024.0, 1024.0) + 1024.0) / 2048.0


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def score_subject(tag, subj, raw_dir, device, teacher, val_patch_size, do_dice):
    sample_p = os.path.join(raw_dir, tag, subj, "sample.nii.gz")
    target_p = _first_existing(os.path.join(DATA_ROOT, subj, "ct.nii"),
                               os.path.join(DATA_ROOT, subj, "ct.nii.gz"))
    mask_p = _first_existing(os.path.join(DATA_ROOT, subj, "mask.nii"),
                             os.path.join(DATA_ROOT, subj, "mask.nii.gz"))
    if not (os.path.exists(sample_p) and target_p and mask_p):
        return None, "missing file"

    pred_hu = load_ras(sample_p)
    gt_hu = load_ras(target_p)
    mask = (load_ras(mask_p) > 0.5).astype(np.float32)

    # Guard against a normalized (non-HU) prediction slipping through.
    if pred_hu.min() >= -50.0 and pred_hu.max() <= 2.5:
        return None, (f"sample looks NORMALIZED not HU (min={pred_hu.min():.2f} max={pred_hu.max():.2f})")
    if not (pred_hu.shape == gt_hu.shape == mask.shape):
        return None, f"shape mismatch pred{pred_hu.shape} gt{gt_hu.shape} mask{mask.shape}"

    rec = {"model": tag, "subj_id": subj, "region": get_region_key(subj)}

    pred01 = to_t(hu_to_amix01(pred_hu), device)
    gt01 = to_t(hu_to_amix01(gt_hu), device)
    mask_t = to_t(mask, device)
    mA = compute_metrics(pred01, gt01, hu_range=2048)
    rec.update(mae_hu=mA["mae_hu"], psnr=mA["psnr"], ssim=mA["ssim"])
    mB = compute_metrics_body(pred01, gt01, mask_t, hu_range=2048)
    rec.update(body_mae_hu=mB["mae_hu"], body_psnr=mB["psnr"], body_ssim=mB["ssim"])

    if do_dice:
        seg_p = _first_existing(os.path.join(DATA_ROOT, subj, "ct_seg.nii"),
                                os.path.join(DATA_ROOT, subj, "ct_seg.nii.gz"))
        if seg_p:
            seg = load_ras(seg_p)
            if seg.shape == pred_hu.shape:
                seg_t = torch.from_numpy(np.ascontiguousarray(seg)).long()[None, None].to(device)
                logits = run_teacher_sw(teacher, pred01, device=device,
                                        val_patch_size=val_patch_size, sw_batch_size=2, overlap=0.25)
                rec.update(compute_dice_hard(logits, seg_t, bone_idx=5))
                del logits, seg_t

    del pred01, gt01, mask_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec, None


def stats(subset, col):
    vals = [r[col] for r in subset if col in r and r[col] is not None and not np.isnan(r[col])]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)


def aggregate(rows, tags, out_dir):
    with open(os.path.join(out_dir, "by_region.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "region", "n"] + [f"{c}_mean" for c in METRIC_COLS])
        for tag in tags:
            for region in REGIONS:
                sub = [r for r in rows if r["model"] == tag and r["region"] == region]
                if not sub:
                    continue
                w.writerow([tag, region, len(sub)] + [f"{stats(sub, c)[0]:.6f}" for c in METRIC_COLS])

    with open(os.path.join(out_dir, "overall.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "agg", "n"] + [f"{c}_mean" for c in METRIC_COLS] +
                   [f"{c}_std" for c in METRIC_COLS])
        for tag in tags:
            sub = [r for r in rows if r["model"] == tag]
            if not sub:
                continue
            micro = [stats(sub, c) for c in METRIC_COLS]
            w.writerow([tag, "micro", len(sub)] + [f"{m:.6f}" for m, _, _ in micro] +
                       [f"{s:.6f}" for _, s, _ in micro])
            macro = []
            for c in METRIC_COLS:
                rmeans = [stats([r for r in sub if r["region"] == reg], c)[0] for reg in REGIONS]
                rmeans = [x for x in rmeans if not np.isnan(x)]
                macro.append(np.mean(rmeans) if rmeans else float("nan"))
            w.writerow([tag, "macro", len(sub)] + [f"{m:.6f}" for m in macro] + ["" for _ in METRIC_COLS])
    print(f"[score] wrote by_region/overall.csv to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="dir holding <tag>/<subj>/sample.nii.gz")
    ap.add_argument("--tags", nargs="+", required=True, help="variant subdir names under raw_dir")
    ap.add_argument("--out_dir", default=None, help="default: <raw_dir>/../metrics")
    ap.add_argument("--split_file", default="splits/center_wise_split.txt")
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--val_patch_size", type=int, default=256)
    ap.add_argument("--no_dice", action="store_true")
    ap.add_argument("--max_subjects", type=int, default=None, help="Smoke: only first N subjects.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    do_dice = not args.no_dice
    subjects = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.raw_dir.rstrip("/")), "metrics")
    os.makedirs(out_dir, exist_ok=True)

    present = {t: {os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.raw_dir, t, "*", "sample.nii.gz"))}
               for t in args.tags}
    print("[score] per-variant volume coverage:")
    for t in args.tags:
        miss = sorted(set(subjects) - present[t])
        print(f"   {t:18} {len(present[t] & set(subjects)):3}/{len(subjects)}"
              + (f"  MISSING {len(miss)}: {miss[:6]}{'...' if len(miss) > 6 else ''}" if miss else "  OK"))
    common = sorted(set(subjects).intersection(*present.values()))
    if len(common) < len(subjects):
        print(f"[score] ⚠️  scoring COMMON set of {len(common)}/{len(subjects)} so every variant has equal n.")
    if not common:
        print("[score] no common subjects — aborting.")
        return

    teacher = load_teacher_model(TEACHER, device=device, n_classes_minus_bg=11) if do_dice else None

    ps_path = os.path.join(out_dir, "per_subject.csv")
    ps_f = open(ps_path, "w", newline="")
    ps_w = csv.writer(ps_f)
    ps_w.writerow(["model", "subj_id", "region"] + METRIC_COLS)

    rows = []
    for tag in args.tags:
        t0 = time.time()
        ok = skip = 0
        for subj in common:
            try:
                rec, err = score_subject(tag, subj, args.raw_dir, device, teacher,
                                         args.val_patch_size, do_dice)
            except RuntimeError as e:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rec, err = None, f"RuntimeError: {str(e)[:120]}"
            if rec is None:
                skip += 1
                print(f"  [skip] {tag}/{subj}: {err}")
                continue
            rows.append(rec)
            ps_w.writerow([rec["model"], rec["subj_id"], rec["region"]]
                          + [f"{rec[c]:.6f}" if c in rec and rec[c] is not None else "" for c in METRIC_COLS])
            ps_f.flush()
            ok += 1
            if ok % 25 == 0:
                print(f"  {tag}: {ok}/{len(common)} (last bMAE={rec['body_mae_hu']:.1f})", flush=True)
        print(f"[score] {tag}: {ok} ok, {skip} skipped, {time.time()-t0:.0f}s", flush=True)
    ps_f.close()
    aggregate(rows, args.tags, out_dir)
    print(f"[score] DONE -> {ps_path}")


if __name__ == "__main__":
    main()
