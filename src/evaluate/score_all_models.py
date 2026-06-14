"""Uniform metric scoring for full_eval — all 6 models through identical code.

Reads the unified per-model volumes (volumes/<model>/<subj>/{sample,target}.nii.gz),
the dataset body mask (mask.nii) and GT seg (ct_seg.nii), and computes for every
(model, subject):

  Track A — amix-clip [-1024, 1024] (hu_range 2048), GPU fused_ssim3d:
    mae_hu / psnr / ssim                   full volume   (compute_metrics)
    body_mae_hu / body_psnr / body_ssim    zero-out mask (compute_metrics_body)
    dice_score_all / dice_score_bone       Baby U-Net teacher, pred fed as [0,1]-from-[-1024,1024]

  Track B — SynthRAD-native [-1024, 3000] (official koalAI ImageMetrics):
    synthrad_mae / synthrad_psnr / synthrad_ms_ssim   body-voxels-only (sum|.|*mask / mask.sum)

Emits metrics/{per_subject.csv, by_region.csv, overall.csv} and cross-checks koalAI
Track-B against the precomputed evaluation_results/koalai_native/fold0 summary.

Run on a GPU node (fused_ssim3d + teacher SW are CUDA). Track B is CPU (numpy/skimage).

Usage:
    python src/evaluate/score_all_models.py --eval_root /gpfs/.../full_eval_20260601
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "baselines", "koalAI")))

from common.data import get_region_key, get_split_subjects  # noqa: E402
from common.utils import compute_metrics, compute_metrics_body  # noqa: E402
from common.eval_utils import compute_dice_hard, load_teacher_model, run_teacher_sw  # noqa: E402
from nnunetv2.analysis.image_metrics import ImageMetricsCompute  # noqa: E402

DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
TEACHER = "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth"
KOALAI_NATIVE = "/home/minsukc/MRI2CT/evaluation_results/koalai_native/fold0"
ALL_MODELS = ["amix", "unet", "maisi", "mcddpm", "cwdm", "koalAI"]
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]


def load_ras(path):
    """Load a NIfTI, reorient to closest-canonical (RAS), return float32 ndarray (squeezed)."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj, dtype=np.float32).squeeze()


def to_t(arr, device):
    return torch.from_numpy(np.ascontiguousarray(arr)).float()[None, None].to(device)


def hu_to_amix01(hu):
    """HU → [0,1] over [-1024,1024] (the amix yardstick), clipped."""
    return (np.clip(hu, -1024.0, 1024.0) + 1024.0) / 2048.0


def mean_time_per_volume(raw_model_dir):
    """Mean of per-subject time_sec parsed from validate_metrics.txt (flat or sharded)."""
    txts = glob.glob(os.path.join(raw_model_dir, "validate_metrics.txt"))
    txts += glob.glob(os.path.join(raw_model_dir, "shard_*_of_*", "validate_metrics.txt"))
    times = []
    for t in txts:
        with open(t) as f:
            lines = f.readlines()
        try:
            hi = next(i for i, l in enumerate(lines) if l.startswith("subj_id"))
        except StopIteration:
            continue
        cols = lines[hi].split()[1:]
        if "time_sec" not in cols:
            continue
        ti = cols.index("time_sec")
        for l in lines[hi + 1:]:
            l = l.strip()
            if not l or l.startswith("="):
                break
            parts = l.split()
            if len(parts) > ti + 1:
                try:
                    times.append(float(parts[ti + 1]))
                except ValueError:
                    pass
    return (float(np.mean(times)), len(times)) if times else (float("nan"), 0)


def score_subject(model, subj, vol_dir, device, im, teacher, val_patch_size, do_dice):
    sample_p = os.path.join(vol_dir, model, subj, "sample.nii.gz")
    # Common ground truth for ALL models = the raw, full-HU dataset CT (not each
    # model's own differently-clipped target.nii.gz). This makes Track B fair to
    # full-range bone (amix/unet/cwdm/maisi clip their saved target at ~1024/1650).
    # clip(raw, -1024, 1024) reproduces amix's [-1024,1024] target exactly, so
    # Track A is unchanged.
    target_p = os.path.join(DATA_ROOT, subj, "ct.nii")
    if not os.path.exists(target_p):
        target_p = os.path.join(DATA_ROOT, subj, "ct.nii.gz")
    mask_p = os.path.join(DATA_ROOT, subj, "mask.nii")
    if not os.path.exists(mask_p):
        mask_p = os.path.join(DATA_ROOT, subj, "mask.nii.gz")
    if not (os.path.exists(sample_p) and os.path.exists(target_p) and os.path.exists(mask_p)):
        return None, "missing file"

    pred_hu = load_ras(sample_p)
    gt_hu = load_ras(target_p)
    mask = (load_ras(mask_p) > 0.5).astype(np.float32)

    # Guard: every model's sample must be in HU, not [0,1] normalized. A real CT with a
    # body always has soft tissue/bone well above ~2 HU and air near -1000; a volume confined
    # to [-1,2] means it was saved normalized (the cWDM-style bug) and would silently produce
    # garbage metrics here. Refuse it loudly instead.
    if pred_hu.min() >= -50.0 and pred_hu.max() <= 2.5:
        return None, (f"sample looks NORMALIZED not HU (min={pred_hu.min():.2f} max={pred_hu.max():.2f}) "
                      "— convert to HU before scoring")
    if not (pred_hu.shape == gt_hu.shape == mask.shape):
        return None, f"shape mismatch pred{pred_hu.shape} gt{gt_hu.shape} mask{mask.shape}"

    rec = {"model": model, "subj_id": subj, "region": get_region_key(subj)}

    # ---- Track A (GPU) ----
    pred01 = to_t(hu_to_amix01(pred_hu), device)
    gt01 = to_t(hu_to_amix01(gt_hu), device)
    mask_t = to_t(mask, device)
    mA = compute_metrics(pred01, gt01, hu_range=2048)
    rec.update(mae_hu=mA["mae_hu"], psnr=mA["psnr"], ssim=mA["ssim"])
    mB = compute_metrics_body(pred01, gt01, mask_t, hu_range=2048)
    rec.update(body_mae_hu=mB["mae_hu"], body_psnr=mB["psnr"], body_ssim=mB["ssim"])

    # ---- Bone Dice (GPU) ----
    if do_dice:
        seg_p = os.path.join(DATA_ROOT, subj, "ct_seg.nii")
        if not os.path.exists(seg_p):
            seg_p = os.path.join(DATA_ROOT, subj, "ct_seg.nii.gz")
        if os.path.exists(seg_p):
            seg = load_ras(seg_p)
            if seg.shape == pred_hu.shape:
                seg_t = torch.from_numpy(np.ascontiguousarray(seg)).long()[None, None].to(device)
                logits = run_teacher_sw(teacher, pred01, device=device,
                                        val_patch_size=val_patch_size, sw_batch_size=2, overlap=0.25)
                d = compute_dice_hard(logits, seg_t, bone_idx=5)
                rec.update(d)
                del logits, seg_t

    # ---- Track B (CPU, official SynthRAD metric) ----
    if mask.sum() < 1:
        return None, "empty body mask (mask.sum()==0)"
    tb = im.score_array(gt_hu, pred_hu, mask)
    if any(v is None or np.isnan(v) for v in (tb["mae"], tb["psnr"], tb["ms_ssim"])):
        return None, f"Track-B NaN (degenerate mask?) mae={tb['mae']} psnr={tb['psnr']} ms_ssim={tb['ms_ssim']}"
    rec.update(synthrad_mae=tb["mae"], synthrad_psnr=tb["psnr"], synthrad_ms_ssim=tb["ms_ssim"])

    del pred01, gt01, mask_t
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rec, None


def aggregate(rows, out_dir):
    metric_cols = ["mae_hu", "psnr", "ssim", "body_mae_hu", "body_psnr", "body_ssim",
                   "dice_score_all", "dice_score_bone",
                   "synthrad_mae", "synthrad_psnr", "synthrad_ms_ssim"]
    # per_subject.csv is written incrementally by main(); here we only roll up.

    def stats(subset, col):
        vals = [r[col] for r in subset if col in r and r[col] is not None and not np.isnan(r[col])]
        return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)

    # by_region.csv
    with open(os.path.join(out_dir, "by_region.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "region", "n"] + [f"{c}_mean" for c in metric_cols])
        for model in ALL_MODELS:
            for region in REGIONS:
                sub = [r for r in rows if r["model"] == model and r["region"] == region]
                if not sub:
                    continue
                w.writerow([model, region, len(sub)] + [f"{stats(sub, c)[0]:.6f}" for c in metric_cols])

    # overall.csv — micro (mean over all subjects) + macro (mean over per-region means)
    with open(os.path.join(out_dir, "overall.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "agg", "n"] + [f"{c}_mean" for c in metric_cols] +
                   [f"{c}_std" for c in metric_cols])
        for model in ALL_MODELS:
            sub = [r for r in rows if r["model"] == model]
            if not sub:
                continue
            micro = [stats(sub, c) for c in metric_cols]
            w.writerow([model, "micro", len(sub)] + [f"{m:.6f}" for m, _, _ in micro] +
                       [f"{s:.6f}" for _, s, _ in micro])
            macro = []
            for c in metric_cols:
                rmeans = [stats([r for r in sub if r["region"] == reg], c)[0] for reg in REGIONS]
                rmeans = [x for x in rmeans if not np.isnan(x)]
                macro.append(np.mean(rmeans) if rmeans else float("nan"))
            w.writerow([model, "macro", len(sub)] + [f"{m:.6f}" for m in macro] +
                       ["" for _ in metric_cols])
    print(f"[score] wrote by_region/overall.csv to {out_dir}")


def crosscheck_koalai(rows):
    print("\n[score] koalAI Track-B cross-check vs koalai_native/fold0:")
    import pandas as pd
    for region in REGIONS:
        csvp = os.path.join(KOALAI_NATIVE, region, "results_individual.csv")
        if not os.path.exists(csvp):
            print(f"   {region}: native csv missing"); continue
        native = pd.read_csv(csvp)
        nat_mae = native["mae"].mean(); nat_psnr = native["psnr"].mean()
        ours = [r for r in rows if r["model"] == "koalAI" and r["region"] == region]
        if not ours:
            print(f"   {region}: no scored koalAI rows"); continue
        our_mae = np.mean([r["synthrad_mae"] for r in ours])
        our_psnr = np.mean([r["synthrad_psnr"] for r in ours])
        print(f"   {region:10} MAE ours={our_mae:7.2f} native={nat_mae:7.2f} (Δ{our_mae-nat_mae:+.2f}) | "
              f"PSNR ours={our_psnr:6.2f} native={nat_psnr:6.2f} (Δ{our_psnr-nat_psnr:+.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--split_file", default="splits/center_wise_split.txt")
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS)
    ap.add_argument("--val_patch_size", type=int, default=256)
    ap.add_argument("--no_dice", action="store_true")
    ap.add_argument("--max_subjects", type=int, default=None, help="Smoke: only first N subjects.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    do_dice = not args.no_dice
    subjects = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
    vol_dir = os.path.join(args.eval_root, "volumes")
    out_dir = os.path.join(args.eval_root, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    # --- Coverage check: score the COMMON subject set so every model has identical n ---
    present = {m: {os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(vol_dir, m, "*", "sample.nii.gz"))}
               for m in args.models}
    print("[score] per-model volume coverage:")
    for m in args.models:
        miss = sorted(set(subjects) - present[m])
        print(f"   {m:8} {len(present[m] & set(subjects)):3}/{len(subjects)}"
              + (f"  MISSING {len(miss)}: {miss[:6]}{'...' if len(miss) > 6 else ''}" if miss else "  OK"))
    common = sorted(set(subjects).intersection(*present.values()))
    if len(common) < len(subjects):
        print(f"[score] ⚠️  Only {len(common)}/{len(subjects)} subjects present for ALL models "
              f"({', '.join(args.models)}). Scoring the COMMON set so every model has identical n. "
              f"If this is unexpected, generation is incomplete — re-run when all models have 207.")
    score_subjects = common
    if not score_subjects:
        print("[score] no common subjects to score — aborting.")
        return

    im = ImageMetricsCompute()
    im.names = ["mae", "psnr", "ms_ssim"]
    teacher = load_teacher_model(TEACHER, device=device, n_classes_minus_bg=11) if do_dice else None

    metric_cols = ["mae_hu", "psnr", "ssim", "body_mae_hu", "body_psnr", "body_ssim",
                   "dice_score_all", "dice_score_bone",
                   "synthrad_mae", "synthrad_psnr", "synthrad_ms_ssim"]
    # Incremental per_subject.csv flush so an OOM/crash mid-run keeps completed rows.
    ps_path = os.path.join(out_dir, "per_subject.csv")
    ps_f = open(ps_path, "w", newline="")
    ps_w = csv.writer(ps_f)
    ps_w.writerow(["model", "subj_id", "region"] + metric_cols)

    rows = []
    for model in args.models:
        t0 = time.time()
        ok = skip = 0
        for subj in score_subjects:
            try:
                rec, err = score_subject(model, subj, vol_dir, device, im, teacher,
                                         args.val_patch_size, do_dice)
            except RuntimeError as e:  # OOM or CUDA error → skip this subject, keep going
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                rec, err = None, f"RuntimeError: {str(e)[:120]}"
            if rec is None:
                skip += 1
                print(f"  [skip] {model}/{subj}: {err}")
                continue
            rows.append(rec)
            ps_w.writerow([rec["model"], rec["subj_id"], rec["region"]]
                          + [f"{rec[c]:.6f}" if c in rec and rec[c] is not None else "" for c in metric_cols])
            ps_f.flush()
            ok += 1
            if ok % 25 == 0:
                print(f"  {model}: {ok}/{len(score_subjects)} "
                      f"(last MAE_A={rec['mae_hu']:.1f} synthrad_MAE={rec['synthrad_mae']:.1f})", flush=True)
        mt, nt = mean_time_per_volume(os.path.join(args.eval_root, "raw", model))
        print(f"[score] {model}: scored {ok}, skipped {skip}, {time.time()-t0:.0f}s; "
              f"raw gen-time/vol={mt:.1f}s NO-SYNC (n={nt}; NOT the report figure)", flush=True)
    ps_f.close()

    aggregate(rows, out_dir)
    if "koalAI" in args.models:
        try:
            crosscheck_koalai(rows)
        except Exception as e:
            print(f"[score] koalAI cross-check failed: {e}")

    # Raw generation-time summary (parsed from each model's validate_metrics.txt).
    # NOTE: these are wall-clock times from the ORIGINAL volume generation, which ran
    # the pre-fix validators WITHOUT torch.cuda.synchronize() — so they under-count the
    # fast (non-diffusion) models by up to ~16x. They are written here only for
    # reference and are deliberately NOT named inference_time.csv: the report's
    # `inference_time.csv` is the separately re-measured, CUDA-synchronized timing and
    # must not be clobbered by this scorer (it would otherwise overwrite good numbers
    # every chain run). See sbatch/fulleval_time.sh.
    with open(os.path.join(out_dir, "generation_time_raw.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "mean_gen_time_sec_per_volume_NO_SYNC", "n"])
        for model in args.models:
            mt, nt = mean_time_per_volume(os.path.join(args.eval_root, "raw", model))
            w.writerow([model, f"{mt:.3f}" if not np.isnan(mt) else "", nt])
    print("[score] done.")


if __name__ == "__main__":
    main()
