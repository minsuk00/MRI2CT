"""Qualitative figures for full_eval: one representative subject per region,
columns = MRI | GT CT | each of the 6 models, rows = axial slices.

Reads from the unified volume tree (volumes/<model>/<subj>/sample.nii.gz) and the
per-subject metrics CSV (metrics/per_subject.csv) for captions. Per region, the
representative subject is the one whose amix body-PSNR is the median across the
region (stable, not cherry-picked) unless overridden with --subjects.

Usage:
    python src/evaluate/visualize_full_eval.py --eval_root /gpfs/.../full_eval_20260601
"""
import argparse
import csv
import os
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.data import get_region_key, get_split_subjects  # noqa: E402

SYNTHRAD_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
MODELS = ["amix", "unet", "maisi", "mcddpm", "cwdm", "koalAI", "amix_bw4", "unet_bw4"]
DISPLAY = {"amix": "Amix", "unet": "UNet", "maisi": "MAISI", "mcddpm": "MC-DDPM",
           "cwdm": "cWDM", "koalAI": "KoalAI",
           "amix_bw4": "Amix·b.4", "unet_bw4": "UNet·b.4"}
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]
HU_WINDOWS = {"brain": (-100, 100), "head_neck": (-100, 100),
              "abdomen": (-1024, 1024), "thorax": (-1024, 1024), "pelvis": (-1024, 1024)}


def load_per_subject(eval_root):
    """metrics/per_subject.csv → {(model,subj): {metric: float}}."""
    out = {}
    p = os.path.join(eval_root, "metrics", "per_subject.csv")
    if not os.path.exists(p):
        return out
    with open(p) as f:
        for row in csv.DictReader(f):
            d = {}
            for k, v in row.items():
                if k in ("model", "subj_id", "region"):
                    continue
                try:
                    d[k] = float(v)
                except (ValueError, TypeError):
                    d[k] = None
            out[(row["model"], row["subj_id"])] = d
    return out


def load_vol(path):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


def pick_subject(region, subjects, metrics):
    """Median-amix-PSNR subject in the region (fallback: first available)."""
    cand = [s for s in subjects if get_region_key(s) == region]
    scored = [(metrics.get(("amix", s), {}).get("psnr"), s) for s in cand]
    scored = [(p, s) for p, s in scored if p is not None]
    if not scored:
        return cand[0] if cand else None
    scored.sort()
    return scored[len(scored) // 2][1]


def caption(m):
    if not m:
        return "(n/a)"
    def g(k, fmt):
        v = m.get(k)
        return fmt.format(v) if v is not None else "-"
    return (f"PSNR {g('psnr','{:.1f}')} / SSIM {g('ssim','{:.3f}')}\n"
            f"MAE {g('body_mae_hu','{:.0f}')}HU  HardBone {g('dice_score_bone','{:.3f}')}\n"
            f"SR-MAE {g('synthrad_mae','{:.0f}')} MS-SSIM {g('synthrad_ms_ssim','{:.3f}')}")


def render(region, subj, eval_root, metrics, num_slices, out_dir):
    vmin, vmax = HU_WINDOWS.get(region, (-1000, 1000))
    mri_p = sorted(glob(os.path.join(SYNTHRAD_ROOT, subj, "moved_mr*.nii*")))
    if not mri_p:
        print(f"[viz] {region}/{subj}: no MRI"); return None
    mri = load_vol(mri_p[0])

    gt_p = os.path.join(eval_root, "volumes", "amix", subj, "target.nii.gz")
    if not os.path.exists(gt_p):
        cand = glob(os.path.join(eval_root, "volumes", "*", subj, "target.nii.gz"))
        gt_p = cand[0] if cand else None
    if not gt_p:
        print(f"[viz] {region}/{subj}: no GT target"); return None
    gt = load_vol(gt_p)

    cols = [("MRI", mri, "mri", None), ("GT CT", gt, "ct", None)]
    for mdl in MODELS:
        sp = os.path.join(eval_root, "volumes", mdl, subj, "sample.nii.gz")
        if not os.path.exists(sp):
            cols.append((DISPLAY[mdl], None, "ct", None)); continue
        cols.append((DISPLAY[mdl], load_vol(sp), "ct", metrics.get((mdl, subj))))

    D = gt.shape[-1]
    zs = np.linspace(0.15 * D, 0.85 * D, num_slices, dtype=int)
    nr, nc = num_slices, len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc, 3.2 * nr), squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.08, top=0.9)
    mvmin, mvmax = float(np.percentile(mri, 1)), float(np.percentile(mri, 99))

    for r, z in enumerate(zs):
        for c, (name, vol, kind, m) in enumerate(cols):
            ax = axes[r, c]; ax.axis("off")
            if vol is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9);
            elif kind == "mri":
                ax.imshow(np.rot90(vol[:, :, min(z, vol.shape[-1]-1)]), cmap="gray", vmin=mvmin, vmax=mvmax)
            else:
                ax.imshow(np.rot90(vol[:, :, min(z, vol.shape[-1]-1)]), cmap="gray", vmin=vmin, vmax=vmax)
            if r == 0:
                ttl = name if (kind != "ct" or m is None) else f"{name}\n{caption(m)}"
                ax.set_title(ttl, fontsize=8)
    fig.suptitle(f"{region}  —  {subj}   |   CT window [{vmin}, {vmax}] HU", fontsize=13, y=0.965)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{region}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"[viz] wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--split_file", default="splits/center_wise_split.txt")
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--num_slices", type=int, default=4)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Explicit subject IDs (one per region) instead of median pick.")
    args = ap.parse_args()

    subjects = get_split_subjects(args.split_file, args.split_name)
    metrics = load_per_subject(args.eval_root)
    out_dir = os.path.join(args.eval_root, "figures")

    # Restrict columns to models that actually have volumes in this eval root, so a
    # subset eval (e.g. just unet + koalAI) doesn't render blank model columns.
    global MODELS
    MODELS = [m for m in MODELS
              if glob(os.path.join(args.eval_root, "volumes", m, "*", "sample.nii.gz"))]

    picks = {}
    if args.subjects:
        for s in args.subjects:
            picks[get_region_key(s)] = s
    for region in REGIONS:
        subj = picks.get(region) or pick_subject(region, subjects, metrics)
        if subj:
            render(region, subj, args.eval_root, metrics, args.num_slices, out_dir)


if __name__ == "__main__":
    main()
