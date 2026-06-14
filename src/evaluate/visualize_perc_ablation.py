"""Qualitative figures for the UNet perceptual-loss ablation: one subject per
region, columns = MRI | GT CT | the 3 variants, rows = axial slices.

Predictions read from <raw_dir>/<tag>/<subj>/sample.nii.gz; GT CT and MRI from the
dataset. Representative subject per region = median body-MAE (ref tag) from the
partial metrics CSV if present, else the middle of the sorted region list.

Usage:
    python src/evaluate/visualize_perc_ablation.py --eval_root /gpfs/.../perc_ablation_20260603
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.data import get_region_key, get_split_subjects  # noqa: E402

DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
# tag -> display label (order = column order)
VARIANTS = [
    ("9xmodnhn_ep400", "no-perc\n(perc0,dice0.1)"),
    ("06e850ny_ep400", "perceptual\n(perc0.5,dice0.1)"),
    ("ye820cq0_ep400", "perc,no-dice\n(perc0.5,dice0)"),
]
REF_TAG = "9xmodnhn_ep400"
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]
HU_WINDOWS = {"brain": (-100, 100), "head_neck": (-100, 100),
              "abdomen": (-1024, 1024), "thorax": (-1024, 1024), "pelvis": (-1024, 1024)}


def load_vol(path):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


def load_per_subject(eval_root):
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


def pick_subject(region, subjects, metrics):
    cand = sorted(s for s in subjects if get_region_key(s) == region)
    scored = [(metrics.get((REF_TAG, s), {}).get("body_mae_hu"), s) for s in cand]
    scored = [(p, s) for p, s in scored if p is not None]
    if not scored:
        return cand[len(cand) // 2] if cand else None
    scored.sort()
    return scored[len(scored) // 2][1]


def caption(m):
    if not m:
        return ""
    def g(k, fmt):
        v = m.get(k)
        return fmt.format(v) if v is not None else "-"
    cap = f"MAE {g('body_mae_hu','{:.0f}')}HU  SSIM {g('body_ssim','{:.3f}')}"
    if m.get("dice_score_bone") is not None:
        cap += f"\nHardBone {g('dice_score_bone','{:.3f}')}"
    return cap


def first_mri(subj):
    p = sorted(glob(os.path.join(DATA_ROOT, subj, "moved_mr*.nii*")))
    if not p:
        p = sorted(glob(os.path.join(DATA_ROOT, subj, "mr*.nii*")))
    return p[0] if p else None


def render(region, subj, raw_dir, metrics, num_slices, out_dir):
    vmin, vmax = HU_WINDOWS.get(region, (-1000, 1000))
    mri_p = first_mri(subj)
    gt_p = os.path.join(DATA_ROOT, subj, "ct.nii")
    if not os.path.exists(gt_p):
        gt_p = os.path.join(DATA_ROOT, subj, "ct.nii.gz")
    if not mri_p or not os.path.exists(gt_p):
        print(f"[viz] {region}/{subj}: missing MRI/GT"); return None
    mri, gt = load_vol(mri_p), load_vol(gt_p)

    cols = [("MRI", mri, "mri", None), ("GT CT", gt, "ct", None)]
    for tag, lbl in VARIANTS:
        sp = os.path.join(raw_dir, tag, subj, "sample.nii.gz")
        cols.append((lbl, load_vol(sp) if os.path.exists(sp) else None, "ct", metrics.get((tag, subj))))

    D = gt.shape[-1]
    zs = np.linspace(0.2 * D, 0.8 * D, num_slices, dtype=int)
    nr, nc = num_slices, len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc, 3.2 * nr), squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.08, top=0.9)
    mvmin, mvmax = float(np.percentile(mri, 1)), float(np.percentile(mri, 99))

    for r, z in enumerate(zs):
        for c, (name, vol, kind, m) in enumerate(cols):
            ax = axes[r, c]; ax.axis("off")
            if vol is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=9)
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
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="override column variants as 'tag=label' (use \\n in label for line breaks)")
    ap.add_argument("--ref_tag", default=None, help="tag used to pick the median-MAE subject")
    args = ap.parse_args()

    global VARIANTS, REF_TAG
    if args.variants:
        VARIANTS = [(s.split("=", 1)[0], s.split("=", 1)[1].replace("\\n", "\n")) for s in args.variants]
    if args.ref_tag:
        REF_TAG = args.ref_tag

    subjects = get_split_subjects(args.split_file, args.split_name)
    metrics = load_per_subject(args.eval_root)
    raw_dir = os.path.join(args.eval_root, "raw")
    out_dir = os.path.join(args.eval_root, "figures")

    picks = {}
    if args.subjects:
        for s in args.subjects:
            picks[get_region_key(s)] = s
    for region in REGIONS:
        subj = picks.get(region) or pick_subject(region, subjects, metrics)
        if subj:
            render(region, subj, raw_dir, metrics, args.num_slices, out_dir)


if __name__ == "__main__":
    main()
