"""Three-model MRI->CT comparison figure: UNet, Amix v1.4, KoalAI.

Usage:
    python src/evaluate/visualize_three_models.py
    python src/evaluate/visualize_three_models.py --subjects 1ABB116 1THB011 1HNC117 1PC011 1BC050
    python src/evaluate/visualize_three_models.py --subjects 1THB006 --out_dir /tmp/viz
"""

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.data import get_region_key, get_split_subjects  # noqa: E402

SYNTHRAD_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
EVAL_ROOT = "/home/minsukc/MRI2CT/evaluation_results/baselines_latest_20260529"
NNSYN_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/nnsyn_workspace/results"
SPLIT_FILE = "/home/minsukc/MRI2CT/splits/center_wise_split.txt"

MODELS = [
    ("UNet", "unet", os.path.join(EVAL_ROOT, "unet")),
    ("Amix v1.4", "amix_v1_4", os.path.join(EVAL_ROOT, "amix_v1_4")),
]

# koalAI: per-region dataset dirs
KOALAI_REGION_DS = {
    "thorax": "Dataset962_synthrad2025_task1_mri2ct_thorax",
    "abdomen": "Dataset960_synthrad2025_task1_mri2ct_abdomen",
    "head_neck": "Dataset964_synthrad2025_task1_mri2ct_head_neck",
    "brain": "Dataset966_synthrad2025_task1_mri2ct_brain",
    "pelvis": "Dataset968_synthrad2025_task1_mri2ct_pelvis",
}
KOALAI_TRAINER = "nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres"

HU_WINDOWS = {
    "brain": (-100, 100),
    "abdomen": (-1000, 1000),
    "thorax": (-1000, 1000),
    "head_neck": (-1000, 1000),
    "pelvis": (-1000, 1000),
}


def find_pred_path(model_root, subj_id):
    flat = os.path.join(model_root, subj_id, "sample.nii.gz")
    if os.path.exists(flat):
        return flat
    sharded = glob(os.path.join(model_root, "shard_*_of_*", subj_id, "sample.nii.gz"))
    if sharded:
        return sharded[0]
    raise FileNotFoundError(f"No sample.nii.gz for {subj_id} under {model_root}")


def find_target_path(model_root, subj_id):
    flat = os.path.join(model_root, subj_id, "target.nii.gz")
    if os.path.exists(flat):
        return flat
    sharded = glob(os.path.join(model_root, "shard_*_of_*", subj_id, "target.nii.gz"))
    if sharded:
        return sharded[0]
    raise FileNotFoundError(f"No target.nii.gz for {subj_id} under {model_root}")


def load_koalai_pred(subj_id):
    region = get_region_key(subj_id)
    ds = KOALAI_REGION_DS[region]
    mha_path = os.path.join(NNSYN_ROOT, ds, KOALAI_TRAINER, "fold_0", "validation_revert_norm", f"{subj_id}.mha")
    if not os.path.exists(mha_path):
        raise FileNotFoundError(f"KoalAI pred not found: {mha_path}")
    img = sitk.ReadImage(mha_path)
    # sitk: [z,y,x] in LPS → transpose to [x,y,z] in LPS, flip x+y to get RAS
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype(np.float32)
    arr = np.flip(arr, axis=(0, 1)).copy()
    return arr


def load_metrics_table(metrics_path):
    rows = {}
    header = None
    with open(metrics_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("="):
                continue
            parts = stripped.split()
            if header is None:
                if parts[0] == "subj_id":
                    header = parts
                continue
            if header is None or len(parts) < 2:
                continue
            row = {}
            for i, col in enumerate(header[1:], start=1):
                try:
                    row[col] = float(parts[i])
                except (ValueError, IndexError):
                    row[col] = None
            rows[parts[0]] = row
    return rows


KOALAI_METRICS_PATH = os.path.join(EVAL_ROOT, "koalai", "validate_metrics.txt")


def load_all_metrics():
    out = {}
    for _, key, root in MODELS:
        for fname in ("validate_metrics_combined.txt", "validate_metrics.txt"):
            path = os.path.join(root, fname)
            if os.path.exists(path):
                out[key] = load_metrics_table(path)
                break
        else:
            out[key] = {}
    if os.path.exists(KOALAI_METRICS_PATH):
        out["koalai"] = load_metrics_table(KOALAI_METRICS_PATH)
    else:
        out["koalai"] = {}
    return out


def metrics_caption(row):
    if not row:
        return "(metrics pending)"
    # prefer body metrics where available
    mae  = row.get("body_mae_hu") or row.get("mae_hu")
    psnr = row.get("body_psnr")   or row.get("psnr")
    ssim = row.get("body_ssim")   or row.get("ssim")
    parts = []
    if mae  is not None: parts.append(f"bMAE:{mae:.1f}")
    if psnr is not None: parts.append(f"bPSNR:{psnr:.2f}")
    if ssim is not None: parts.append(f"bSSIM:{ssim:.3f}")
    return "  ".join(parts) if parts else "(no metrics)"


def pick_one_per_region(split_file):
    subjects = get_split_subjects(split_file, "val")
    seen = set()
    picks = []
    for s in subjects:
        r = get_region_key(s)
        if r not in seen:
            seen.add(r)
            picks.append(s)
        if len(picks) == 5:
            break
    return picks


def render_subject(subj_id, metrics_all, out_dir, num_slices):
    region = get_region_key(subj_id)
    ct_vmin, ct_vmax = HU_WINDOWS.get(region, (-1000, 1000))

    mri_path = sorted(glob(os.path.join(SYNTHRAD_ROOT, subj_id, "moved_mr*.nii*")))
    if not mri_path:
        raise FileNotFoundError(f"No moved_mr*.nii* in {SYNTHRAD_ROOT}/{subj_id}")
    mri = nib.as_closest_canonical(nib.load(mri_path[0])).get_fdata()

    gt = nib.load(find_target_path(MODELS[0][2], subj_id)).get_fdata()

    # Columns: MRI, GT, UNet, Amix, KoalAI
    cols = [("MRI", mri, "mri", None), ("GT CT", gt, "ct", None)]

    for name, key, root in MODELS:
        try:
            arr = nib.load(find_pred_path(root, subj_id)).get_fdata()
        except FileNotFoundError:
            print(f"[viz] WARNING: {key} pred not found for {subj_id}, skipping")
            continue
        row = metrics_all.get(key, {}).get(subj_id)
        cols.append((name, arr, "ct", row))

    try:
        koalai_arr = load_koalai_pred(subj_id)
        if koalai_arr.shape != gt.shape:
            out_arr = np.full(gt.shape, -1024.0, dtype=np.float32)
            s = tuple(slice(0, min(a, b)) for a, b in zip(koalai_arr.shape, gt.shape))
            out_arr[s] = koalai_arr[s]
            koalai_arr = out_arr
        koalai_row = metrics_all.get("koalai", {}).get(subj_id)
        cols.append(("KoalAI ep999", koalai_arr, "ct", koalai_row))
    except FileNotFoundError as e:
        print(f"[viz] WARNING: {e}")

    D = gt.shape[-1]
    slice_idx = np.linspace(0.15 * D, 0.85 * D, num_slices, dtype=int)

    n_cols = len(cols)
    n_rows = num_slices
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.5 * n_rows), squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.08, top=0.90)

    mri_vmin, mri_vmax = float(np.percentile(mri, 1)), float(np.percentile(mri, 99))

    for r, z in enumerate(slice_idx):
        for c, (name, vol, kind, row) in enumerate(cols):
            ax = axes[r, c]
            if kind == "ct":
                ax.imshow(np.rot90(vol[:, :, z]), cmap="gray", vmin=ct_vmin, vmax=ct_vmax)
            else:
                ax.imshow(np.rot90(vol[:, :, z]), cmap="gray", vmin=mri_vmin, vmax=mri_vmax)
            ax.axis("off")
            if r == 0:
                title = name
                if row:
                    title += f"\n{metrics_caption(row)}"
                ax.set_title(title, fontsize=8.5)

    fig.suptitle(
        f"{subj_id}  ({region})  |  CT window [{ct_vmin}, {ct_vmax}] HU",
        fontsize=13, y=0.975,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subj_id}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")

    # Save individual per-model PNGs (middle slice only)
    mid_z = slice_idx[len(slice_idx) // 2]
    indiv_dir = os.path.join(out_dir, subj_id)
    os.makedirs(indiv_dir, exist_ok=True)
    for name, vol, kind, _ in cols:
        slug = name.lower().replace(" ", "_").replace(".", "").replace("/", "")
        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
        if kind == "ct":
            ax2.imshow(np.rot90(vol[:, :, mid_z]), cmap="gray", vmin=ct_vmin, vmax=ct_vmax)
        else:
            ax2.imshow(np.rot90(vol[:, :, mid_z]), cmap="gray", vmin=mri_vmin, vmax=mri_vmax)
        ax2.axis("off")
        ax2.set_title(f"{name}  (z={mid_z})", fontsize=9)
        p = os.path.join(indiv_dir, f"{slug}.png")
        fig2.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig2)
    print(f"[viz]   individual slices → {indiv_dir}/")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="Subject IDs. Defaults to 1 per region from val split.")
    parser.add_argument("--out_dir", default=os.path.join(EVAL_ROOT, "figures"))
    parser.add_argument("--num_slices", type=int, default=5)
    args = parser.parse_args()

    subjects = args.subjects or pick_one_per_region(SPLIT_FILE)
    print(f"[viz] rendering {len(subjects)} subjects: {subjects}")

    metrics_all = load_all_metrics()

    for subj in subjects:
        try:
            render_subject(subj, metrics_all, args.out_dir, args.num_slices)
        except Exception as e:
            print(f"[viz] FAILED {subj}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
