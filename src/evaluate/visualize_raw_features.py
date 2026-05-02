import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from matplotlib.colors import Normalize
from monai.inferers import sliding_window_inference

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anatomix.model.network import Unet

from common.utils import anatomix_normalize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
OUTPUT_DIR = "evaluation_results/raw_amix_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBJECTS = {
    "Abdomen": "1ABA005",
    "Pelvis": "1PA001",
    "Thorax": "1THB011",
}

N_FEAT = 16  # all models output 16 feature channels
SHOW_CHANNELS = list(range(0, 16, 2))  # every other channel: 0,2,4,6,8,10,12,14


def clean_state_dict(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_model(version, device):
    print(f"Loading Anatomix {version}...")
    if version == "v1.0":
        model = Unet(3, 1, 16, 4, 16).to(device)
        ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
    elif version == "v1.2":
        model = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
        ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth"
    elif version == "v1.3":
        model = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
        ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_real_v1_3.pth"
    elif version == "v1.4":
        model = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(device)
        ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
    else:
        raise ValueError(f"Unknown version: {version}")

    if os.path.exists(ckpt):
        state_dict = torch.load(ckpt, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(clean_state_dict(state_dict), strict=True)
        print(f"  Loaded {ckpt}")
    else:
        print(f"  WARNING: weights not found at {ckpt}")

    model.eval()
    return model


def normalize_input(volume_np, is_ct):
    t = torch.from_numpy(volume_np).float()
    if is_ct:
        return anatomix_normalize(t, clip_range=(-1024, 1024)).numpy()
    else:
        return anatomix_normalize(t).numpy()


def extract_features(model, volume_np, device, patch_size=256):
    with torch.no_grad():
        inp = torch.from_numpy(volume_np[None, None]).float().to(device)
        feats = sliding_window_inference(
            inputs=inp,
            roi_size=(patch_size, patch_size, patch_size),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
            device=device,
        )
        return feats.squeeze(0).cpu().numpy()  # (16, D, H, W)


def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def visualize_subject(region, subj_id, models, versions):
    subj_path = os.path.join(DATA_ROOT, subj_id)
    mr_path = os.path.join(subj_path, "moved_mr.nii")
    ct_path = os.path.join(subj_path, "ct.nii")

    if not (os.path.exists(mr_path) and os.path.exists(ct_path)):
        print(f"  Skipping {subj_id}: data not found at {subj_path}")
        return

    mr_vol = normalize_input(nib.load(mr_path).get_fdata().transpose(2, 1, 0), is_ct=False)
    ct_vol = normalize_input(nib.load(ct_path).get_fdata().transpose(2, 1, 0), is_ct=True)

    mid_z = ct_vol.shape[0] // 2

    n_versions = len(versions)
    n_rows = n_versions * 2  # MR + CT per version
    n_cols = 1 + len(SHOW_CHANNELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0), constrained_layout=True)

    all_ch_norms = {}
    for v_idx, ver in enumerate(versions):
        print(f"  Extracting {ver} features for {subj_id}...")
        model = models[ver]
        mr_feats = extract_features(model, mr_vol, DEVICE)  # (16, D, H, W)
        ct_feats = extract_features(model, ct_vol, DEVICE)

        # Per-channel vmin/vmax shared between MR and CT — each channel uses its own scale
        ch_norms = {
            ch: Normalize(vmin=min(mr_feats[ch].min(), ct_feats[ch].min()),
                          vmax=max(mr_feats[ch].max(), ct_feats[ch].max()))
            for ch in SHOW_CHANNELS
        }
        all_ch_norms[ver] = ch_norms

        for mod_idx, (label, input_vol, feats) in enumerate([("MR", mr_vol, mr_feats), ("CT", ct_vol, ct_feats)]):
            row = v_idx * 2 + mod_idx
            slice_data = input_vol[mid_z]

            # Column 0: input image (already in [0,1]) with row label overlay
            ax = axes[row, 0]
            ax.imshow(np.rot90(slice_data), cmap="gray", vmin=0, vmax=1)
            ax.text(
                0.02, 0.97, f"{ver} — {label}", transform=ax.transAxes, fontsize=7, fontweight="bold", va="top", ha="left", color="white", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6)
            )
            ax.axis("off")

            # Feature channel columns — per-channel scale, shared across MR and CT
            for col_idx, ch in enumerate(SHOW_CHANNELS):
                axes[row, col_idx + 1].imshow(np.rot90(feats[ch, mid_z]), cmap="inferno", norm=ch_norms[ch])
                axes[row, col_idx + 1].axis("off")
                if row == 0:
                    axes[row, col_idx + 1].set_title(f"ch{ch}", fontsize=7)

        del mr_feats, ct_feats
        torch.cuda.empty_cache()

    # Add one horizontal colorbar per (version, channel) below each axes pair
    from matplotlib.cm import ScalarMappable
    for v_idx, ver in enumerate(versions):
        for col_idx, ch in enumerate(SHOW_CHANNELS):
            sm = ScalarMappable(cmap="inferno", norm=all_ch_norms[ver][ch])
            sm.set_array([])
            cb = fig.colorbar(sm, ax=[axes[v_idx * 2, col_idx + 1], axes[v_idx * 2 + 1, col_idx + 1]],
                              location="bottom", shrink=0.9, pad=0.02, aspect=15)
            cb.ax.tick_params(labelsize=5)

    axes[0, 0].set_title("Input", fontsize=6)
    plt.suptitle(f"Raw Feature Maps — {region} ({subj_id})  |  middle slice Z={mid_z}", fontsize=11)
    out_path = os.path.join(OUTPUT_DIR, f"raw_features_{region.lower()}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    versions = ["v1.0", "v1.2", "v1.3", "v1.4"]
    models = {ver: load_model(ver, DEVICE) for ver in versions}

    for region, subj_id in SUBJECTS.items():
        print(f"\nProcessing {region} ({subj_id})...")
        visualize_subject(region, subj_id, models, versions)


if __name__ == "__main__":
    main()
