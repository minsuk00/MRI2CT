import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anatomix.model.network import Unet

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
OUTPUT_DIR = "viz_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Subjects for each region
REGIONS = {
    # "Head": "1HNA001",
    "Thorax": "1THB011",
    "Abdomen": "1ABA005",
    "Pelvis": "1PA001",
}


def clean_state_dict(state_dict):
    """Strip _orig_mod prefix from torch.compile"""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    return new_state_dict


def load_anatomix_model(version, device):
    print(f"Loading Anatomix {version}...")
    if version == "v1.0":
        model = Unet(3, 1, 16, 4, 16).to(device)
        ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
    elif version in ["v1.2", "v1.3"]:
        model = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
        if version == "v1.2":
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth"
        else:  # v1.3
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
        print(f"Loaded weights from {ckpt}")
    else:
        print(f"WARNING: Weights NOT FOUND at {ckpt}")

    model.eval()
    return model


def extract_features(model, volume_np, device):
    with torch.no_grad():
        inp = torch.from_numpy(volume_np[None, None]).float().to(device)
        inp = (inp - inp.min()) / (inp.max() - inp.min() + 1e-8)

        _, _, D, H, W = inp.shape
        pad_d = (32 - (D % 32)) % 32
        pad_h = (32 - (H % 32)) % 32
        pad_w = (32 - (W % 32)) % 32

        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        inp_padded = F.pad(inp, padding, mode="reflect")
        feats_padded = model(inp_padded)
        feats = feats_padded[:, :, :D, :H, :W]

        return feats.squeeze(0).cpu().numpy()


def project_pca_raw(feats, pca_obj):
    C, D, H, W = feats.shape
    X = feats.reshape(C, -1).T
    Y = pca_obj.transform(X)
    return Y.reshape(D, H, W, 3)


def normalize_pca(Y_raw, vmin=None, vmax=None):
    if vmin is None:
        vmin = Y_raw.min(axis=(0, 1, 2))
    if vmax is None:
        vmax = Y_raw.max(axis=(0, 1, 2))
    Y_norm = (Y_raw - vmin) / (vmax - vmin + 1e-8)
    return np.clip(Y_norm, 0, 1)


def get_bone_similarity(feats, seed_coord):
    C, D, H, W = feats.shape
    z, y, x = seed_coord
    z, y, x = min(int(z), D - 1), min(int(y), H - 1), min(int(x), W - 1)

    seed_vec = feats[:, z, y, x]
    X = feats.reshape(C, -1)

    seed_vec_norm = seed_vec / (np.linalg.norm(seed_vec) + 1e-8)
    X_norm = X / (np.linalg.norm(X, axis=0, keepdims=True) + 1e-8)

    sim = np.dot(seed_vec_norm, X_norm)
    return sim.reshape(D, H, W)


def run_comparison():
    models = {ver: load_anatomix_model(ver, DEVICE) for ver in ["v1.0", "v1.2", "v1.3", "v1.4"]}

    for region_name, subj_id in REGIONS.items():
        print(f"\nProcessing {region_name} ({subj_id})...")
        subj_path = os.path.join(DATA_ROOT, subj_id)
        mr_path = os.path.join(subj_path, "moved_mr.nii")
        ct_path = os.path.join(subj_path, "ct.nii")

        if not (os.path.exists(mr_path) and os.path.exists(ct_path)):
            continue

        mr_nii, ct_nii = nib.load(mr_path), nib.load(ct_path)
        mr_input = mr_nii.get_fdata().transpose(2, 1, 0)
        ct_input = ct_nii.get_fdata().transpose(2, 1, 0)

        # 1. Paired Feature Extraction
        mr_feats, ct_feats = {}, {}
        for ver, model in models.items():
            print(f"Extracting {ver} (MR & CT)...")
            mr_feats[ver] = extract_features(model, mr_input, DEVICE)
            ct_feats[ver] = extract_features(model, ct_input, DEVICE)

        # 2. Shared PCA fitting
        print("Fitting Shared PCA...")
        samples = []
        for ver in models:
            for f_dict in [mr_feats, ct_feats]:
                C = f_dict[ver].shape[0]
                X = f_dict[ver].reshape(C, -1).T
                idx = np.random.choice(X.shape[0], 25000, replace=False)
                samples.append(X[idx])
        shared_pca = PCA(n_components=3).fit(np.concatenate(samples, axis=0))

        s_min, s_max = None, None
        raw_shared_mr, raw_shared_ct = {}, {}
        for ver in models:
            for f_dict, raw_dict in [(mr_feats, raw_shared_mr), (ct_feats, raw_shared_ct)]:
                y_raw = project_pca_raw(f_dict[ver], shared_pca)
                raw_dict[ver] = y_raw
                cur_min, cur_max = y_raw.min(axis=(0, 1, 2)), y_raw.max(axis=(0, 1, 2))
                s_min = cur_min if s_min is None else np.minimum(s_min, cur_min)
                s_max = cur_max if s_max is None else np.maximum(s_max, cur_max)

        # 3. Individual PCAs fitting
        indiv_pca_mr, indiv_pca_ct = {}, {}
        for ver in models:
            samples_v = []
            for f_dict in [mr_feats, ct_feats]:
                C = f_dict[ver].shape[0]
                X = f_dict[ver].reshape(C, -1).T
                idx = np.random.choice(X.shape[0], 25000, replace=False)
                samples_v.append(X[idx])
            pca_v = PCA(n_components=3).fit(np.concatenate(samples_v, axis=0))
            indiv_pca_mr[ver] = project_pca_raw(mr_feats[ver], pca_v)
            indiv_pca_ct[ver] = project_pca_raw(ct_feats[ver], pca_v)
            v_min = np.minimum(indiv_pca_mr[ver].min(axis=(0, 1, 2)), indiv_pca_ct[ver].min(axis=(0, 1, 2)))
            v_max = np.maximum(indiv_pca_mr[ver].max(axis=(0, 1, 2)), indiv_pca_ct[ver].max(axis=(0, 1, 2)))
            indiv_pca_mr[ver] = normalize_pca(indiv_pca_mr[ver], v_min, v_max)
            indiv_pca_ct[ver] = normalize_pca(indiv_pca_ct[ver], v_min, v_max)

        # 4. Visualization Selection
        bone_mask_global = ct_input > (ct_input.max() * 0.7)
        indices_global = np.argwhere(bone_mask_global)
        mid_z = indices_global[len(indices_global) // 2][0] if len(indices_global) > 0 else ct_input.shape[0] // 2
        slices_to_show = [mid_z, min(mid_z + 25, ct_input.shape[0] - 1), max(mid_z - 25, 0), max(mid_z - 50, 0)]

        fig, axes = plt.subplots(4, 17, figsize=(51, 12), constrained_layout=True)
        display_cols = [
            "MRI",
            "CT",
            "v1.0 MR-S",
            "v1.0 CT-S",
            "v1.2 MR-S",
            "v1.2 CT-S",
            "v1.3 MR-S",
            "v1.3 CT-S",
            "v1.0 MR-I",
            "v1.0 CT-I",
            "v1.2 MR-I",
            "v1.2 CT-I",
            "v1.3 MR-I",
            "v1.3 CT-I",
            "v1.0 Sim",
            "v1.2 Sim",
            "v1.3 Sim",
        ]
        for ax, col in zip(axes[0], display_cols):
            ax.set_title(col, fontsize=9)

        for r_idx, s_idx in enumerate(slices_to_show):
            ct_slice = ct_input[s_idx]
            y_max, x_max = np.unravel_index(np.argmax(ct_slice), ct_slice.shape)
            seed_row = (s_idx, y_max, x_max)
            axes[r_idx, 0].text(-0.2, 0.5, f"Z={s_idx}", transform=axes[r_idx, 0].transAxes, rotation=90, verticalalignment="center", fontweight="bold", fontsize=8)

            axes[r_idx, 0].imshow(np.rot90(mr_input[s_idx]), cmap="gray")
            axes[r_idx, 1].imshow(np.rot90(ct_input[s_idx]), cmap="gray")

            H_slice, W_slice = mr_input[s_idx].shape
            x_rot, y_rot = seed_row[1], W_slice - 1 - seed_row[2]

            for c, ver in enumerate(["v1.0", "v1.2", "v1.3"]):
                axes[r_idx, 2 + 2 * c].imshow(np.rot90(normalize_pca(raw_shared_mr[ver], s_min, s_max)[s_idx]))
                axes[r_idx, 3 + 2 * c].imshow(np.rot90(normalize_pca(raw_shared_ct[ver], s_min, s_max)[s_idx]))
                axes[r_idx, 8 + 2 * c].imshow(np.rot90(indiv_pca_mr[ver][s_idx]))
                axes[r_idx, 9 + 2 * c].imshow(np.rot90(indiv_pca_ct[ver][s_idx]))
                sim_slice = get_bone_similarity(mr_feats[ver], seed_row)[s_idx]
                im_sim = axes[r_idx, 14 + c].imshow(np.rot90(sim_slice), cmap="jet", vmin=0, vmax=1)
                axes[r_idx, 14 + c].scatter(x_rot, y_rot, color="white", s=40, marker="x", linewidth=1.5)

            for c in range(17):
                axes[r_idx, c].axis("off")

        fig.colorbar(im_sim, ax=axes[:, 16], location="right", fraction=0.046, pad=0.04).set_label("MRI-Bone Cosine Similarity", fontsize=10)
        plt.suptitle(f"Modality-Paired Axial Comparison: {region_name}", fontsize=16)
        plt.savefig(os.path.join(OUTPUT_DIR, f"paired_comparison_axial_{region_name.lower()}.png"), bbox_inches="tight", dpi=120)
        plt.close()
        del mr_feats, ct_feats, mr_input, ct_input
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_comparison()
