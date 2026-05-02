"""
OOD inference on CHAOS MR volumes using checkpoint_last.pt from each run.

Models: unet (i7beiuac), amixv1 (oe9z4ww1), amixv2 (dyxmg6n1), amixv3 (ld5qr1m2), amixv4 (u5512bav)
Volumes: T1 inphase, T1 outphase, T2 SPIR (CHAOS subject 1)

Output: evaluation_results/OOD_inference/<model_name>/
  - pred_ct_<vol_name>.nii.gz  (HU space)
  - viz_<vol_name>.png         (5-slice figure: MRI | Pred CT)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.inferers import sliding_window_inference

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from anatomix.model.network import Unet

from common.data import DataPreprocessing
from common.utils import clean_state_dict, unpad

# ─── configuration ───────────────────────────────────────────────────────────
SUBJ_LIST = ["1", "20", "34"]

# ─── paths ───────────────────────────────────────────────────────────────────
GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
RUNS = os.path.join(GPFS, "wandb_logs/runs")

MODELS = {
    "unet":   {"run": "20260426_2051_i7beiuac", "ckpt": "unet_baseline_epoch00200.pt"},
    # "amixv1": {"run": "20260426_2220_oe9z4ww1"},
    # "amixv2": {"run": "20260426_2336_dyxmg6n1"},
    # "amixv3": {"run": "20260426_2216_ld5qr1m2"},
    "amixv4": {"run": "20260426_2155_u5512bav", "ckpt": "anatomix_translator_epoch00200.pt"},
}

def get_chaos_vols(subj_id):
    return {
        "T1_inphase":  os.path.join(GPFS, f"CHAOS/nifti/Train_Sets/MR/{subj_id}/T1DUAL/inphase.nii.gz"),
        "T1_outphase": os.path.join(GPFS, f"CHAOS/nifti/Train_Sets/MR/{subj_id}/T1DUAL/outphase.nii.gz"),
        "T2_spir":     os.path.join(GPFS, f"CHAOS/nifti/Train_Sets/MR/{subj_id}/T2SPIR/t2spir.nii.gz"),
    }


AMIX_CKPTS = {
    "v1": "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth",
    "v1_2": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth",
    "v1_3": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_real_v1_3.pth",
    "v1_4": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth",
}

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "evaluation_results", "OOD_inference")


# ─── preprocessing ────────────────────────────────────────────────────────────
def preprocess_volume(vol_path, cfg):
    """Load and preprocess an MRI volume using the same DataPreprocessing as training."""
    nii = nib.load(vol_path)
    mri_data = torch.from_numpy(nii.get_fdata(dtype=np.float32)).unsqueeze(0)  # (1,X,Y,Z)

    # DataPreprocessing normalizes both mri and ct; pass a dummy ct (zeros)
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=mri_data, affine=nii.affine),
        ct=tio.ScalarImage(tensor=torch.zeros_like(mri_data), affine=nii.affine),
    )

    preprocess = DataPreprocessing(
        patch_size=cfg.get("patch_size", 128),
        res_mult=cfg.get("res_mult", 32),
        use_weighted_sampler=False,
        enforce_ras=True,
        mri_norm=cfg.get("mri_norm", "minmax"),
    )
    subject = preprocess(subject)

    mri_tensor = subject["mri"].data.float()  # (1,X',Y',Z')
    orig_shape = subject["original_shape"].tolist()
    pad_offset = int(subject["pad_offset"])
    affine = subject["mri"].affine  # RAS-reoriented affine
    return mri_tensor, orig_shape, pad_offset, affine


# ─── model loading ────────────────────────────────────────────────────────────
def build_feat_extractor(anat_weights, feat_norm, device):
    if anat_weights == "v1":
        model = Unet(3, 1, 16, 4, 16).to(device)
        ckpt_key = "v1"
    elif anat_weights in ("v2", "v1_2"):
        model = Unet(3, 1, 16, 5, 20, norm=feat_norm, interp="trilinear", pooling="Avg", use_bias=True).to(device)
        ckpt_key = "v1_2"
    elif anat_weights == "v1_3":
        model = Unet(3, 1, 16, 5, 20, norm=feat_norm, interp="trilinear", pooling="Avg", use_bias=True).to(device)
        ckpt_key = "v1_3"
    elif anat_weights == "v1_4":
        model = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(device)
        ckpt_key = "v1_4"
    else:
        raise ValueError(f"Unknown anatomix_weights: {anat_weights}")

    sd = clean_state_dict(torch.load(AMIX_CKPTS[ckpt_key], map_location=device))
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_model(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    cfg = state.get("config", {})
    epoch = state.get("epoch", "?")
    sd = clean_state_dict(state["model_state_dict"])

    if cfg.get("model_type") == "unet_baseline":
        model = Unet(
            dimension=3,
            input_nc=cfg.get("input_nc", 1),
            output_nc=cfg.get("output_nc", 1),
            num_downs=cfg.get("num_downs", 4),
            ngf=cfg.get("ngf", 16),
            norm=cfg.get("norm", "batch"),
            final_act="sigmoid",
        ).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model, None, cfg, epoch
    else:
        pass_mri = cfg.get("pass_mri_to_translator", False)
        input_nc = 17 if pass_mri else 16
        translator = Unet(
            dimension=3,
            input_nc=input_nc,
            output_nc=1,
            num_downs=4,
            ngf=16,
            final_act="sigmoid",
        ).to(device)
        translator.load_state_dict(sd, strict=True)
        translator.eval()

        feat_extractor = build_feat_extractor(cfg.get("anatomix_weights", "v1_3"), cfg.get("feat_norm", "instance"), device)
        return translator, feat_extractor, cfg, epoch


# ─── inference ────────────────────────────────────────────────────────────────
def make_amix_forward(feat_extractor, translator, cfg):
    feat_in = cfg.get("feat_instance_norm", False)
    feat_scale = cfg.get("feat_scale_down", 1)
    zero_mask = cfg.get("use_zero_mask", False)
    pass_mri = cfg.get("pass_mri_to_translator", False)

    def forward(x):
        with torch.no_grad():
            f = feat_extractor(x)
        if feat_in:
            f = torch.nn.functional.instance_norm(f)
        if feat_scale != 1:
            f = f / feat_scale
        if zero_mask:
            f = f * (x > 0.01).to(x.dtype)
        if pass_mri:
            f = torch.cat([f, x], dim=1)
        return translator(f)

    return forward


@torch.inference_mode()
def run_inference(model, feat_extractor, cfg, mri_tensor, device):
    val_ps = cfg.get("val_patch_size", cfg.get("patch_size", 128))
    sw_batch = cfg.get("val_sw_batch_size", 1)
    sw_overlap = cfg.get("val_sw_overlap", 0.25)
    predictor = model if feat_extractor is None else make_amix_forward(feat_extractor, model, cfg)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        pred = sliding_window_inference(
            inputs=mri_tensor,
            roi_size=(val_ps, val_ps, val_ps),
            sw_batch_size=sw_batch,
            predictor=predictor,
            overlap=sw_overlap,
            device=device,
        )
    return pred


# ─── visualisation ────────────────────────────────────────────────────────────
def save_viz(mri_np, pred_np, vol_name, model_name, out_dir):
    D = mri_np.shape[2]
    slices = np.linspace(int(0.1 * D), int(0.9 * D), 5, dtype=int)

    items = [
        (mri_np, "Input MRI", "gray", (0, 1)),
        (pred_np, "Pred CT", "gray", (0, 1)),
    ]

    fig, axes = plt.subplots(len(slices), len(items), figsize=(3 * len(items), 3.5 * len(slices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    if len(slices) == 1:
        axes = axes.reshape(1, -1)

    for i, z in enumerate(slices):
        for j, (vol, title, cmap, clim) in enumerate(items):
            ax = axes[i, j]
            ax.imshow(vol[:, :, z], cmap=cmap, vmin=clim[0], vmax=clim[1])
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.suptitle(f"Model: {model_name} | Volume: {vol_name}", fontsize=13, y=1.01)
    out_path = os.path.join(out_dir, f"viz_{vol_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved viz  → {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for subj_id in SUBJ_LIST:
        print(f"\n{'#' * 60}\nSubject: {subj_id}\n{'#' * 60}")
        chaos_vols = get_chaos_vols(subj_id)

        for model_name, info in MODELS.items():
            print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")

            ckpt_name = info.get("ckpt", "checkpoint_last.pt")
            ckpt_path = os.path.join(RUNS, info["run"], ckpt_name)
            model, feat_extractor, cfg, epoch = load_model(ckpt_path, device)
            print(f"  Loaded (epoch={epoch}) from {ckpt_name}")

            out_dir = os.path.join(OUTPUT_BASE, f"subject_{subj_id}", model_name)
            os.makedirs(out_dir, exist_ok=True)

            for vol_name, vol_path in chaos_vols.items():
                print(f"  [{vol_name}] preprocessing...")
                mri_tensor, orig_shape, pad_offset, affine = preprocess_volume(vol_path, cfg)
                mri_tensor = mri_tensor.unsqueeze(0).to(device)  # (1,1,X',Y',Z')

                print(f"    Shape: {list(mri_tensor.shape[2:])}, running inference...")
                pred = run_inference(model, feat_extractor, cfg, mri_tensor, device)

                pred_unpad = unpad(pred, orig_shape, pad_offset)
                mri_unpad = unpad(mri_tensor, orig_shape, pad_offset)

                pred_np = pred_unpad.float().squeeze().cpu().numpy()  # (X,Y,Z) in [0,1]
                pred_hu = pred_np * 2048.0 - 1024.0

                ct_path = os.path.join(out_dir, f"pred_ct_{vol_name}.nii.gz")
                nib.save(nib.Nifti1Image(pred_hu.astype(np.float32), affine), ct_path)
                print(f"    Saved CT   → {ct_path}")

                mri_vis = mri_unpad.float().squeeze().cpu().numpy()
                save_viz(mri_vis, pred_np, vol_name, model_name, out_dir)

            del model
            if feat_extractor is not None:
                del feat_extractor
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
