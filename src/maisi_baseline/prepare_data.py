import json
import os
from glob import glob

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.bundle import ConfigParser
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
)
from tqdm import tqdm

from src.maisi_baseline.config import AUTOENCODER_PATH, DATA_ROOT, MAISI_DATA_ROOT, NETWORK_CONFIG_PATH


def round_number(number: int, base_number: int = 32) -> int:
    """Matches original MAISI rounding (base 32 for full U-Net and VAE parity)."""
    return int(max(round(float(number) / float(base_number)), 1.0) * float(base_number))


def get_autoencoder(device):
    with open(NETWORK_CONFIG_PATH, "r") as f:
        model_def = json.load(f)

    parser = ConfigParser()
    parser.update(model_def)
    autoencoder = parser.get_parsed_content("autoencoder_def", instantiate=True).to(device)

    checkpoint = torch.load(AUTOENCODER_PATH, map_location=device, weights_only=False)
    if "unet_state_dict" in checkpoint:
        checkpoint = checkpoint["unet_state_dict"]
    autoencoder.load_state_dict(checkpoint)
    autoencoder.eval()
    return autoencoder


@torch.no_grad()
def process_pair(mr_path, ct_path, autoencoder, device, target_dim=None):
    # 1. Load MRI (Strict Parity: RAS, clip=True)
    loader = Compose(
        [
            LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
            Orientationd(keys=["image"], axcodes="RAS"),
            ScaleIntensityRangePercentilesd(keys=["image"], lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )
    mr_data = loader({"image": mr_path})
    mr_raw = mr_data["image"].unsqueeze(0).to(device)

    # 2. Load CT (Strict Parity: RAS)
    ct_loader = Compose(
        [
            LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
            Orientationd(keys=["image"], axcodes="RAS"),
            EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )
    ct_data = ct_loader({"image": ct_path})
    ct_raw = ct_data["image"].unsqueeze(0).to(device)

    # 3. Determine Target Shape (Base 4 Parity)
    if target_dim:
        target_shape = target_dim
    else:
        orig_shape = mr_raw.shape[2:]
        target_shape = [round_number(d, 32) for d in orig_shape]

    # Interpolate to target shape if necessary
    if list(mr_raw.shape[2:]) != list(target_shape):
        mr_tensor = F.interpolate(mr_raw, size=target_shape, mode="trilinear", align_corners=False)
        ct_tensor = F.interpolate(ct_raw, size=target_shape, mode="trilinear", align_corners=False)
    else:
        mr_tensor, ct_tensor = mr_raw, ct_raw

    # 4. Encode CT (Needs [0, 1] normalization for VAE input)
    # Strictly [-1000, 1000] range for ENCODER input only
    ct_norm = torch.clamp((ct_tensor + 1000) / 2000, 0, 1)

    inferer = SlidingWindowInferer(
        roi_size=[64, 64, 64],
        sw_batch_size=1,
        overlap=0.4,
        mode="gaussian",  # GAUSSIAN PARITY
        device=device,
    )

    with torch.amp.autocast("cuda"):
        ct_latent = inferer(ct_norm, autoencoder.encode_stage_2_inputs)

    affine = mr_data["image"].meta["affine"]
    if hasattr(affine, "cpu"):
        affine = affine.cpu().numpy()
    else:
        affine = np.array(affine)

    # Return raw ct_tensor (unclipped) for saving
    return mr_tensor.squeeze(0), ct_tensor.squeeze(0), ct_latent.squeeze(0), affine


def process_all(subjects=None, force=False, input_base=None, target_dim=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    autoencoder = get_autoencoder(device)

    datalist_path = os.path.join(MAISI_DATA_ROOT, "datalist.json")
    if os.path.exists(datalist_path):
        with open(datalist_path, "r") as f:
            datalist = json.load(f)
    else:
        datalist = {"training": []}

    source_root = input_base if input_base else DATA_ROOT

    if input_base:
        folders = sorted([d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))])
        splits = [("train", folders)]
    else:
        splits = [("train", sorted(os.listdir(os.path.join(source_root, "train")))), ("val", sorted(os.listdir(os.path.join(source_root, "val"))))]

    for split_name, folders in splits:
        output_split_dir = os.path.join(MAISI_DATA_ROOT, split_name)
        os.makedirs(output_split_dir, exist_ok=True)

        for folder in tqdm(folders, desc=f"Encoding {split_name}"):
            if subjects and folder not in subjects:
                continue

            if input_base:
                subj_dir = os.path.join(source_root, folder)
            else:
                subj_dir = os.path.join(source_root, split_name, folder)

            mr_candidates = glob(os.path.join(subj_dir, "*mr*T1*")) + glob(os.path.join(subj_dir, "mr.nii.gz"))
            ct_candidates = glob(os.path.join(subj_dir, "ct.nii.gz"))

            if not mr_candidates or not ct_candidates:
                continue
            mr_path, ct_path = mr_candidates[0], ct_candidates[0]

            output_subj_dir = os.path.join(output_split_dir, folder)
            os.makedirs(output_subj_dir, exist_ok=True)

            mr_out = os.path.join(output_subj_dir, "mr.nii.gz")
            ct_out = os.path.join(output_subj_dir, "ct.nii.gz")
            emb_out = os.path.join(output_subj_dir, "ct_emb.nii.gz")

            if force or not os.path.exists(emb_out) or not os.path.exists(mr_out):
                print(f"  Processing {folder}...")
                mr_t, ct_t, emb_t, affine = process_pair(mr_path, ct_path, autoencoder, device, target_dim=target_dim)

                # Safety: break symlinks if they exist to avoid overwriting original data
                for out_path, orig_path in [(mr_out, mr_path), (ct_out, ct_path), (emb_out, ct_path)]:
                    if os.path.exists(out_path):
                        if os.path.islink(out_path):
                            os.remove(out_path)
                        elif os.path.samefile(out_path, orig_path):
                            print(f"⚠️ DANGER: {out_path} is SAME FILE as {orig_path}. Skipping to avoid alteration.")
                            continue
                        elif force:
                            os.remove(out_path)

                if os.path.exists(mr_out) and not force:
                    pass  # Skip if exists
                else:
                    nib.save(nib.Nifti1Image(mr_t.cpu().numpy().transpose(1, 2, 3, 0), affine), mr_out)
                    nib.save(nib.Nifti1Image(ct_t.cpu().numpy().transpose(1, 2, 3, 0), affine), ct_out)

                    latent_affine = affine.copy()
                    latent_affine[:3, :3] *= 4
                    nib.save(nib.Nifti1Image(emb_t.cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32), latent_affine), emb_out)

            subj_mr_key = os.path.join(split_name, folder, "mr.nii.gz")
            entry = {
                "mr_image": subj_mr_key,
                "ct_image": os.path.join(split_name, folder, "ct.nii.gz"),
                "ct_emb": os.path.join(split_name, folder, "ct_emb.nii.gz"),
                "fold": 0 if split_name == "train" else 1,
                "spacing": [float(s) for s in nib.load(mr_out).header.get_zooms()[:3]],
            }

            found = False
            for idx, d in enumerate(datalist["training"]):
                if d["mr_image"] == subj_mr_key:
                    datalist["training"][idx] = entry
                    found = True
                    break
            if not found:
                datalist["training"].append(entry)

    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", help="Specific subjects to encode")
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    parser.add_argument("--input_base", type=str, help="Base directory for search")
    parser.add_argument("--target_dim", type=int, nargs=3, help="Forced target dimensions")
    args = parser.parse_args()
    process_all(subjects=args.subjects, force=args.force, input_base=args.input_base, target_dim=args.target_dim)
