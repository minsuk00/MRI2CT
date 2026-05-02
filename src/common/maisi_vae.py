"""
Utility for loading the MAISI VAE (AutoencoderKlMaisi) from NV-Generate-CT/MR checkpoints.

Checkpoint paths (via ckpt/ symlink → GPFS):
  CT VAE:  ckpt/nv-generate-ct/models/autoencoder_v1.pt
  MR VAE:  ckpt/nv-generate-mr/models/autoencoder_v2.pt

Architecture defined in NV-Generate-CTMR/configs/config_network_rflow.json.
Both CT and MR share the same architecture; the MR VAE was additionally trained on MRI.
"""

import os

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from torch.amp import autocast

# Mirrors the autoencoder_def block in config_network_rflow.json
_VAE_KWARGS = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    latent_channels=4,
    num_channels=[64, 128, 256],
    num_res_blocks=[2, 2, 2],
    norm_num_groups=32,
    norm_eps=1e-6,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    use_checkpointing=False,
    use_convtranspose=False,
    norm_float16=True,
    num_splits=1,  # set to 1 for inference; training uses 4
    dim_split=1,
)

_CKPT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "ckpt")

CKPT_PATHS = {
    "ct": os.path.join(_CKPT_ROOT, "nv-generate-ct", "models", "autoencoder_v1.pt"),
    "mr": os.path.join(_CKPT_ROOT, "nv-generate-mr", "models", "autoencoder_v2.pt"),
}


def load_maisi_vae(modality: str = "mr", device: torch.device | None = None) -> AutoencoderKlMaisi:
    """
    Load and return the MAISI AutoencoderKlMaisi in eval mode.

    Args:
        modality: "ct" (autoencoder_v1.pt) or "mr" (autoencoder_v2.pt).
        device: target device; defaults to cuda if available, else cpu.

    Returns:
        AutoencoderKlMaisi with frozen weights in eval mode.

    Usage:
        vae = load_maisi_vae("mr")
        with torch.no_grad():
            z_mu, z_sigma = vae.encode(x)   # x: (B,1,H,W,D) float16/32
            latent = vae.sampling(z_mu, z_sigma)
            recon = vae.decode(latent)
    """
    if modality not in CKPT_PATHS:
        raise ValueError(f"modality must be 'ct' or 'mr', got {modality!r}")

    ckpt_path = os.path.abspath(CKPT_PATHS[modality])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Download with:\n"
            f"  huggingface-cli download nvidia/NV-Generate-{'MR' if modality == 'mr' else 'CT'} "
            f"models/autoencoder_v{'2' if modality == 'mr' else '1'}.pt "
            f"--local-dir ckpt/nv-generate-{modality}/"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = AutoencoderKlMaisi(**_VAE_KWARGS).to(device)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "unet_state_dict" in state_dict:
        state_dict = state_dict["unet_state_dict"]
    vae.load_state_dict(state_dict)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    return vae


def encode_patch(vae: AutoencoderKlMaisi, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
    """
    Encode a patch to latent space.

    Args:
        vae: loaded MAISI VAE (from load_maisi_vae).
        x: input tensor (B, 1, H, W, D), float16 or float32.
        sample: if True return a sampled latent; if False return z_mu (deterministic).

    Returns:
        Latent tensor (B, 4, H/4, W/4, D/4).
    """
    with torch.no_grad(), autocast("cuda"):
        z_mu, z_sigma = vae.encode(x)
    return vae.sampling(z_mu, z_sigma) if sample else z_mu


def decode_latent(vae: AutoencoderKlMaisi, z: torch.Tensor) -> torch.Tensor:
    """
    Decode a latent tensor back to image space.

    Args:
        vae: loaded MAISI VAE (from load_maisi_vae).
        z: latent tensor (B, 4, H/4, W/4, D/4).

    Returns:
        Reconstructed image (B, 1, H, W, D), same dtype as z.
    """
    with torch.no_grad(), autocast("cuda"):
        return vae.decode(z)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from monai.inferers import SlidingWindowInferer

    from common.utils import anatomix_normalize

    SUBJECT_DIR = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat/1ABA005"
    OUT_DIR = "/home/minsukc/MRI2CT/tests"
    SW_ROI = (128, 128, 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    inferer = SlidingWindowInferer(
        roi_size=SW_ROI,
        sw_batch_size=1,
        overlap=0.25,
        progress=True,
        device=torch.device("cpu"),  # accumulate on CPU to save VRAM
        sw_device=device,
    )

    def to_np(t):
        return t.squeeze().float().cpu().numpy()

    def sliding_recon(vae, vol_tensor):
        """Run sliding window VAE reconstruction over a full volume."""
        x = vol_tensor.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)
        with torch.no_grad(), autocast("cuda"):
            recon = inferer(x, lambda t: vae.reconstruct(t))
        return recon  # (1,1,H,W,D)

    def center_slices(arr):
        x, y, z = [s // 2 for s in arr.shape]
        return arr[x], arr[:, y], arr[:, :, z]

    def save_comparison(mr_arr, ct_arr, recon_arr, l1, path):
        """3 rows (axial/coronal/sagittal) × 4 cols (MR | CT | Recon | Error)."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        planes = ["Axial", "Coronal", "Sagittal"]
        mr_slices = center_slices(mr_arr)
        ct_slices = center_slices(ct_arr)
        re_slices = center_slices(recon_arr)

        for row, (plane, mr_sl, ct_sl, re_sl) in enumerate(zip(planes, mr_slices, ct_slices, re_slices)):
            err = np.abs(ct_sl - re_sl)
            axes[row, 0].imshow(mr_sl, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"{plane} — MR Input")
            axes[row, 1].imshow(ct_sl, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_title(f"{plane} — CT Input")
            axes[row, 2].imshow(re_sl, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].set_title(f"{plane} — CT Recon")
            im = axes[row, 3].imshow(err, cmap="hot", vmin=0, vmax=0.2)
            axes[row, 3].set_title(f"{plane} — |Error|")
            fig.colorbar(im, ax=axes[row, 3], fraction=0.046)

        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"CT VAE sliding window reconstruction  (L1={l1:.4f})", fontsize=14)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    # ── Load volumes ─────────────────────────────────────────────────────────
    mr_nib = nib.load(os.path.join(SUBJECT_DIR, "moved_mr.nii"))
    mr_vol = anatomix_normalize(
        torch.as_tensor(np.array(mr_nib.dataobj), dtype=torch.float32),
        percentile_range=(0, 99.5),
    )

    ct_nib = nib.load(os.path.join(SUBJECT_DIR, "ct.nii"))
    ct_vol = anatomix_normalize(
        torch.as_tensor(np.array(ct_nib.dataobj), dtype=torch.float32),
        clip_range=(-1000, 1000),
    )
    print(f"MR volume: {mr_vol.shape},  CT volume: {ct_vol.shape}")

    # ── CT VAE sliding window reconstruction ─────────────────────────────────
    print("\nLoading CT VAE (autoencoder_v1.pt)...")
    vae_ct = load_maisi_vae("ct", device=device)

    print("Running sliding window on CT...")
    recon_ct = sliding_recon(vae_ct, ct_vol)
    recon_arr = to_np(recon_ct)
    ct_arr = ct_vol.numpy()
    l1 = float(np.abs(recon_arr - ct_arr).mean())
    print(f"CT L1 recon error (full volume): {l1:.4f}")

    # Save NIfTI
    recon_nii = nib.Nifti1Image(recon_arr, affine=ct_nib.affine, header=ct_nib.header)
    nib.save(recon_nii, os.path.join(OUT_DIR, "maisi_vae_ct_recon.nii.gz"))
    print(f"Saved: {OUT_DIR}/maisi_vae_ct_recon.nii.gz")

    # Save PNG comparison
    save_comparison(
        mr_vol.numpy(),
        ct_arr,
        recon_arr,
        l1,
        os.path.join(OUT_DIR, "maisi_vae_ct_recon.png"),
    )

    print("\nDone.")
