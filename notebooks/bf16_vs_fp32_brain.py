"""One-off: amix brain inference, bf16-autocast vs fp32, brain window [-100,100] HU.

Reuses src/amix/validate.py model-building so the forward pass is identical to
the real validator; the ONLY difference between the two panels is the autocast
wrapper around sliding_window_inference.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference

from amix.validate import build_amix_models, make_combined_forward
from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
)
from common.utils import clean_state_dict, unpad

CKPT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/runs/20260509_1413_6hjye9gp/checkpoint_last.pt"
SUBJ = "1BB005"
OUT = "/home/minsukc/MRI2CT/notebooks/bf16_vs_fp32_brain.png"
device = torch.device("cuda")


@torch.inference_mode()
def infer(combined, mri, val_ps, overlap, bsz, use_bf16):
    if use_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return sliding_window_inference(
                inputs=mri, roi_size=(val_ps,) * 3, sw_batch_size=bsz,
                predictor=combined, overlap=overlap, device=device).float()
    return sliding_window_inference(
        inputs=mri, roi_size=(val_ps,) * 3, sw_batch_size=bsz,
        predictor=combined, overlap=overlap, device=device).float()


def main():
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    feat, translator, res_mult = build_amix_models(cfg, device)
    translator.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=True)
    translator.eval()
    combined = make_combined_forward(feat, translator, cfg)

    val_xform = get_cached_transforms(
        patch_size=cfg.get("patch_size", 128), res_mult=res_mult, enforce_ras=True,
        mri_norm=cfg.get("mri_norm", "minmax"),
        ct_range=tuple(cfg.get("ct_range", (-1024, 1024))),
        load_seg=False, load_body_mask=True,
    )
    dicts = build_data_dicts(cfg.get("root_dir"), [SUBJ], load_seg=False, load_body_mask=True)
    ds = PersistentDataset(data=dicts, transform=val_xform,
                           cache_dir=default_monai_cache_dir(), hash_transform=pickle_hashing)
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))

    mri = batch["mri"].to(device).float()
    ct = batch["ct"].to(device).float()
    orig_shape = batch["original_shape"][0].tolist()
    val_ps = cfg.get("val_patch_size", 256)
    overlap = cfg.get("val_sw_overlap", 0.25)
    bsz = cfg.get("val_sw_batch_size", 1)

    pred_bf16 = unpad(infer(combined, mri, val_ps, overlap, bsz, True), orig_shape)
    pred_fp32 = unpad(infer(combined, mri, val_ps, overlap, bsz, False), orig_shape)
    ct_u = unpad(ct, orig_shape)

    to_hu = lambda t: (t * 2048.0 - 1024.0).cpu().numpy().squeeze()
    bf16_hu, fp32_hu, gt_hu = to_hu(pred_bf16), to_hu(pred_fp32), to_hu(ct_u)

    # count distinct HU levels inside the window for the caption
    def nlev(v):
        w = v[(v > -100) & (v < 100)]
        return np.unique(np.round(w, 3)).size

    # pick the axial slice with the most soft-tissue voxels (in-window) and crop
    # to the body bounding box on that slice so brain parenchyma banding is visible.
    inwin = ((gt_hu > -100) & (gt_hu < 100)).sum(axis=(0, 1))
    z = int(np.argmax(inwin))
    rows = np.any(gt_hu[:, :, z] > -500, axis=1)
    cols = np.any(gt_hu[:, :, z] > -500, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    sl = lambda v: np.rot90(v[r0:r1 + 1, c0:c1 + 1, z])

    fig, ax = plt.subplots(1, 4, figsize=(22, 7))
    for a in ax:
        a.axis("off")
    ax[0].imshow(sl(gt_hu), cmap="gray", vmin=-100, vmax=100, interpolation="nearest")
    ax[0].set_title("GT CT", fontsize=14)
    ax[1].imshow(sl(bf16_hu), cmap="gray", vmin=-100, vmax=100, interpolation="nearest")
    ax[1].set_title(f"pred bf16 autocast (current)\n{nlev(bf16_hu)} HU levels in window", fontsize=14)
    ax[2].imshow(sl(fp32_hu), cmap="gray", vmin=-100, vmax=100, interpolation="nearest")
    ax[2].set_title(f"pred fp32 (fix)\n{nlev(fp32_hu)} HU levels in window", fontsize=14)
    d = ax[3].imshow(sl(bf16_hu) - sl(fp32_hu), cmap="RdBu", vmin=-8, vmax=8, interpolation="nearest")
    ax[3].set_title("bf16 - fp32 (HU)", fontsize=14)
    fig.colorbar(d, ax=ax[3], fraction=0.046)
    fig.suptitle(f"amix brain {SUBJ}  axial z={z}  |  window [-100,100] HU", y=1.0, fontsize=15)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"bf16 window levels={nlev(bf16_hu)}  fp32 window levels={nlev(fp32_hu)}")
    print(f"whole-vol uniq: bf16={np.unique(bf16_hu).size}  fp32={np.unique(fp32_hu).size}")


if __name__ == "__main__":
    main()
