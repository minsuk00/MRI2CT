"""Quantization example matched to full_eval report, UNet baseline: subject
1BC065, report slices linspace(0.15D,0.85D,4), brain window [-100,100] HU."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from anatomix.model.network import Unet

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference

from common.data import build_data_dicts, default_monai_cache_dir, get_cached_transforms
from common.utils import clean_state_dict, unpad

CKPT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/runs/20260507_0952_9xmodnhn/checkpoint_last.pt"
EVAL = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/full_eval_20260609/volumes/unet"
SUBJ = "1BC065"
OUT = "/home/minsukc/MRI2CT/notebooks/bf16_vs_fp32_1BC065_unet.png"
FP32_TMP = "/home/minsukc/MRI2CT/notebooks/_fp32_1BC065_unet_sample.nii.gz"
device = torch.device("cuda")


def load_vol(path):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


@torch.inference_mode()
def run_fp32():
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    model = Unet(
        dimension=3, input_nc=cfg.get("input_nc", 1), output_nc=cfg.get("output_nc", 1),
        num_downs=cfg.get("num_downs", 4), ngf=cfg.get("ngf", 16),
        norm=cfg.get("norm", "batch"), final_act="sigmoid",
    ).to(device)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=True)
    model.eval()

    val_xform = get_cached_transforms(
        patch_size=cfg.get("patch_size", 128), res_mult=cfg.get("res_mult", 32), enforce_ras=True,
        mri_norm=cfg.get("mri_norm", "minmax"), ct_range=tuple(cfg.get("ct_range", (-1024, 1024))),
        load_seg=False, load_body_mask=True,
    )
    dicts = build_data_dicts(cfg.get("root_dir"), [SUBJ], load_seg=False, load_body_mask=True)
    ds = PersistentDataset(data=dicts, transform=val_xform,
                           cache_dir=default_monai_cache_dir(), hash_transform=pickle_hashing)
    batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))

    mri = batch["mri"].to(device).float()
    orig_shape = batch["original_shape"][0].tolist()
    affine = batch["ct_affine"][0].cpu().numpy()
    pred = sliding_window_inference(  # fp32, no autocast
        inputs=mri, roi_size=(cfg.get("val_patch_size", 256),) * 3,
        sw_batch_size=cfg.get("val_sw_batch_size", 1), predictor=model,
        overlap=cfg.get("val_sw_overlap", 0.25), device=device)
    pred_hu = (unpad(pred.float(), orig_shape) * 2048.0 - 1024.0).cpu().numpy().squeeze()
    nib.save(nib.Nifti1Image(pred_hu, affine), FP32_TMP)


def main():
    if not os.path.exists(FP32_TMP):
        run_fp32()

    gt = load_vol(os.path.join(EVAL, SUBJ, "target.nii.gz"))
    bf16 = load_vol(os.path.join(EVAL, SUBJ, "sample.nii.gz"))
    fp32 = load_vol(FP32_TMP)

    D = gt.shape[-1]
    zs = np.linspace(0.15 * D, 0.85 * D, 4, dtype=int)

    def nlev(v):
        w = v[(v > -100) & (v < 100)]
        return np.unique(np.round(w, 3)).size

    cols = [("GT CT", gt, "g"), (f"UNet bf16 (current)\n{nlev(bf16)} HU levels", bf16, "g"),
            (f"UNet fp32 (fix)\n{nlev(fp32)} HU levels", fp32, "g"),
            ("bf16 - fp32 (HU)", None, "d")]
    nr, nc = len(zs), len(cols)
    fig, ax = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.4 * nr), squeeze=False)
    plt.subplots_adjust(wspace=0.03, hspace=0.05, top=0.93)
    for r, z in enumerate(zs):
        for c, (name, vol, kind) in enumerate(cols):
            a = ax[r, c]; a.axis("off")
            if kind == "d":
                a.imshow(np.rot90(bf16[:, :, z] - fp32[:, :, z]), cmap="RdBu", vmin=-8, vmax=8)
            else:
                a.imshow(np.rot90(vol[:, :, z]), cmap="gray", vmin=-100, vmax=100)
            if r == 0:
                a.set_title(name, fontsize=11)
            if c == 0:
                a.text(-0.05, 0.5, f"z={z}", transform=a.transAxes, rotation=90,
                       va="center", ha="right", fontsize=10)
    fig.suptitle(f"UNet brain {SUBJ}  |  report slices {zs.tolist()}  |  window [-100,100] HU",
                 fontsize=14, y=0.965)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"bf16 levels={nlev(bf16)}  fp32 levels={nlev(fp32)}")


if __name__ == "__main__":
    main()
