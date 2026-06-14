"""Quantization example matched to full_eval report: subject 1BC065, amix, the
report's exact slices (linspace(0.15D,0.85D,4)), brain window [-100,100] HU.

bf16 panel = the volume already saved in full_eval (the buggy one).
fp32 panel = fresh fp32 inference, saved + reloaded through the SAME path so the
orientation/slicing is byte-identical to the report's renderer.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference

from amix.validate import build_amix_models, make_combined_forward
from common.data import build_data_dicts, default_monai_cache_dir, get_cached_transforms
from common.utils import clean_state_dict, unpad

CKPT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/runs/20260509_1413_6hjye9gp/checkpoint_last.pt"
EVAL = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/full_eval_20260609/volumes/amix"
SUBJ = "1BC065"
OUT = "/home/minsukc/MRI2CT/notebooks/bf16_vs_fp32_1BC065.png"
FP32_TMP = "/home/minsukc/MRI2CT/notebooks/_fp32_1BC065_sample.nii.gz"
device = torch.device("cuda")


# report's exact loader
def load_vol(path):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


@torch.inference_mode()
def run_fp32():
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    feat, translator, res_mult = build_amix_models(cfg, device)
    translator.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=True)
    translator.eval()
    combined = make_combined_forward(feat, translator, cfg)

    val_xform = get_cached_transforms(
        patch_size=cfg.get("patch_size", 128), res_mult=res_mult, enforce_ras=True,
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
        sw_batch_size=cfg.get("val_sw_batch_size", 1), predictor=combined,
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
    zs = np.linspace(0.15 * D, 0.85 * D, 4, dtype=int)  # report's slices

    def nlev(v):
        w = v[(v > -100) & (v < 100)]
        return np.unique(np.round(w, 3)).size

    cols = [("GT CT", gt, "g"), (f"Amix bf16 (current)\n{nlev(bf16)} HU levels", bf16, "g"),
            (f"Amix fp32 (fix)\n{nlev(fp32)} HU levels", fp32, "g"),
            ("bf16 - fp32 (HU)", None, "d")]
    nr, nc = len(zs), len(cols)
    fig, ax = plt.subplots(nr, nc, figsize=(3.0 * nc, 3.4 * nr), squeeze=False)
    plt.subplots_adjust(wspace=0.03, hspace=0.05, top=0.93)
    for r, z in enumerate(zs):
        for c, (name, vol, kind) in enumerate(cols):
            a = ax[r, c]; a.axis("off")
            if kind == "d":
                im = a.imshow(np.rot90(bf16[:, :, z] - fp32[:, :, z]), cmap="RdBu", vmin=-8, vmax=8)
            else:
                a.imshow(np.rot90(vol[:, :, z]), cmap="gray", vmin=-100, vmax=100)
            if r == 0:
                a.set_title(name, fontsize=11)
            if c == 0:
                a.text(-0.05, 0.5, f"z={z}", transform=a.transAxes, rotation=90,
                       va="center", ha="right", fontsize=10)
    fig.suptitle(f"amix brain {SUBJ}  |  report slices {zs.tolist()}  |  window [-100,100] HU",
                 fontsize=14, y=0.965)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"bf16 levels={nlev(bf16)}  fp32 levels={nlev(fp32)}")


if __name__ == "__main__":
    main()
