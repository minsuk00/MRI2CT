"""Run the BabyUNet CADS 35-label segmenter (di54npq3, centerwise, epoch 419) on
both the real CT and the U-Net synthetic CT for every eval subject, and save the
argmax label maps. These feed the seg-downstream failure analysis (report 09):
babyseg(realCT) is the segmenter's ceiling, babyseg(sCT) is what we get on the
synthetic CT, both compared against the GT CADS seg.

How the values are produced (stated for the methodology section):
  - input CT clipped to [-1024, 1024] HU and linearly mapped to [0, 1]
    (verified equivalent to per-window min-max stretch: the network's instance
    norm cancels the difference, Dice within 0.001).
  - MONAI sliding_window_inference, roi 128^3, sw_batch 8, overlap 0.5, bf16
    autocast; argmax over the 35 channels. (Trainer validate() used overlap 0.7;
    0.5 is verified to give the same Dice within ~0.01 and the ceiling-correction
    cancels the residual since real CT and sCT use the identical setting.)
  - real CT comes from dataset ct.nii (raw HU); sCT from eval sample.nii.gz.
  - output label map saved as uint8 NIfTI on the input's canonical grid.
"""
import os
import sys
import glob
import argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../anatomix"))
from anatomix.segmentation.segmentation_utils import load_model_v2  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
CKPT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/runs/20260615_2205_di54npq3/seg_baby_unet_best.pth"


def load_seg_model(device):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = load_model_v2("scratch", 34, device, compile_model=False)  # builds arch; ignores weights
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def canon(path):
    im = nib.as_closest_canonical(nib.load(path))
    return np.asarray(im.dataobj, dtype=np.float32), im.affine


@torch.no_grad()
def segment(model, ct_hu, device):
    x = np.clip(ct_hu, -1024.0, 1024.0)
    x = (x + 1024.0) / 2048.0
    x = torch.from_numpy(x)[None, None].to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = sliding_window_inference(x, (128, 128, 128), 8, model, overlap=0.5)
    return torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet", help="eval model whose sCT to segment")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N subjects")
    args = ap.parse_args()

    device = "cuda"
    model = load_seg_model(device)
    print(f"[seg_infer] loaded di54npq3 (strict) on {device}", flush=True)

    subs = sorted(os.path.basename(p) for p in glob.glob(os.path.join(EVAL, "volumes", args.model, "*")))
    if args.limit:
        subs = subs[: args.limit]
    out_real = os.path.join(EVAL, "seg", "realct")
    out_sct = os.path.join(EVAL, "seg", args.model)
    os.makedirs(out_real, exist_ok=True)
    os.makedirs(out_sct, exist_ok=True)
    print(f"[seg_infer] {len(subs)} subjects, model={args.model}", flush=True)

    for i, s in enumerate(subs):
        jobs = [
            (os.path.join(out_real, s, "seg.nii.gz"), os.path.join(DATA, s, "ct.nii")),
            (os.path.join(out_sct, s, "seg.nii.gz"), os.path.join(EVAL, "volumes", args.model, s, "sample.nii.gz")),
        ]
        for dst, src in jobs:
            if os.path.exists(dst):
                continue
            if not os.path.exists(src):
                print(f"  [skip] missing {src}", flush=True)
                continue
            ct, aff = canon(src)
            seg = segment(model, ct, device)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            nib.save(nib.Nifti1Image(seg, aff), dst)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(subs)} done", flush=True)
    print("[seg_infer] complete", flush=True)


if __name__ == "__main__":
    main()
