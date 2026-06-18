"""Per-term gradient-magnitude probe for the converged UNet baseline (9xmodnhn, ep799).

For each training batch we run one forward pass, then backprop EACH raw loss term
separately (L1, SSIM, teacher-Dice, bone-Dice) and measure the L2 norm of its
gradient w.r.t. the model parameters. This isolates how much each term actually
moves the weights (gradient magnitude) vs. how big its loss value is.

Mirrors src/unet_baseline/train.py: same cached transforms, weighted random crop,
GPU augmentation, bf16 autocast forward, teacher (BabyUNet) for the Dice terms.

Run in the `mrct` env on a GPU.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from anatomix.model.network import Unet
from anatomix.segmentation.segmentation_utils import load_model_v1_2

from common.config import DEFAULT_CONFIG
from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_gpu_transforms,
    get_random_crop,
    get_split_subjects,
    gpu_augment_batch,
)
from common.loss import AnatomixPerceptualLoss, get_class_dice
from common.utils import clean_state_dict
from fused_ssim import fused_ssim3d
from monai.data import DataLoader, Dataset, PersistentDataset

# --- exact config of run 9xmodnhn (from checkpoint['config']) ---
CKPT = "/home/minsukc/MRI2CT/wandb/runs/20260507_0952_9xmodnhn/checkpoint_last.pt"
ROOT = DEFAULT_CONFIG["root_dir"]
SPLIT = "splits/center_wise_split.txt"
TEACHER = DEFAULT_CONFIG["teacher_weights_path"]
PATCH = 128
RES_MULT = 16
BATCH_SIZE = 1
N_BATCHES = 48
WEIGHTS = {"l1": 1.0, "ssim": 0.1, "perceptual": 0.5, "dice": 0.1, "dice_bone": 0.3}
BONE_IDX = 5
N_CLASSES = 12
DEVICE = "cuda"
SEED = 42


def grad_norm(model):
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().pow(2).sum().item()
    return sq ** 0.5


def main():
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("high")

    # --- model (raw, no compile) ---
    model = Unet(dimension=3, input_nc=1, output_nc=1, num_downs=4, ngf=16,
                 norm="batch", final_act="sigmoid").to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]), strict=True)
    model.train()  # batchnorm in train mode = training-time behavior
    print(f"[probe] loaded ep{ckpt.get('epoch')} ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- teacher (frozen) for Dice ---
    teacher = load_model_v1_2(pretrained_ckpt=TEACHER, n_classes=N_CLASSES - 1,
                              device=DEVICE, compile_model=False)
    teacher.to(device=DEVICE, dtype=torch.bfloat16).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # --- perceptual extractor (frozen Anatomix), as used by 06e850ny ---
    perceptual = AnatomixPerceptualLoss(layers=None, device=DEVICE)

    # --- data pipeline (matches training: cached -> weighted crop -> gpu augment) ---
    cache_dir = default_monai_cache_dir()
    cached_xform = get_cached_transforms(patch_size=PATCH, res_mult=RES_MULT, enforce_ras=True,
                                         mri_norm="minmax", load_seg=True, use_float16_storage=True)
    train_subjects = get_split_subjects(SPLIT, "train")
    train_dicts = build_data_dicts(ROOT, train_subjects, load_seg=True)
    base = PersistentDataset(data=train_dicts, transform=cached_xform, cache_dir=cache_dir)
    crop = get_random_crop(patch_size=PATCH, use_weighted_sampler=True, has_seg=True, num_samples=1)
    ds = Dataset(data=base, transform=crop)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                        persistent_workers=True, pin_memory=False)
    gpu_xform = get_gpu_transforms(augment=True, has_seg=True)

    # --- accumulate per-term grad norm + raw loss over N_BATCHES ---
    terms = ["l1", "ssim", "perceptual", "dice", "dice_bone"]
    g_acc = {t: [] for t in terms}
    v_acc = {t: [] for t in terms}

    it = iter(loader)
    for b in range(N_BATCHES):
        batch = next(it)
        batch = gpu_augment_batch(batch, gpu_xform, DEVICE)
        mri, ct, seg = batch["mri"], batch["ct"], batch["seg"]

        # entire forward+loss runs under autocast, exactly as in _train_step
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(mri)
            pred_probs = teacher(pred)
            l1 = torch.nn.functional.l1_loss(pred, ct)
            ssim = 1.0 - fused_ssim3d(pred.float(), ct.float(), train=True)
            perc = perceptual(pred, ct)
            class_dices, bone_dice = get_class_dice(pred_probs, seg, bone_idx=BONE_IDX)
            dice = 1.0 - class_dices[1:].mean()
            dice_bone = 1.0 - bone_dice
        loss_terms = {"l1": l1, "ssim": ssim, "perceptual": perc, "dice": dice, "dice_bone": dice_bone}

        for t in terms:
            model.zero_grad(set_to_none=True)
            loss_terms[t].backward(retain_graph=True)
            g_acc[t].append(grad_norm(model))
            v_acc[t].append(loss_terms[t].item())
        print(f"[probe] batch {b + 1}/{N_BATCHES} done")

    # --- report ---
    def mean(xs):
        return sum(xs) / len(xs)

    def std(xs):
        m = mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    def median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    print("\n" + "=" * 92)
    print(f"Per-term diagnostics over {N_BATCHES} training batches (bs={BATCH_SIZE}, augment=ON)")
    print("=" * 92)
    hdr = f"{'term':<11}{'weight':>8}{'med loss':>11}{'med |grad|':>13}{'w*med|grad|':>14}{'w*mean|grad|':>15}"
    print(hdr)
    print("-" * 92)
    wg_med, wg_mean = {}, {}
    for t in terms:
        wg_med[t] = WEIGHTS[t] * median(g_acc[t])
        wg_mean[t] = WEIGHTS[t] * mean(g_acc[t])
        print(f"{t:<11}{WEIGHTS[t]:>8.2f}{median(v_acc[t]):>11.4f}{median(g_acc[t]):>13.4f}"
              f"{wg_med[t]:>14.4f}{wg_mean[t]:>15.4f}")

    print("-" * 92)
    print("Normalized WEIGHTED gradient ratio (L1 = 1.0), by MEDIAN |grad|:")
    print("   " + "  ".join(f"{t}={wg_med[t] / wg_med['l1']:6.3f}" for t in terms))
    print("Normalized WEIGHTED gradient ratio (L1 = 1.0), by MEAN |grad|:")
    print("   " + "  ".join(f"{t}={wg_mean[t] / wg_mean['l1']:6.3f}" for t in terms))
    print("Normalized raw LOSS-VALUE ratio (L1 = 1.0), by MEDIAN:")
    print("   " + "  ".join(f"{t}={median(v_acc[t]) / median(v_acc['l1']):6.3f}" for t in terms))
    print("=" * 92)


if __name__ == "__main__":
    main()
