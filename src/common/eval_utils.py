"""Helpers for standalone validate scripts (cWDM / MC-DDPM / MAISI).

Module-level versions of the teacher-Dice plumbing that lives on `BaseTrainer`,
plus a plain-text report writer. Kept separate from `trainer_base.py` so the
validate scripts don't have to instantiate a Trainer (which would also bring up
WandB, optimizers, dataloaders, etc.) just to score a checkpoint.
"""
import glob
import os
import re
import time
from datetime import datetime

import numpy as np
import torch

from common.loss import get_class_dice


# ---------------------------------------------------------------------------
# Teacher (Baby U-Net) loading + sliding-window inference
# ---------------------------------------------------------------------------
def load_teacher_model(
    weights_path,
    *,
    device,
    n_classes_minus_bg=11,  # teacher trained on 11 fg classes (n_classes=12 incl. bg)
    arch="v1_2",            # "v1_2" (legacy 12-class) or "v2" (CADS 35-class)
    dtype=torch.bfloat16,
):
    """Load + freeze the Baby U-Net teacher used for Dice eval.

    Mirrors BaseTrainer._setup_teacher_model but takes explicit args so callers
    don't need a Trainer/Config in scope. `n_classes_minus_bg` matches the
    `cfg.n_classes - 1` convention in the trainer. Set arch="v2" with
    n_classes_minus_bg=34 to load the CADS 35-class teacher.
    """
    from anatomix.segmentation.segmentation_utils import load_model_v1_2, load_model_v2

    loader = load_model_v2 if arch == "v2" else load_model_v1_2
    print(f"[eval] 👨‍🏫 Loading teacher ({arch}) from {weights_path}")
    teacher = loader(
        pretrained_ckpt=weights_path,
        n_classes=n_classes_minus_bg,
        device=device,
        compile_model=False,
    )
    teacher.to(device=device, dtype=dtype)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@torch.inference_mode()
def run_teacher_sw(
    teacher,
    inputs,
    *,
    device,
    val_patch_size=128,
    sw_batch_size=2,
    overlap=0.25,
    autocast_dtype=torch.bfloat16,
):
    """Sliding-window teacher inference under bf16 autocast.

    `inputs` is the (B=1, 1, D, H, W) prediction in [0, 1]. Returns logits with
    the same spatial shape and `n_classes` channels.
    """
    from monai.inferers import sliding_window_inference

    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                        dtype=autocast_dtype):
        return sliding_window_inference(
            inputs=inputs,
            roi_size=(val_patch_size,) * 3,
            sw_batch_size=sw_batch_size,
            predictor=teacher,
            overlap=overlap,
            device=device,
        )


# ---------------------------------------------------------------------------
# Dual-teacher Dice: legacy 12-class (ct_seg.nii) + CADS 35-class
# ---------------------------------------------------------------------------
LEGACY_TEACHER_PATH = "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth"
CADS35_TEACHER_PATH = "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_cads_35_center_wise_di54npq3_epoch_1000.pth"


def default_teacher_specs():
    """Canonical teachers for evaluation Dice. The 12-class teacher keeps the
    UNSUFFIXED metric keys (dice_score_all/dice_score_bone) so historical reports
    keep meaning the same; the 35-class teacher adds `_cads35`-suffixed keys.
    Each spec names its matching GT seg file so the label spaces line up.
    """
    from common.labels import BONE_CLASS_INDICES

    return [
        {"tag": "seg12", "suffix": "", "arch": "v1_2", "weights": LEGACY_TEACHER_PATH,
         "n_classes": 12, "seg_filename": "ct_seg.nii", "bone_idx": 5},
        {"tag": "cads35", "suffix": "_cads35", "arch": "v2", "weights": CADS35_TEACHER_PATH,
         "n_classes": 35, "seg_filename": "cads_grouped_35_labels_seg.nii.gz",
         "bone_idx": BONE_CLASS_INDICES},
    ]


def load_teachers(specs, device, dtype=torch.bfloat16):
    """Load every teacher in `specs` (skipping any whose weights are missing).
    Returns the spec dicts augmented with a loaded `model`.
    """
    loaded = []
    for s in specs:
        if not os.path.exists(s["weights"]):
            print(f"[eval] ⚠️ teacher weights missing, skipping {s['tag']}: {s['weights']}")
            continue
        model = load_teacher_model(
            s["weights"], device=device, n_classes_minus_bg=s["n_classes"] - 1,
            arch=s["arch"], dtype=dtype,
        )
        loaded.append({**s, "model": model})
    return loaded


def load_seg_oriented(root_dir, subj_id, seg_filename, device):
    """Load one subject's GT seg at its RAS pre-pad (original) shape, aligned with
    the unpadded prediction. Returns (1, 1, X, Y, Z) long tensor, or None if absent.

    A plain RAS load matches the cached pipeline's unpadded seg exactly (verified):
    the pipeline records original_shape after RAS, pads at the end, then unpads back.
    """
    from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd

    base = seg_filename[:-7] if seg_filename.endswith(".nii.gz") else os.path.splitext(seg_filename)[0]
    path = None
    for ext in (".nii", ".nii.gz"):
        cand = os.path.join(root_dir, subj_id, base + ext)
        if os.path.exists(cand):
            path = cand
            break
    if path is None:
        return None
    xform = Compose([
        LoadImaged(keys=["seg"], image_only=True),
        EnsureChannelFirstd(keys=["seg"]),
        Orientationd(keys=["seg"], axcodes="RAS"),
        EnsureTyped(keys=["seg"]),
    ])
    seg = xform({"seg": path})["seg"]  # (1, X, Y, Z)
    return seg[None].to(device).long()  # (1, 1, X, Y, Z)


def load_subject_segs(root_dir, subj_id, teachers, device):
    """Preload each distinct seg file referenced by `teachers` for one subject."""
    segs = {}
    for t in teachers:
        fn = t["seg_filename"]
        if fn not in segs:
            segs[fn] = load_seg_oriented(root_dir, subj_id, fn, device)
    return segs


def rescale_pred_to_teacher(pred, ct_range):
    """Map a prediction from ct_range [0,1] to the teachers' native (-1024,1024)
    [0,1] convention (clipping), so teachers trained on that range segment it
    correctly. No-op when ct_range == (-1024,1024). Mirrors BaseTrainer._pred_for_teacher.
    """
    lo, hi = ct_range
    if (lo, hi) == (-1024, 1024):
        return pred
    pred_hu = pred * (hi - lo) + lo
    return ((pred_hu + 1024.0) / 2048.0).clamp(0.0, 1.0)


@torch.inference_mode()
def dual_teacher_dice(teachers, pred_unpad, seg_by_file, device, *, body_mask=None, sw_kwargs=None):
    """Run every teacher on `pred_unpad` once and score Dice against its matching
    seg. Returns (full_metrics, body_metrics): full keys like dice_score_all{suffix},
    body keys prefixed `body_`. A teacher whose seg is missing is skipped.
    """
    sw_kwargs = sw_kwargs or {}
    full, body = {}, {}
    for t in teachers:
        seg = seg_by_file.get(t["seg_filename"])
        if seg is None:
            continue
        logits = run_teacher_sw(t["model"], pred_unpad, device=device, **sw_kwargs)
        for k, v in compute_dice(logits, seg, bone_idx=t["bone_idx"]).items():
            full[k + t["suffix"]] = v
        if body_mask is not None:
            for k, v in compute_dice(logits, seg, mask=body_mask, bone_idx=t["bone_idx"]).items():
                body["body_" + k + t["suffix"]] = v
        del logits
    return full, body


def compute_dice(pred_logits, seg, *, mask=None, bone_idx=5, exclude_background=True):
    """Compute {dice_score_all, dice_score_bone} from teacher logits + GT seg.

    `pred_logits`: (B, C, D, H, W) raw teacher output on the prediction volume.
    `seg`:        (B, 1, D, H, W) integer GT label map at the same spatial shape.
    `mask`:       optional (B, 1, D, H, W) body mask — restricts the Dice to body
                  voxels (matches `compute_metrics_body`).
    """
    class_dices = get_class_dice(pred_logits, seg, mask=mask)
    out = {}
    out["dice_score_all"] = (
        class_dices[1:].mean() if exclude_background else class_dices.mean()
    ).item()
    bone_idxs = [bone_idx] if isinstance(bone_idx, int) else list(bone_idx)
    in_range = [b for b in bone_idxs if b < class_dices.shape[0]]
    if in_range:
        out["dice_score_bone"] = class_dices[in_range].mean().item()
    return out


@torch.inference_mode()
def compute_dice_hard(pred_logits, seg, *, bone_idx=5):
    """Hard (argmax-label) per-class Dice, following the dice-score-3d convention
    (ancestor-mithril/dice-score-3d). This is the standard *evaluation* Dice, as
    opposed to the soft probability Dice in `get_class_dice` (a training loss).

    Per foreground class c in 1..C-1, with A = {pred argmax == c}, B = {GT == c}:
        |A| == |B| == 0  -> 1.0   (true-negative agreement; class absent from both)
        |A| == 0 xor |B| == 0 -> 0.0
        else             -> 2|A∩B| / (|A| + |B|)
    `dice_score_all` = unweighted mean over the C-1 foreground classes;
    `dice_score_bone` = the class-`bone_idx` score. Whole-volume (no body mask),
    matching dice-score-3d.
    """
    pred_lab = pred_logits.argmax(dim=1)        # (B, D, H, W)
    if seg.ndim == 5:
        seg = seg.squeeze(1)                    # (B, D, H, W)
    n_classes = pred_logits.shape[1]
    bone_idxs = {bone_idx} if isinstance(bone_idx, int) else set(bone_idx)
    fg_scores = []
    bone_scores = []
    out = {}
    for c in range(1, n_classes):
        a = pred_lab == c
        b = seg == c
        a_sum = int(a.sum().item())
        b_sum = int(b.sum().item())
        if a_sum == 0 and b_sum == 0:
            s = 1.0
        elif a_sum == 0 or b_sum == 0:
            s = 0.0
        else:
            s = 2.0 * int((a & b).sum().item()) / (a_sum + b_sum)
        fg_scores.append(s)
        if c in bone_idxs:
            bone_scores.append(s)
    if bone_scores:
        out["dice_score_bone"] = float(np.mean(bone_scores))
    out["dice_score_all"] = float(np.mean(fg_scores))
    return out


# ---------------------------------------------------------------------------
# Checkpoint metadata (epoch / step / etc.) for logging
# ---------------------------------------------------------------------------
def extract_checkpoint_info(ckpt_path, ckpt_dict=None):
    """Return a dict of available training metadata for `ckpt_path`.

    Sources (in order of preference):
      1. `ckpt_dict` top-level keys saved by BaseTrainer.save_checkpoint:
         `epoch`, `global_step`, `samples_seen`, `elapsed_time` (seconds).
      2. Step embedded in the filename (`*_NNNNNN.pt`).
      3. For `*_last.pt`, the highest step found among sibling step-numbered
         files (e.g. `synthrad_last.pt` → max(`synthrad_060000.pt`, ...)).

    Always returns `filename`. Other keys are present only when known so the
    caller / formatter can skip missing fields cleanly.
    """
    info = {"filename": os.path.basename(ckpt_path)}
    if isinstance(ckpt_dict, dict):
        for k in ("epoch", "global_step", "samples_seen"):
            v = ckpt_dict.get(k)
            if isinstance(v, (int, float)):
                info[k] = int(v) if isinstance(v, int) or v.is_integer() else v
        et = ckpt_dict.get("elapsed_time")
        if isinstance(et, (int, float)):
            info["elapsed_time_s"] = float(et)

    # Parse step from filename: matches "..._<digits>.pt" or "...<digits>.pt"
    m = re.search(r"_(\d{3,})\.pt$", info["filename"]) or re.search(r"(\d{3,})\.pt$", info["filename"])
    if m:
        info["step_from_filename"] = int(m.group(1))

    # *_last.pt → peek at sibling step-numbered files
    if info["filename"].endswith("last.pt"):
        ckpt_dir = os.path.dirname(ckpt_path) or "."
        # Try both naming conventions: <prefix>_last.pt and checkpoint_last.pt
        if info["filename"] == "checkpoint_last.pt":
            patterns = ["*_epoch*.pt", "*_[0-9]*.pt"]
        else:
            prefix = info["filename"].rsplit("_last.pt", 1)[0]
            patterns = [f"{prefix}_*.pt"]
        sib_steps = []
        for pat in patterns:
            for p in glob.glob(os.path.join(ckpt_dir, pat)):
                bn = os.path.basename(p)
                if bn == info["filename"]:
                    continue
                mm = re.search(r"_(\d{3,})\.pt$", bn) or re.search(r"epoch(\d+)\.pt$", bn)
                if mm:
                    sib_steps.append(int(mm.group(1)))
        if sib_steps:
            info["latest_sibling_step"] = max(sib_steps)
    return info


def format_checkpoint_info(info):
    """Human-readable single-line summary suitable for TXT headers / stdout."""
    parts = [f"file={info.get('filename', '?')}"]
    if "epoch" in info:
        parts.append(f"epoch={info['epoch']}")
    if "global_step" in info:
        parts.append(f"global_step={info['global_step']}")
    elif "step_from_filename" in info:
        parts.append(f"step={info['step_from_filename']} (from filename)")
    elif "latest_sibling_step" in info:
        parts.append(f"latest_sibling_step={info['latest_sibling_step']} (from sibling files; last.pt has no embedded step)")
    if "samples_seen" in info:
        parts.append(f"samples_seen={info['samples_seen']}")
    if "elapsed_time_s" in info:
        parts.append(f"train_elapsed={info['elapsed_time_s']/3600:.1f}h")
    return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Output: per-subject TXT report
# ---------------------------------------------------------------------------
def default_validate_dir(ckpt_path, *, prefix="validate"):
    """Sibling of the checkpoint: <ckpt_dir>/<prefix>_<timestamp>/."""
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(ckpt_dir, f"{prefix}_{ts}")


def write_metrics_txt(out_path, *, header_lines, per_subject, metric_keys=None):
    """Write a human-readable TXT report.

    Layout:
        # <header lines...>
        ==================== Per subject ====================
        subj_id        k1=val k2=val ...
        ...
        ==================== Aggregate (n=N) ====================
        k1   mean=...  std=...  median=...  min=...  max=...
        ...

    `per_subject`: list of dicts; each must have `subj_id` and a `metrics` dict
    (numeric values, possibly with optional keys missing per subject — they're
    skipped for that subject and aggregate ignores NaNs / missing).

    `metric_keys`: optional explicit column order. If None, taken from the first
    record's `metrics` keys (in iteration order).
    """
    if metric_keys is None:
        metric_keys = list(per_subject[0]["metrics"].keys()) if per_subject else []

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        f.write("\n")
        f.write("==================== Per subject ====================\n")
        # Header row
        f.write(f"{'subj_id':<24} " + " ".join(f"{k:>14}" for k in metric_keys) + "\n")
        for rec in per_subject:
            subj_id = rec["subj_id"]
            mets = rec["metrics"]
            row = " ".join(
                f"{mets[k]:>14.4f}" if k in mets and mets[k] is not None else f"{'-':>14}"
                for k in metric_keys
            )
            f.write(f"{subj_id:<24} {row}\n")

        # Aggregate
        f.write(f"\n==================== Aggregate (n={len(per_subject)}) ====================\n")
        for k in metric_keys:
            vals = [r["metrics"][k] for r in per_subject
                    if k in r["metrics"] and r["metrics"][k] is not None]
            if not vals:
                f.write(f"{k:<22} (no values)\n")
                continue
            arr = np.array(vals, dtype=np.float64)
            f.write(
                f"{k:<22} mean={arr.mean():>10.4f}  std={arr.std():>10.4f}  "
                f"median={np.median(arr):>10.4f}  min={arr.min():>10.4f}  max={arr.max():>10.4f}  "
                f"(n={len(arr)})\n"
            )

        # Timing block — easier to find than scrolling through the per-metric rows.
        times = [r["metrics"].get("time_sec") for r in per_subject
                 if isinstance(r["metrics"].get("time_sec"), (int, float))]
        if times:
            t_arr = np.array(times, dtype=np.float64)
            total_s = float(t_arr.sum())
            mean_s = float(t_arr.mean())
            f.write("\n==================== Timing ====================\n")
            f.write(f"per-subject mean : {mean_s:8.2f} s   ({mean_s/60:.2f} min)\n")
            f.write(f"per-subject min  : {t_arr.min():8.2f} s\n")
            f.write(f"per-subject max  : {t_arr.max():8.2f} s\n")
            f.write(f"total inference  : {total_s:8.2f} s   ({total_s/60:.2f} min  /  {total_s/3600:.2f} h)\n")
            f.write(f"subjects         : {len(times)}\n")
    print(f"[eval] 📝 Wrote {out_path}")


# ---------------------------------------------------------------------------
# Convenience: small timer that returns elapsed seconds
# ---------------------------------------------------------------------------
class Stopwatch:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.t0
