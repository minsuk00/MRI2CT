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
    dtype=torch.bfloat16,
):
    """Load + freeze the Baby U-Net teacher used for Dice eval.

    Mirrors BaseTrainer._setup_teacher_model but takes explicit args so callers
    don't need a Trainer/Config in scope. `n_classes_minus_bg` matches the
    `cfg.n_classes - 1` convention in the trainer (DEFAULT_CONFIG.n_classes=12).
    """
    from anatomix.segmentation.segmentation_utils import load_model_v1_2

    print(f"[eval] 👨‍🏫 Loading teacher from {weights_path}")
    teacher = load_model_v1_2(
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


def compute_dice(pred_logits, seg, *, mask=None, bone_idx=5, exclude_background=True):
    """Compute {dice_score_all, dice_score_bone} from teacher logits + GT seg.

    `pred_logits`: (B, C, D, H, W) raw teacher output on the prediction volume.
    `seg`:        (B, 1, D, H, W) integer GT label map at the same spatial shape.
    `mask`:       optional (B, 1, D, H, W) body mask — restricts the Dice to body
                  voxels (matches `compute_metrics_body`).
    """
    class_dices, bone_dice = get_class_dice(pred_logits, seg, mask=mask, bone_idx=bone_idx)
    out = {}
    out["dice_score_all"] = (
        class_dices[1:].mean() if exclude_background else class_dices.mean()
    ).item()
    if bone_dice is not None:
        out["dice_score_bone"] = bone_dice.item()
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
