"""Standalone post-training evaluator for amix on SynthRAD MR→CT.

Mirrors `AmixTrainer.validate()` exactly — same combined_forward(feat→translator)
sliding-window inference, same teacher Dice, same metric definitions — but
without the surrounding training scaffolding (no WandB, no optimizer, no
augmentation, no loss). Reads the architecture flags (`anatomix_weights`,
`pass_mri_to_translator`, `feat_instance_norm`, `feat_scale_down`, `res_mult`,
`val_patch_size`, etc.) directly from the checkpoint's saved config so the
caller doesn't need to know the run's hyperparameters.

Output layout matches MAISI/cWDM/MC-DDPM validators so the same
`merge_validate_shards.py` can combine them all.
"""
import argparse
import gc
import os
import sys
import time

import nibabel as nib
import torch
import torch.nn.functional as F
from anatomix.model.network import Unet
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_split_subjects,
)
from common.eval_utils import (
    compute_dice,
    default_validate_dir,
    extract_checkpoint_info,
    format_checkpoint_info,
    load_teacher_model,
    run_teacher_sw,
    write_metrics_txt,
)
from common.utils import (
    clean_state_dict,
    compute_metrics,
    compute_metrics_body,
    unpad,
)


DEFAULT_TEACHER = "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth"


# ---------------------------------------------------------------------------
# Build feat_extractor + translator exactly as AmixTrainer does
# ---------------------------------------------------------------------------
def build_amix_models(cfg, device):
    aw = cfg.get("anatomix_weights", "v1_4")
    print(f"[VAL-AMIX] 🏗️ Building Anatomix ({aw})")
    if aw == "v1":
        res_mult = 16
        feat = Unet(3, 1, 16, 4, 16).to(device)
        feat_ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
    elif aw in ("v2", "v1_2", "v1_3"):
        res_mult = 32
        feat_norm = cfg.get("feat_norm", "instance")
        feat = Unet(3, 1, 16, 5, 20, norm=feat_norm, interp="trilinear",
                    pooling="Avg", use_bias=True).to(device)
        feat_ckpt = ("/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_real_v1_3.pth"
                     if aw == "v1_3" else
                     "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth")
    elif aw == "v1_4":
        res_mult = 16
        feat = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest",
                    pooling="Max").to(device)
        feat_ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
    else:
        raise ValueError(f"Invalid anatomix_weights: {aw}")
    if os.path.exists(feat_ckpt):
        feat.load_state_dict(clean_state_dict(torch.load(feat_ckpt, map_location=device)), strict=True)
        print(f"[VAL-AMIX] Loaded feat weights from {feat_ckpt}")
    else:
        print(f"[VAL-AMIX] ⚠️ feat weights NOT FOUND at {feat_ckpt}")
    for p in feat.parameters():
        p.requires_grad = False
    feat.eval()

    # Translator
    translator_input_nc = 17 if cfg.get("pass_mri_to_translator", False) else 16
    print(f"[VAL-AMIX] 🏗️ Building Translator (input_nc={translator_input_nc})")
    translator = Unet(
        dimension=3, input_nc=translator_input_nc, output_nc=1,
        num_downs=4, ngf=16, final_act="sigmoid",
    ).to(device)
    return feat, translator, res_mult


def make_combined_forward(feat_extractor, translator, cfg):
    """Mirror AmixTrainer.combined_forward (val path: dropout disabled)."""
    pass_mri = cfg.get("pass_mri_to_translator", False)
    feat_instance_norm = cfg.get("feat_instance_norm", False)
    feat_scale = cfg.get("feat_scale_down", 1) or 1

    def fn(x):
        f = feat_extractor(x)
        if feat_instance_norm:
            f = F.instance_norm(f)
        if feat_scale != 1:
            f = f / feat_scale
        if pass_mri:
            f = torch.cat([f, x], dim=1)
            # input_dropout_p only fires during training (training=False at val), so skip
        return translator(f)
    return fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_file", default=None,
                        help="Defaults to the split_file recorded in the checkpoint config.")
    parser.add_argument("--split_name", default="val")
    parser.add_argument("--root_dir", default=None,
                        help="Defaults to DEFAULT_CONFIG['root_dir'] if not in checkpoint config.")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir; defaults to <ckpt_dir>/validate_<ts>/")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    # Teacher / Dice
    parser.add_argument("--teacher_weights_path", default=DEFAULT_TEACHER,
                        help="'none' to disable Dice.")
    parser.add_argument("--n_classes", type=int, default=12)
    parser.add_argument("--dice_bone_idx", type=int, default=5)
    parser.add_argument("--teacher_sw_batch_size", type=int, default=2)
    parser.add_argument("--teacher_sw_overlap", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_on = bool(args.teacher_weights_path and args.teacher_weights_path.lower() != "none")

    # Load checkpoint (contains config + model state)
    print(f"[VAL-AMIX] 📥 Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    ckpt_info = extract_checkpoint_info(args.checkpoint, ckpt_dict=ckpt)
    ckpt_info_str = format_checkpoint_info(ckpt_info)
    print(f"[VAL-AMIX] checkpoint: {ckpt_info_str}")

    # Resolve split/root
    if args.split_file is None:
        args.split_file = cfg.get("split_file", "splits/center_wise_split.txt")
    if args.root_dir is None:
        from common.config import DEFAULT_CONFIG
        args.root_dir = cfg.get("root_dir", DEFAULT_CONFIG["root_dir"])

    if args.out_dir is None:
        args.out_dir = default_validate_dir(args.checkpoint, prefix="validate")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[VAL-AMIX] output dir: {args.out_dir}")

    # Build models from saved config
    feat_extractor, translator, res_mult = build_amix_models(cfg, device)
    state = clean_state_dict(ckpt["model_state_dict"])
    translator.load_state_dict(state, strict=True)
    translator.eval()
    for p in translator.parameters():
        p.requires_grad = False

    combined = make_combined_forward(feat_extractor, translator, cfg)

    # Teacher
    teacher = None
    if dice_on:
        teacher = load_teacher_model(
            args.teacher_weights_path, device=device, n_classes_minus_bg=args.n_classes - 1
        )

    # Val data — match training preset: ct_range=(-1024, 1024), mri_norm from cfg
    val_subj = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects is not None:
        val_subj = val_subj[: args.max_subjects]
    if args.num_shards and args.num_shards > 1:
        val_subj = val_subj[args.shard_idx :: args.num_shards]
        print(f"[VAL-AMIX] shard {args.shard_idx}/{args.num_shards} → {len(val_subj)} subjects")
    val_dicts = build_data_dicts(args.root_dir, val_subj, load_seg=dice_on, load_body_mask=True)
    print(f"[VAL-AMIX] 📂 {len(val_dicts)} subjects (split={args.split_name})")

    val_xform = get_cached_transforms(
        patch_size=cfg.get("patch_size", 128),
        res_mult=res_mult,
        enforce_ras=True,
        mri_norm=cfg.get("mri_norm", "minmax"),
        ct_range=tuple(cfg.get("ct_range", (-1024, 1024))),
        load_seg=dice_on,
        load_body_mask=True,
    )
    cache_dir = default_monai_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    ds = PersistentDataset(data=val_dicts, transform=val_xform, cache_dir=cache_dir, hash_transform=pickle_hashing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    val_ps = cfg.get("val_patch_size", 256)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.25)
    val_sw_batch_size = cfg.get("val_sw_batch_size", 1)

    per_subject = []
    for batch in tqdm(loader, desc="validate", dynamic_ncols=True):
        subj_id = batch["subj_id"][0]
        mri = batch["mri"].to(device).float()
        ct  = batch["ct"].to(device).float()
        body_mask = batch["body_mask"].to(device).float() if "body_mask" in batch else None
        seg = batch["seg"].to(device).long() if (dice_on and "seg" in batch) else None
        orig_shape = batch["original_shape"][0].tolist()
        affine = batch["ct_affine"][0].cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        # fp32 inference (no bf16 autocast): bf16's 7-bit mantissa quantizes a
        # single-pass regressor's output to ~4-8 HU steps, which posterizes soft
        # tissue when viewed in a narrow window (e.g. brain [-100,100] HU).
        pred = sliding_window_inference(
            inputs=mri, roi_size=(val_ps,) * 3,
            sw_batch_size=val_sw_batch_size,
            predictor=combined, overlap=val_sw_overlap, device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()  # CUDA is async; sync so time_sec reflects real GPU compute
        elapsed = time.time() - t0

        # Unpad before metrics
        pred_unpad = unpad(pred.float(), orig_shape)
        ct_unpad   = unpad(ct, orig_shape)
        mask_unpad = unpad(body_mask, orig_shape) if body_mask is not None else None
        seg_unpad  = unpad(seg, orig_shape) if seg is not None else None

        # Image metrics (amix uses ct_range [-1024, 1024] → hu_range 2048)
        met = compute_metrics(pred_unpad, ct_unpad, hu_range=2048)
        record = {"mae_hu": met["mae_hu"], "psnr": met["psnr"],
                  "ssim": met["ssim"], "grad_diff": met["grad_diff"]}

        if mask_unpad is not None:
            bm = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad, hu_range=2048)
            record["body_mae_hu"] = bm["mae_hu"]
            record["body_psnr"]   = bm["psnr"]
            record["body_ssim"]   = bm["ssim"]

        # Dice
        if teacher is not None and seg_unpad is not None:
            pred_logits = run_teacher_sw(
                teacher, pred_unpad, device=device,
                val_patch_size=val_ps,
                sw_batch_size=args.teacher_sw_batch_size,
                overlap=args.teacher_sw_overlap,
            )
            d = compute_dice(pred_logits, seg_unpad, bone_idx=args.dice_bone_idx)
            record.update(d)
            if mask_unpad is not None:
                db = compute_dice(pred_logits, seg_unpad, mask=mask_unpad, bone_idx=args.dice_bone_idx)
                record["body_dice_score_all"]  = db["dice_score_all"]
                if "dice_score_bone" in db:
                    record["body_dice_score_bone"] = db["dice_score_bone"]
            del pred_logits

        record["time_sec"] = elapsed
        per_subject.append({"subj_id": subj_id, "metrics": record})

        # NIfTI (HU = pred * 2048 - 1024)
        subj_dir = os.path.join(args.out_dir, subj_id)
        os.makedirs(subj_dir, exist_ok=True)
        pred_hu = (pred_unpad * 2048.0 - 1024.0).float().cpu().numpy().squeeze()
        ct_hu   = (ct_unpad   * 2048.0 - 1024.0).float().cpu().numpy().squeeze()
        nib.save(nib.Nifti1Image(pred_hu, affine), os.path.join(subj_dir, "sample.nii.gz"))
        nib.save(nib.Nifti1Image(ct_hu,   affine), os.path.join(subj_dir, "target.nii.gz"))

        tqdm.write(
            f"  {subj_id} | {elapsed:6.1f}s | MAE={record['mae_hu']:6.1f}HU "
            f"PSNR={record['psnr']:5.2f} SSIM={record['ssim']:.3f}"
            + (f" | dice_all={record.get('dice_score_all', float('nan')):.3f}"
               f" bone={record.get('dice_score_bone', float('nan')):.3f}"
               if teacher else "")
        )

        del pred, pred_unpad, ct_unpad, mri, ct
        gc.collect()
        torch.cuda.empty_cache()

    # TXT report
    metric_keys = []
    for r in per_subject:
        for k in r["metrics"]:
            if k not in metric_keys:
                metric_keys.append(k)
    header = [
        "Validation report — amix",
        f"checkpoint: {args.checkpoint}",
        f"checkpoint info: {ckpt_info_str}",
        f"split_file: {args.split_file}   split_name: {args.split_name}",
        f"anatomix_weights: {cfg.get('anatomix_weights', '?')}   "
        f"pass_mri_to_translator: {cfg.get('pass_mri_to_translator', False)}   "
        f"feat_instance_norm: {cfg.get('feat_instance_norm', False)}",
        f"val_patch_size: {val_ps}   val_sw_overlap: {val_sw_overlap}   val_sw_batch_size: {val_sw_batch_size}",
        f"teacher: {args.teacher_weights_path if teacher is not None else 'disabled'}",
        f"subjects: {len(per_subject)}",
    ]
    if args.num_shards and args.num_shards > 1:
        header.insert(1, f"shard: {args.shard_idx}/{args.num_shards}")
    write_metrics_txt(
        os.path.join(args.out_dir, "validate_metrics.txt"),
        header_lines=header, per_subject=per_subject, metric_keys=metric_keys,
    )


if __name__ == "__main__":
    main()
