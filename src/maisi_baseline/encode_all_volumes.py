"""Pre-encode all CT volumes (train + val) to MAISI VAE latent space.

Latents are saved as `{output_dir}/{subj_id}_ct_latent.pt`, ALREADY multiplied
by `scale_factor` (matches the runtime convention in trainer.py:
`ct_emb = self._encode_sliding_window(ct) * self.scale_factor`).

The preprocessing is identical to `MAISITrainer._setup_data` (RAS, clip to
[-1000, 1000] HU → [0, 1], pad-end to multiples of 32), and the encoder is
the SAME SlidingWindowInferer config the trainer uses on the fly. This
guarantees the precomputed latents are bit-equivalent (modulo bf16 noise)
to what on-the-fly mode would produce mid-training.

Usage:
    python src/maisi_baseline/encode_all_volumes.py \\
        --split_file splits/center_wise_split.txt \\
        --output_dir /gpfs/.../1.5mm_registered_flat_masked_maisi_latents

Idempotent: subjects whose latent file already exists are skipped unless
`--force` is passed.
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import torch
from monai.bundle import ConfigParser
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import build_data_dicts, default_monai_cache_dir, get_cached_transforms, get_split_subjects
from common.utils import dynamic_infer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTOENCODER_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "autoencoder_v1.pt")
DIFFUSION_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "diff_unet_3d_rflow-ct.pt")
NETWORK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "NV-Generate-CTMR", "configs", "config_network_rflow.json")
DEFAULT_DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"


def load_autoencoder(device):
    with open(NETWORK_CONFIG_PATH, "r") as f:
        model_def = json.load(f)
    model_def["autoencoder_def"]["num_splits"] = 8
    parser = ConfigParser()
    parser.update(model_def)
    autoencoder = parser.get_parsed_content("autoencoder_def", instantiate=True).to(device)
    ae_ckpt = torch.load(AUTOENCODER_PATH, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ae_ckpt["unet_state_dict"] if "unet_state_dict" in ae_ckpt else ae_ckpt)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False
    # NOTE: no torch.compile — dynamic_infer runs small volumes (e.g. brain) whole-volume,
    # so input shapes vary per subject and would force constant recompilation.
    return autoencoder


def load_scale_factor(device):
    diff_ckpt = torch.load(DIFFUSION_PATH, map_location=device, weights_only=False)
    sf = diff_ckpt.get("scale_factor", 1.0)
    if isinstance(sf, torch.Tensor):
        return sf.to(device)
    return torch.tensor(sf, device=device)


def encode_sliding_window(ct_norm, autoencoder, device, roi_size=(320, 320, 160), sw_batch_size=1, overlap=0.4):
    """Encode CT -> latent. Uses dynamic_infer: whole-volume when the volume fits the
    ROI (the common case here — never pads small brains), else sliding-window with the
    ROI clamped to the volume. ROI [320,320,160] matches NV-Generate-CTMR's encoder.
    """
    inferer = SlidingWindowInferer(
        roi_size=list(roi_size),
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        sw_device=device,
        device=torch.device("cpu"),
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        latent = dynamic_infer(inferer, autoencoder.encode_stage_2_inputs, ct_norm.to(device))
    return latent.to(device)


def main():
    parser = argparse.ArgumentParser(description="Pre-encode MAISI CT latents for all train+val subjects.")
    parser.add_argument("--split_file", type=str, default="splits/center_wise_split.txt")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_DATA_ROOT + "_maisi_latents",
        help="Destination dir on GPFS. Will be created if missing.",
    )
    parser.add_argument("--patch_size", type=int, default=128, help="Match MAISI trainer patch_size (controls pad floor).")
    parser.add_argument("--force", action="store_true", help="Re-encode subjects whose latent already exists.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to encode (default: train val).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Encode] Device: {device}")
    print(f"[Encode] Output dir: {args.output_dir}")

    # Same preprocessing the MAISI trainer uses on the val side (full CT pipeline).
    # Reusing default_monai_cache_dir() warms /tmp for subsequent training jobs.
    cache_dir = default_monai_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    cached_xform = get_cached_transforms(
        patch_size=args.patch_size,
        res_mult=32,
        enforce_ras=True,
        mri_norm="percentile",
        ct_range=(-1000, 1000),
        load_seg=False,
        load_ct_image=True,
        load_ct_latent_from=None,
    )

    # Collect subjects across requested splits.
    all_subjects = []
    for split in args.splits:
        subs = get_split_subjects(args.split_file, split)
        all_subjects.extend(subs)
        print(f"[Encode] Split '{split}': {len(subs)} subjects")
    all_subjects = sorted(set(all_subjects))
    print(f"[Encode] Total unique subjects: {len(all_subjects)}")

    # Skip already-encoded unless --force.
    if args.force:
        todo = all_subjects
    else:
        todo = [s for s in all_subjects if not os.path.exists(os.path.join(args.output_dir, f"{s}_ct_latent.pt"))]
    print(f"[Encode] To encode: {len(todo)} (skipping {len(all_subjects) - len(todo)} already cached)")
    if not todo:
        print("[Encode] Nothing to do.")
        return

    dicts = build_data_dicts(args.data_root, todo, load_seg=False)
    ds = PersistentDataset(
        data=dicts,
        transform=cached_xform,
        cache_dir=cache_dir,
        hash_transform=pickle_hashing,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    print("[Encode] Loading autoencoder + scale_factor...")
    autoencoder = load_autoencoder(device)
    scale_factor = load_scale_factor(device)
    print(f"[Encode] Scale factor: {scale_factor.item():.6f}")

    timings = []
    errors = []
    pbar = tqdm(loader, total=len(todo), desc="Encoding")
    for batch in pbar:
        subj_id = batch["subj_id"][0]
        out_path = os.path.join(args.output_dir, f"{subj_id}_ct_latent.pt")
        try:
            ct = batch["ct"].to(device)
            # Promote storage fp16 (set by get_cached_transforms when use_float16_storage=True
            # — currently off for MAISI by default, but be defensive) to fp32 before encode.
            if not ct.is_floating_point() or ct.dtype != torch.float32:
                ct = ct.float()
            ct = ct.clamp(0.0, 1.0)
            t0 = time.time()
            latent = encode_sliding_window(ct, autoencoder, device)
            latent_scaled = latent * scale_factor  # save pre-scaled; trainer uses it directly.
            elapsed = time.time() - t0

            # Strip MetaTensor subclass — LoadLatentd uses weights_only=True at
            # train time, which refuses MONAI subclasses without explicit allowlisting.
            if hasattr(latent_scaled, "as_tensor"):
                latent_scaled = latent_scaled.as_tensor()
            torch.save(latent_scaled.cpu(), out_path)
            timings.append({"subj_id": subj_id, "elapsed_s": round(elapsed, 3),
                            "input_shape": list(ct.shape), "latent_shape": list(latent_scaled.shape)})
            pbar.set_postfix({"sid": subj_id, "t": f"{elapsed:.1f}s"})

            del ct, latent, latent_scaled
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            errors.append({"subj_id": subj_id, "error": str(e)})
            tqdm.write(f"  [ERROR] {subj_id}: {e}")

    # Summary
    if timings:
        ts = np.array([t["elapsed_s"] for t in timings])
        print(f"\n[Encode] Done — encoded {len(timings)} | errors {len(errors)}")
        print(f"[Encode] Per-subject encode time: min={ts.min():.1f}s mean={ts.mean():.1f}s max={ts.max():.1f}s")
        print(f"[Encode] Total wall time on encode loop: {ts.sum() / 60:.1f} min")
        # Save timing log alongside latents for later analysis.
        log_path = os.path.join(args.output_dir, "encode_timings.json")
        with open(log_path, "w") as f:
            json.dump({"timings": timings, "errors": errors}, f, indent=2)
        print(f"[Encode] Timing log: {log_path}")
    if errors:
        print(f"\n[Encode] Failed subjects ({len(errors)}):")
        for e in errors:
            print(f"  - {e['subj_id']}: {e['error']}")


if __name__ == "__main__":
    main()
