"""
Pre-encode all CT volumes to MAISI VAE latent space and save to disk.
Supports distributed encoding across multiple SLURM array nodes.
Saves per-subject timing to a JSON file for later analysis.

Usage (single node):
    python src/maisi_baseline/encode_all_volumes.py

Usage (SLURM array, 5 nodes):
    python src/maisi_baseline/encode_all_volumes.py --node_id $SLURM_ARRAY_TASK_ID --num_nodes 5
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
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import build_tio_subjects
from maisi_baseline.trainer import MAISITrainer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
AUTOENCODER_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "autoencoder_v1.pt")
NETWORK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "NV-Generate-CTMR", "configs", "config_network_rflow.json")


def encode_sliding_window(ct_tensor, autoencoder, scale_factor, device, sw_batch_size=1, overlap=0.4):
    ct_hu = (ct_tensor * 2048.0) - 1024.0
    ct_norm = torch.clamp((ct_hu + 1000.0) / 2000.0, 0.0, 1.0)
    inferer = SlidingWindowInferer(
        roi_size=[256, 256, 256],
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        sw_device=device,
        device=device,
    )
    with torch.no_grad(), torch.amp.autocast("cuda"):
        latent = inferer(ct_norm.to(device), autoencoder.encode_stage_2_inputs)
    return latent.to(device) * scale_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/home/minsukc/MRI2CT/dataset/1.5mm_registered_maisi_encoding")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--node_id", type=int, default=0, help="0-indexed node ID for SLURM array")
    parser.add_argument("--num_nodes", type=int, default=1, help="Total number of parallel nodes")
    parser.add_argument("--force", action="store_true", default=False, help="Re-encode even if latent already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Node {args.node_id}/{args.num_nodes}] Device: {device}")

    # Load autoencoder
    print("Loading autoencoder...")
    with open(NETWORK_CONFIG_PATH, "r") as f:
        model_def = json.load(f)
    parser_cfg = ConfigParser()
    parser_cfg.update(model_def)
    autoencoder = parser_cfg.get_parsed_content("autoencoder_def", instantiate=True).to(device)
    ae_ckpt = torch.load(AUTOENCODER_PATH, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ae_ckpt["unet_state_dict"] if "unet_state_dict" in ae_ckpt else ae_ckpt)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Load scale factor from diffusion unet checkpoint
    diff_path = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "diff_unet_3d_rflow-ct.pt")
    diff_ckpt = torch.load(diff_path, map_location=device, weights_only=False)
    if "scale_factor" in diff_ckpt:
        scale_factor = diff_ckpt["scale_factor"]
        if not isinstance(scale_factor, torch.Tensor):
            scale_factor = torch.tensor(scale_factor, device=device)
        scale_factor = scale_factor.to(device)
    else:
        scale_factor = torch.tensor(1.0, device=device)
    print(f"Scale factor: {scale_factor.item():.6f}")

    # Discover and shard subjects across nodes
    all_subjects = sorted(os.listdir(args.data_root))
    my_subjects = all_subjects[args.node_id::args.num_nodes]
    print(f"[Node {args.node_id}] Total subjects: {len(all_subjects)} | This node: {len(my_subjects)}")

    if not args.force:
        to_encode = [s for s in my_subjects
                     if not os.path.exists(os.path.join(args.output_dir, f"{s}_ct_latent.pt"))]
        skipped = len(my_subjects) - len(to_encode)
        print(f"To encode: {len(to_encode)} | Already cached (skipped): {skipped}")
    else:
        to_encode = my_subjects
        print(f"Force re-encode: {len(to_encode)} subjects")

    # Load existing timing file for this node if resuming
    timing_path = os.path.join(args.output_dir, f"timing_node{args.node_id}.json")
    if os.path.exists(timing_path) and not args.force:
        with open(timing_path, "r") as f:
            timing_records = json.load(f)
    else:
        timing_records = {}

    errors = []

    for sid in tqdm(to_encode, desc=f"Node {args.node_id}"):
        out_path = os.path.join(args.output_dir, f"{sid}_ct_latent.pt")
        try:
            subj_objs = build_tio_subjects(args.data_root, [sid], use_weighted_sampler=False, load_seg=False)
            if not subj_objs:
                errors.append((sid, "build_tio_subjects returned empty"))
                continue

            subj_obj = MAISITrainer._resample_subject(subj_objs[0])
            ct = subj_obj["ct"].data.unsqueeze(0).to(device)
            input_shape = list(ct.shape)

            t0 = time.time()
            ct_emb = encode_sliding_window(ct, autoencoder, scale_factor, device)
            elapsed = time.time() - t0

            torch.save(ct_emb.cpu(), out_path)
            tqdm.write(f"  {sid}: {input_shape} -> latent {list(ct_emb.shape)} | {elapsed:.1f}s")

            timing_records[sid] = {
                "elapsed_s": round(elapsed, 3),
                "input_shape": input_shape,
                "latent_shape": list(ct_emb.shape),
            }
            # Save after each volume so partial runs are preserved
            with open(timing_path, "w") as f:
                json.dump(timing_records, f, indent=2)

            del ct, ct_emb
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            errors.append((sid, str(e)))
            tqdm.write(f"  [ERROR] {sid}: {e}")

    # Summary
    encode_times = [v["elapsed_s"] for v in timing_records.values()]
    print("\n" + "=" * 50)
    print(f"[Node {args.node_id}] Encoded: {len(encode_times)} | Errors: {len(errors)}")
    if encode_times:
        print(f"Encode time — Min: {np.min(encode_times):.1f}s | Mean: {np.mean(encode_times):.1f}s | Max: {np.max(encode_times):.1f}s")
        print(f"Total encode time: {sum(encode_times)/60:.1f} min")
    if errors:
        print(f"\nFailed subjects:")
        for sid, msg in errors:
            print(f"  {sid}: {msg}")
    print(f"\nLatents saved to: {args.output_dir}")
    print(f"Timing saved to: {timing_path}")


if __name__ == "__main__":
    main()
