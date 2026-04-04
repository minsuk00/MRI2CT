"""
Runs evaluate_models.py over 5 epochs (600-604) for 4 models,
both with and without body mask, then prints averaged results.

Usage:
    python src/evaluate/run_epoch_avg_eval.py
    python src/evaluate/run_epoch_avg_eval.py --output /tmp/epoch_avg_results.json
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict

import numpy as np

WANDB = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/wandb"
GPFS_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
SPLIT_FILE = "splits/thorax_center_wise_split.txt"
EPOCHS = [600, 601, 602, 603, 604]

MODELS = {
    "ct5bgykw": {
        "type": "amix",
        "checkpoints": {
            600: f"{WANDB}/run-20260330_000151-ct5bgykw/files/anatomix_translator_epoch00600.pt",
            601: f"{WANDB}/run-20260401_012548-ct5bgykw/files/anatomix_translator_epoch00601.pt",
            602: f"{WANDB}/run-20260401_012548-ct5bgykw/files/anatomix_translator_epoch00602.pt",
            603: f"{WANDB}/run-20260401_012548-ct5bgykw/files/anatomix_translator_epoch00603.pt",
            604: f"{WANDB}/run-20260401_012548-ct5bgykw/files/anatomix_translator_epoch00604.pt",
        },
    },
    "5aim9k43": {
        "type": "amix",
        "checkpoints": {ep: f"{WANDB}/run-20260401_012526-5aim9k43/files/anatomix_translator_epoch00{ep}.pt" for ep in EPOCHS},
    },
    "w4suexvx": {
        "type": "amix",
        "checkpoints": {ep: f"{WANDB}/run-20260401_012548-w4suexvx/files/anatomix_translator_epoch00{ep}.pt" for ep in EPOCHS},
    },
    "m3nvntkd": {
        "type": "unet",
        "checkpoints": {ep: f"{WANDB}/run-20260330_000210-m3nvntkd/files/unet_baseline_epoch00{ep}.pt" for ep in EPOCHS},
    },
}

COLS = ["mae_hu", "ssim", "psnr", "dice_score_all", "dice_score_bone"]
COL_NAMES = ["MAE HU", "SSIM", "PSNR", "Dice (All)", "Dice (Bone)"]


def run_eval(checkpoints_spec, body_mask, output_json):
    """Call evaluate_models.py and return parsed results list."""
    cmd = [
        "micromamba",
        "run",
        "-n",
        "mrct",
        "python",
        "src/evaluate/evaluate_models.py",
        "--split_file",
        SPLIT_FILE,
        "--checkpoints",
        *checkpoints_spec,
        "--output",
        output_json,
    ]
    if body_mask:
        cmd.append("--body_mask")

    print(f"  Running: {' '.join(cmd[-6:])}")  # show tail of command
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] eval failed (exit {result.returncode})")
        return []

    with open(output_json) as f:
        return json.load(f)


def average_results(all_results):
    """Group by label (run_id) and average metrics across epochs."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        label = r["label"] + "/" + r.get("run_id", r["ckpt_name"])
        for k, v in r["metrics"].items():
            grouped[label][k].append(v)
    return {label: {k: float(np.mean(v)) for k, v in metrics.items()} for label, metrics in grouped.items()}


def print_table(avg, title, out=None):
    import sys

    f = out if out is not None else sys.stdout
    f.write(f"\n{'=' * 70}\n")
    f.write(f"{title}\n")
    f.write(f"{'=' * 70}\n")
    header = f"{'Model':<12}" + "".join(f"  {n:>11}" for n in COL_NAMES)
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for label, metrics in avg.items():
        vals = [metrics.get(c, float("nan")) for c in COLS]
        f.write(f"{label:<12}" + "".join(f"  {v:>11.4f}" for v in vals) + "\n")
    f.write("-" * len(header) + "\n")


def run_viz(checkpoint_specs, output_dir, body_mask=False):
    """Call visualize_predictions.py with one representative checkpoint per model."""
    suffix = "viz_masked" if body_mask else "viz"
    viz_dir = os.path.join(output_dir, suffix)
    cmd = [
        "micromamba",
        "run",
        "-n",
        "mrct",
        "python",
        "src/evaluate/visualize_predictions.py",
        "--checkpoints",
        *checkpoint_specs,
        "--output_dir",
        viz_dir,
        "--root_dir",
        GPFS_ROOT,
    ]
    if body_mask:
        cmd.append("--body_mask")
    print(f"  Running: {' '.join(cmd[-5:])}")
    subprocess.run(cmd, capture_output=False, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results txt and json files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_results_no_mask = []
    all_results_with_mask = []

    for run_id, info in MODELS.items():
        model_type = info["type"]
        print(f"\n{'=' * 60}")
        print(f"Model: {run_id} ({model_type}), epochs {EPOCHS}")

        ckpt_specs = [f"{model_type}:{info['checkpoints'][ep]}" for ep in EPOCHS if os.path.exists(info["checkpoints"][ep])]
        missing = [ep for ep in EPOCHS if not os.path.exists(info["checkpoints"][ep])]
        if missing:
            print(f"  [WARNING] Missing epochs: {missing}")
        if not ckpt_specs:
            print("  [SKIP] No checkpoints found.")
            continue

        out_no_mask = os.path.join(args.output_dir, f"{run_id}_no_mask.json")
        print("\n--- WITHOUT body mask ---")
        results = run_eval(ckpt_specs, body_mask=False, output_json=out_no_mask)
        for r in results:
            r["run_id"] = run_id
        all_results_no_mask.extend(results)

        out_with_mask = os.path.join(args.output_dir, f"{run_id}_with_mask.json")
        print("\n--- WITH body mask ---")
        results = run_eval(ckpt_specs, body_mask=True, output_json=out_with_mask)
        for r in results:
            r["run_id"] = run_id
        all_results_with_mask.extend(results)

    avg_no_mask = average_results(all_results_no_mask)
    avg_with_mask = average_results(all_results_with_mask)

    # Print and save tables
    import io

    for avg, title, fname in [
        (avg_no_mask, "AVERAGED RESULTS — No Body Mask (epochs 600-604)", "results_no_mask.txt"),
        (avg_with_mask, "AVERAGED RESULTS — With Body Mask (epochs 600-604)", "results_with_mask.txt"),
    ]:
        buf = io.StringIO()
        print_table(avg, title, out=buf)
        text = buf.getvalue()
        print(text)
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            f.write(text)
        print(f"[INFO] Saved {path}")

    # Save raw averaged numbers as JSON too
    json_path = os.path.join(args.output_dir, "results_avg.json")
    with open(json_path, "w") as f:
        json.dump({"epochs": EPOCHS, "no_mask": avg_no_mask, "with_mask": avg_with_mask}, f, indent=2)
    print(f"[INFO] Saved {json_path}")

    # Visualization: pick median epoch per model as representative
    print(f"\n{'=' * 60}")
    print("Running visualization (median epoch per model)...")
    mid_ep = EPOCHS[len(EPOCHS) // 2]
    viz_specs = []
    for run_id, info in MODELS.items():
        ckpt = info["checkpoints"].get(mid_ep)
        if ckpt and os.path.exists(ckpt):
            viz_specs.append(f"{info['type']}:{ckpt}")
        else:
            # Fallback to first available epoch
            for ep in EPOCHS:
                fb = info["checkpoints"].get(ep)
                if fb and os.path.exists(fb):
                    viz_specs.append(f"{info['type']}:{fb}")
                    break

    if viz_specs:
        run_viz(viz_specs, args.output_dir, body_mask=False)
        run_viz(viz_specs, args.output_dir, body_mask=True)
    else:
        print("[WARNING] No checkpoints available for visualization.")


if __name__ == "__main__":
    main()
