import subprocess
import itertools
import csv
import os
import time
import re
from tqdm import tqdm
import numpy as np

# --- 1. Define Hyperparameters ---
subject = "1ABA005_3.0x3.0x3.0_resampled"
epochs = 50
# epochs = 1

# Common Settings
use_segs = [True, False]

# Loss Combinations
loss_configs = [
    {"l1": 1.0, "l2": 0.0, "ssim": 0.0, "name": "L1_Only"},
    # {"l1": 0.0, "l2": 1.0, "ssim": 0.0, "name": "L2_Only"},
    # {"l1": 1.0, "l2": 0.0, "ssim": 1.0, "name": "L1_SSIM_1.0"},
    {"l1": 1.0, "l2": 0.0, "ssim": 0.1, "name": "L1_SSIM_0.1"},
]

# MLP Specifics
fourier_options = [True, False]
sigmas = [1.0, 10.0, 20.0]

# CNN Specifics
cnn_depths = [3, 5]
cnn_hiddens = [32, 64]
cnn_activations = ["relu_clamp", "sigmoid", "none"]

# --- 2. Generate Valid Experiments Lists ---
experiments = []

# --- A. Generate MLP Experiments ---
mlp_combos = list(itertools.product(loss_configs, use_segs, fourier_options, sigmas))
for loss_conf, seg, fourier, sigma in mlp_combos:
    # Rule: MLP cannot use SSIM
    if loss_conf["ssim"] > 0: continue
    # Rule: No Fourier means Sigma is irrelevant (run only once)
    if not fourier and sigma != sigmas[0]: continue

    experiments.append({
        "type": "mlp", "loss": loss_conf, "seg": seg, 
        "fourier": fourier, "sigma": sigma,
        # CNN params are N/A for MLP
        "depth": "N/A", "hidden": "N/A", "act": "N/A"
    })

# --- B. Generate CNN Experiments ---
cnn_combos = list(itertools.product(loss_configs, use_segs, cnn_depths, cnn_hiddens, cnn_activations))
for loss_conf, seg, depth, hidden, act in cnn_combos:
    # CNN supports all losses, including SSIM. No filtering needed there.
    experiments.append({
        "type": "cnn", "loss": loss_conf, "seg": seg,
        "depth": depth, "hidden": hidden, "act": act,
        # MLP params are N/A for CNN
        "fourier": "N/A", "sigma": "N/A"
    })

print(f"✅ Generated {len(experiments)} total valid experiments.")

# --- 3. Setup Logging ---
csv_file = "experiment_results.csv"
# Added SSIM and PSNR columns
fieldnames = ["timestamp", "model", "loss_name", "seg", "fourier", "sigma", 
              "cnn_depth", "cnn_hidden", "activation", 
              "avg_mae", "avg_ssim", "avg_psnr", "wandb_url", "status"]

if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def log_result(data):
    with open(csv_file, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writerow(data)

results_history = []

# --- 4. The Loop ---
pbar = tqdm(experiments, desc="Grid Search", unit="job")

for exp in pbar:
    pbar.set_postfix_str(f"{exp['type'].upper()} | {exp['loss']['name']}")

    # Base Command
    cmd = [
        "python", "-u", "src/train_single.py",
        "--subject", subject,
        "--epochs", str(epochs),
        "--model_type", exp["type"],
        "--l1_w", str(exp["loss"]["l1"]),
        "--l2_w", str(exp["loss"]["l2"]),
        "--ssim_w", str(exp["loss"]["ssim"]),
        "--val_interval", "10"
    ]

    if exp["seg"]: cmd.append("--use_seg")

    # Append Model Specific Args
    if exp["type"] == "mlp":
        if not exp["fourier"]:
            cmd.append("--no_fourier")
        else:
            cmd.extend(["--sigma", str(exp["sigma"])])
            
    elif exp["type"] == "cnn":
        cmd.extend(["--cnn_depth", str(exp["depth"])])
        cmd.extend(["--cnn_hidden", str(exp["hidden"])])
        cmd.extend(["--final_activation", exp["act"]])

    # --- Run & Capture ---
    start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    wandb_url = "N/A"
    status = "Success"
    val_metrics = {"mae": [], "ssim": [], "psnr": []}


    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        for line in process.stdout:
            pbar.write(line.strip())
            
            # Capture W&B
            if "__WANDB_URL__" in line:
                wandb_url = line.strip().split(":", 1)[1]
            
            # Capture MAE (looks for "MAE: 0.1234")
            for metric in ["mae", "ssim", "psnr"]:
                match = re.search(rf"{metric.upper()}:\s*([0-9.]+)", line, re.IGNORECASE)
                if match:
                    try:
                        val_metrics[metric].append(float(match.group(1)))
                    except ValueError: pass

        process.wait()
        if process.returncode != 0: 
            status = "Failed"
            final_mae = float('inf')

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user."); break
    except Exception as e:
        print(f"❌ Error: {e}"); status = "Error"

    def get_avg(metric_list):
        if not metric_list: return "N/A"
        # Take last 5, or whatever is available if < 5
        return np.mean(metric_list[-5:])

    avg_mae = get_avg(val_metrics["mae"])
    avg_ssim = get_avg(val_metrics["ssim"])
    avg_psnr = get_avg(val_metrics["psnr"])
    
    # --- Log Data ---
    row = {
        "timestamp": start_ts,
        "model": exp["type"],
        "loss_name": exp["loss"]["name"],
        "seg": exp["seg"],
        "fourier": exp["fourier"],
        "sigma": exp["sigma"],
        "cnn_depth": exp["depth"],
        "cnn_hidden": exp["hidden"],
        "activation": exp["act"],
        "avg_mae": avg_mae,
        "avg_ssim": avg_ssim,
        "avg_psnr": avg_psnr,
        "wandb_url": wandb_url,
        "status": status
    }
    
    log_result(row)
    if status == "Success": results_history.append(row)

# --- 5. Leaderboard Summary ---
print("\n" + "="*60)
print("🏆 FINAL LEADERBOARDS (Avg Last 5 Epochs)")
print("="*60)

valid_results = [r for r in results_history if r['avg_mae'] != "N/A"]

# Helper function to print tables
def print_ranking(results, metric_key, metric_name, reverse=False, category="GLOBAL"):
    # reverse=False for MAE (Lower is better), True for SSIM/PSNR (Higher is better)
    sorted_res = sorted(results, key=lambda x: x[metric_key], reverse=reverse)
    
    print(f"\n📊 {category} - TOP 5 {metric_name} {'(Higher is better)' if reverse else '(Lower is better)'}")
    print("-" * 60)
    for i, res in enumerate(sorted_res[:5]):
        desc = f"{res['model'].upper()} + {res['loss_name']}"
        if res['seg']: desc += " + Seg"
        if res['model'] == 'cnn':
            desc += f" | D={res['cnn_depth']} H={res['cnn_hidden']} {res['activation']}"
        elif res['model'] == 'mlp':
            is_fourier = res['fourier'] == 'True' or res['fourier'] is True
            four_str = f"Four(σ={res['sigma']})" if is_fourier else "NoFour"
            desc += f" | {four_str}"
            
        val = res[metric_key]
        print(f"{i+1}. {metric_name}: {val:.5f} | {desc}")
    print("-" * 60)

# Run rankings
if valid_results:
    # 1. Global Rankings
    print_ranking(valid_results, "avg_mae", "MAE", reverse=False, category="ALL MODELS")
    print_ranking(valid_results, "avg_ssim", "SSIM", reverse=True, category="ALL MODELS")
    print_ranking(valid_results, "avg_psnr", "PSNR", reverse=True, category="ALL MODELS")
    
    # 2. Per-Model Rankings
    for model_type in ["mlp", "cnn"]:
        model_res = [r for r in valid_results if r['model'] == model_type]
        if model_res:
            print(f"\n🔎 {model_type.upper()} SPECIFIC RANKINGS")
            print_ranking(model_res, "avg_mae", "MAE", reverse=False, category=f"{model_type.upper()} ONLY")
            print_ranking(model_res, "avg_ssim", "SSIM", reverse=True, category=f"{model_type.upper()} ONLY")
            print_ranking(model_res, "avg_psnr", "PSNR", reverse=True, category=f"{model_type.upper()} ONLY")
else:
    print("No valid results to rank.")