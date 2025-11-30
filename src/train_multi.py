#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")

import argparse
import os
import time
import yaml
from datetime import datetime
import torch
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
import random

from anatomix.model.network import Unet
from utils import (load_image_pair, cleanup_gpu, CompositeLoss,  set_seed, 
                   load_segmentation, one_hot_encode, get_dataloader_multi, 
                   unpad_np, compute_metrics)
from models import MLPTranslator, CNNTranslator
from vis import visualize_ct_feature_comparison
from engine import train_one_epoch, evaluate

ROOT_DIR = "/home/minsukc/MRI2CT"
CKPT_PATH = os.path.join(ROOT_DIR, "anatomix", "model-weights", "anatomix.pth")
DATA_DIR = os.path.join(ROOT_DIR, "data")

def discover_subjects(data_dir, target_list=None):
    if target_list:
        candidates = target_list
    else:
        candidates = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    valid = []
    # required = ["mr_resampled.nii.gz", "ct_resampled.nii.gz"]
    required = ["ct_resampled.nii.gz"]
    for subj in candidates:
        path = os.path.join(data_dir, subj)
        if all(os.path.exists(os.path.join(path, f)) for f in required):
            valid.append(subj)
    print(f"Found {len(valid)} subjects")
    return valid

def parse_args():
    parser = argparse.ArgumentParser()
    # Basics
    parser.add_argument("-W", "--no_wandb", action="store_true", help="DISABLE W&B")
    parser.add_argument("-E", "--epochs", type=int)
    parser.add_argument("-V", "--val_interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    
    # Config File
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    # Subject Selection
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of subjects to use for validation (default: last 10%)")
    
    # Model
    parser.add_argument("-M", "--model_type", type=str, choices=["mlp", "cnn"])
    parser.add_argument("--no_fourier", action="store_true")
    parser.add_argument("--sigma", type=float, default=10.0)
    
    # Seg
    parser.add_argument("--use_seg", action="store_true")
    parser.add_argument("--seg_name", type=str, default="labels_moved.nii.gz")
    parser.add_argument("--seg_classes", type=int, default=60)

    # CNN
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--patches_per_epoch", type=int, default=100)
    parser.add_argument("--cnn_depth", type=int, default=3)
    parser.add_argument("--cnn_hidden", type=int, default=32)
    parser.add_argument("--final_activation", type=str, default="relu_clamp")

    # Loss
    parser.add_argument("--l1_w", type=float, default=1.0)
    parser.add_argument("--l2_w", type=float, default=1.0)
    parser.add_argument("--ssim_w", type=float, default=0.1)
    
    # Parse initial args to get config path
    temp_args, _ = parser.parse_known_args()
    config_path = os.path.join(ROOT_DIR, temp_args.config)
    
    if os.path.exists(config_path):
        print(f"📄 Loading config from {config_path}")
        with open(config_path) as f:
            parser.set_defaults(**yaml.safe_load(f))
    args = parser.parse_args()
    
    if args.epochs is not None:
        print(f"⚙️ Epochs set via CLI override: {args.epochs}")
    elif args.model_type == "mlp":
        args.epochs = args.epochs_mlp
        print(f"⚙️ Epochs set via Config (MLP): {args.epochs}")
    elif args.model_type == "cnn":
        args.epochs = args.epochs_cnn
        print(f"⚙️ Epochs set via Config (CNN): {args.epochs}")
    else:
        args.epochs = 200
        print(f"⚠️ Epochs defaulted: {args.epochs}")
    
    return args

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    # 1. Discover Subjects
    target_list = args.subjects
    subjects = discover_subjects(DATA_DIR, target_list=target_list)
    
    if not subjects:
        print("❌ No subjects found.")
        return
    print(f"📋 Found {len(subjects)} total subjects.")

    # 2. Train/Val Split
    # By default, use at least 1 validation subject
    random.shuffle(subjects)
    num_val = max(1, int(len(subjects) * args.val_split))
    train_subjects = subjects[:-num_val]
    val_subjects = subjects[-num_val:]
    
    print(f"   Train: {len(train_subjects)} | Val: {len(val_subjects)}")
    print(f"   Validation Subjects: {val_subjects}")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = f"MULTI_{args.model_type}_N{len(train_subjects)}"
        run = wandb.init(project="mri2ct_multi", name=run_name, config=vars(args))
        # print(f"__WANDB_URL__:{run.get_url()}")

    # 3. Initialize Feature Extractor
    feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
    feat_extractor.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    feat_extractor.eval()

    # 4. Load Data (Train & Val)
    train_data_list = []
    # Keep validation metadata separate to run full inference
    val_meta_list = [] 

    # Helper to load and process one subject
    def process_subject(subj_id):
        try:
            mri, ct, pad_vals = load_image_pair(DATA_DIR, subj_id)
            with torch.no_grad():
                inp_mri = torch.from_numpy(mri[None, None]).float().to(device)
                feats = feat_extractor(inp_mri).squeeze(0).cpu().numpy() # [16, D, H, W]
            
            if args.use_seg:
                try:
                    seg = load_segmentation(DATA_DIR, subj_id, args.seg_name, pad_vals)
                    seg_one_hot = one_hot_encode(seg, num_classes=args.seg_classes)
                    feats = np.concatenate([feats, seg_one_hot], axis=0)
                except Exception as e:
                    print(f"⚠️ Seg failed for {subj_id}: {e}. Skipping subject.")
                    return None

            return {'id': subj_id, 'mri': mri, 'ct': ct, 'feats': feats, 'pad_vals': pad_vals}
        except Exception as e:
            print(f"❌ Error loading {subj_id}: {e}")
            return None

    # Load Train
    print("🚀 Pre-loading Training Data...")
    for subj_id in tqdm(train_subjects, desc="Train Loading"):
        data = process_subject(subj_id)
        if data:
            train_data_list.append((data['feats'], data['ct']))
    
    # Load Val
    print("🚀 Pre-loading Validation Data...")
    for subj_id in tqdm(val_subjects, desc="Val Loading"):
        data = process_subject(subj_id)
        if data:
            val_meta_list.append(data)

    cleanup_gpu()
    
    if not train_data_list:
        print("❌ No valid training data.")
        return

    # 5. Create Loader & Model
    loader = get_dataloader_multi(train_data_list, args)
    total_channels = train_data_list[0][0].shape[0]
    print(f"✅ Data Ready. Input Channels: {total_channels}")

    if args.model_type == "mlp":
        model = MLPTranslator(
            in_feat_dim=total_channels, 
            use_fourier=not args.no_fourier, 
            fourier_scale=args.sigma
        ).to(device)
    else:
        print(f"Building CNN: Depth={args.cnn_depth}, Hidden={args.cnn_hidden}, Act={args.final_activation}")
        model = CNNTranslator(
            in_channels=total_channels, 
            hidden_channels=args.cnn_hidden, 
            depth=args.cnn_depth, 
            final_activation=args.final_activation
        ).to(device)

    # 6. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = CompositeLoss(weights={
        "l1": args.l1_w, 
        "l2": args.l2_w, 
        "ssim": args.ssim_w
    }).to(device)
    scaler = torch.amp.GradScaler()

    # --- Validation Function ---
    def run_validation(epoch_idx, current_loss):
        """Runs evaluation on ALL validation subjects."""
        model.eval()
        val_metrics = {'mae': [], 'psnr': [], 'ssim': []}
        
        # We define a helper to handle visualization for just the first val subject
        viz_done = False 

        for v_data in val_meta_list:
            try:
                # Note: evaluate returns UNPADDED metrics but PADDED prediction
                (mae, psnr, ssim), pred_ct_padded = evaluate(
                    model, v_data['feats'], v_data['ct'], device, args.model_type, pad_vals=v_data['pad_vals']
                )
                
                val_metrics['mae'].append(mae)
                val_metrics['psnr'].append(psnr)
                val_metrics['ssim'].append(ssim)
                
                # Visualize the FIRST validation subject
                if not viz_done:
                    print(f"   🔎 Val ({v_data['id']}): MAE={mae:.4f}, PSNR={psnr:.2f}")
                    if use_wandb:
                        visualize_ct_feature_comparison(
                            pred_ct_padded, v_data['ct'], v_data['mri'], feat_extractor, 
                            v_data['id'], ROOT_DIR, epoch=epoch_idx, use_wandb=True
                        )
                    viz_done = True
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"⚠️ OOM during Validation of {v_data['id']}. Skipping.")
                    cleanup_gpu()
                else:
                    raise e
        
        # Log Average Metrics
        avg_mae = np.mean(val_metrics['mae']) if val_metrics['mae'] else 0.0
        avg_psnr = np.mean(val_metrics['psnr']) if val_metrics['psnr'] else 0.0
        avg_ssim = np.mean(val_metrics['ssim']) if val_metrics['ssim'] else 0.0
        
        print(f"Epoch {epoch_idx:03d} | Train Loss: {current_loss:.5f} | Val MAE: {avg_mae:.4f} | Val PSNR: {avg_psnr:.2f}")
        
        if use_wandb:
            wandb.log({
                # "epoch": epoch_idx, 
                "val/mae": avg_mae, 
                "val/psnr": avg_psnr, 
                "val/ssim": avg_ssim
            }, step=epoch_idx)

    # --- Pre-Training Sanity Check ---
    print("🎨 Running initial sanity check (Epoch 0)...")
    run_validation(0, 0.0)

    # 7. Training Loop
    start_time = time.time()
    print(f"🚀 Training for {args.epochs} epochs...")
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True, dynamic_ncols=True)
    
    for epoch in epoch_iter:
        loss, loss_comps = train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, args.model_type)
        epoch_iter.set_postfix({"train_loss": f"{loss:.5f}"})
        
        if use_wandb:
            # log = {"epoch": epoch, "loss/total": loss}
            log = {"loss/total": loss}
            for k, v in loss_comps.items(): log[k.replace("loss_", "loss/")] = v
            wandb.log(log, step=epoch)

        # Periodic Validation
        if ((epoch - 1) % args.val_interval == 0) or (epoch == args.epochs):
            run_validation(epoch, loss)

    # Save Final Model
    save_path = os.path.join(ROOT_DIR, "results", "models", f"multi_{args.model_type}_N{len(train_subjects)}_{datetime.now():%Y%m%d_%H%M}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved model to {save_path}")
    
    if use_wandb:
        wandb.finish()
    print(f"⏱️ Total: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()