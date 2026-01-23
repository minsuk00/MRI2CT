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

# Local imports
from anatomix.model.network import Unet
from utils import load_image_pair, cleanup_gpu, CompositeLoss, set_seed, load_segmentation, one_hot_encode, get_dataloader_single
from models import MLPTranslator, CNNTranslator
from vis import visualize_ct_feature_comparison
from engine import train_one_epoch, evaluate

ROOT_DIR = "/home/minsukc/MRI2CT"
CKPT_PATH = os.path.join(ROOT_DIR, "anatomix", "model-weights", "anatomix.pth")
DATA_DIR = os.path.join(ROOT_DIR, "data")

def parse_args():
    parser = argparse.ArgumentParser()

    # Basics
    parser.add_argument("-W", "--no_wandb", action="store_true", help="DISABLE W&B (Default: Enabled)")
    parser.add_argument("-E", "--epochs", type=int, help="Number of epochs")
    parser.add_argument("-V", "--val_interval", type=int, help="Validation interval")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("-S", "--subject", type=str, help="Subject ID")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Model & Config
    parser.add_argument("-M", "--model_type", type=str, choices=["mlp", "cnn"], help="Model type")
    parser.add_argument("--no_fourier", action="store_true", help="Disable Fourier positional encodings for MLP")
    parser.add_argument("--sigma", type=float, help="Scale (std dev) for Random Fourier Features")
    parser.add_argument("--use_seg", action="store_true", help="Use MRI segmentation mask as input")
    parser.add_argument("--seg_name", type=str, help="Segmentation filename")
    parser.add_argument("--dropout", type=float)

    # CNN Config
    parser.add_argument("--patch_size", type=int, help="Cube size for CNN patch training")
    parser.add_argument("--samples_per_volume", type=int, help="Number of samples per volume")
    parser.add_argument("--cnn_depth", type=int, help="Number of layers in CNN")
    parser.add_argument("--cnn_hidden", type=int, help="Hidden channels for CNN")
    parser.add_argument("--final_activation", type=str, default="relu_clamp", 
                        choices=["sigmoid", "relu_clamp", "none"],
                        help="Final layer activation function")
    # Loss Weights
    parser.add_argument("--l1_w", type=float, help="L1 Loss Weight")
    parser.add_argument("--l2_w", type=float, help="MSE Loss Weight")
    parser.add_argument("--ssim_w", type=float, help="SSIM Loss Weight")
    
    # Load Config File
    config_path = os.path.join(ROOT_DIR, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        parser.set_defaults(**config) # Set defaults in parser based on config
    else:
        raise FileNotFoundError("Config file not found.")
    # Parse Args (CLI overrides Config)
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
    
    use_wandb = not args.no_wandb
    if use_wandb:
        run = wandb.init(project="mri2ct", name=f"{args.model_type}_{args.subject}", config=vars(args))
        print(f"__WANDB_URL__:{run.get_url()}")
        print(f"__WANDB_ID__:{run.id}")
    
    # --- 1. Load Data (Raw) ---
    mri, ct, pad_vals = load_image_pair(DATA_DIR, args.subject)
    
    # --- 2. Extract Features (Anatomix) ---
    feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
    feat_extractor.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    feat_extractor.eval()

    print("Extracting Anatomix features...")
    with torch.no_grad():
        inp_mri = torch.from_numpy(mri[None, None]).float().to(device)
        feats_mri = feat_extractor(inp_mri).squeeze(0).cpu().numpy()
    
    cleanup_gpu()
    
    # --- 3. Setup Datasets & Model ---
    if args.use_seg:
        print(f"Loading Segmentation: {args.seg_name}...")
        seg_np = load_segmentation(DATA_DIR, args.subject, seg_filename=args.seg_name, pad_vals=pad_vals)
        seg_one_hot = one_hot_encode(seg_np)
        print(f"Segmentation encoded: {seg_one_hot.shape} (Channels={seg_one_hot.shape[0]})")
        feats_mri = np.concatenate([feats_mri, seg_one_hot], axis=0)
        print(f"Combined Feature Shape: {feats_mri.shape}")
    cleanup_gpu()

    loader = get_dataloader_single(feats_mri, ct, args)
    total_channels = feats_mri.shape[0]

    if args.model_type == "mlp":
        model = MLPTranslator(
            in_feat_dim=total_channels, 
            use_fourier=not args.no_fourier, 
            fourier_scale=args.sigma,
            dropout=args.dropout,
        ).to(device)
    elif args.model_type == "cnn":
        print(f"Building CNN: Depth={args.cnn_depth}, Hidden={args.cnn_hidden}, Act={args.final_activation}")
        model = CNNTranslator(
            in_channels=total_channels,
            hidden_channels=args.cnn_hidden,
            depth=args.cnn_depth,
            final_activation=args.final_activation,
            dropout=args.dropout,
        ).to(device)
    
    # --- 4. Optimizer & Loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = CompositeLoss(weights={
        "l1": args.l1_w, 
        "l2": args.l2_w, 
        "ssim": args.ssim_w
    }).to(device)
    scaler = torch.amp.GradScaler()

    def run_validation(epoch_idx, current_loss):
        try:
            (mae, psnr, ssim), pred_ct = evaluate(model, feats_mri, ct, device, args.model_type, pad_vals=pad_vals)
            
            print(f"Epoch {epoch_idx:03d} | Loss: {current_loss:.5f} | MAE: {mae:.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")
            
            if use_wandb: 
                # wandb.log({"epoch": epoch_idx, "val/mae": mae, "val/psnr": psnr, "val/ssim": ssim}, step=epoch_idx)
                wandb.log({"val/mae": mae, "val/psnr": psnr, "val/ssim": ssim}, step=epoch_idx)
            
            visualize_ct_feature_comparison(
                pred_ct, ct, mri, feat_extractor, args.subject, 
                ROOT_DIR, epoch=epoch_idx, use_wandb=use_wandb
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️ OOM during Validation at epoch {epoch_idx}. Skipping visualization.")
                cleanup_gpu()
            else:
                raise e
    # --- Pre-Training Evaluation (Epoch 0) ---
    print("🎨 Running initial visualization (Epoch 0)...")
    run_validation(0, 0.0)
    
    # --- 5. Training Loop ---
    start_time = time.time()
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True, dynamic_ncols=True)
    for epoch in epoch_iter:
        loss, loss_comps = train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, args.model_type)
        epoch_iter.set_postfix({"train_loss": f"{loss:.5f}"})
        
        if use_wandb: 
            # log_data = {"epoch": epoch, "loss/total": loss}
            log_data = {"loss/total": loss}
            for k, v in loss_comps.items():
                clean_k = k.replace("loss_", "loss/")
                log_data[clean_k] = v
            wandb.log(log_data, step=epoch)

        if ((epoch - 1) % args.val_interval == 0) or (epoch == args.epochs):
            run_validation(epoch,loss)

    # --- 6. Finish ---
    save_path = os.path.join(ROOT_DIR, "results", "models", f"model_{datetime.now():%Y%m%d_%H%M}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved to {save_path}")
    
    if use_wandb:
        wandb.finish()
    print(f"⏱️ Total: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()