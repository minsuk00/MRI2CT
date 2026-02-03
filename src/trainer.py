import os
import gc
import time
import random
import datetime
from glob import glob
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio
from monai.inferers import sliding_window_inference
from anatomix.model.network import Unet

from src.config import Config
from src.utils import set_seed, cleanup_gpu, unpad, compute_metrics
from src.data import DataPreprocessing, get_augmentations, get_subject_paths
from src.loss import CompositeLoss

class Trainer:
    def __init__(self, config_dict):
        # 1. Config Setup
        self.cfg = Config(config_dict)
        set_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        print(f"[DEBUG] 🚀 Initializing Trainer on {self.device}")

        # Default run name
        self.run_name = f"{self.cfg.model_type.upper()}_Train_{datetime.datetime.now():%Y%m%d_%H%M}"

        # 2. Setup Components
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()
        
        # 3. State Tracking
        self.start_epoch = 0
        self.global_step = 0
        self.global_start_time = None 
        self._load_resume()

    def _setup_wandb(self):
        if not self.cfg.wandb: return
        
        # Consistent run name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")                          
        self.run_name = f"{self.cfg.model_type.upper()}_Train{len(self.train_subjects)}_{timestamp}"
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        
        print(f"[DEBUG] 📡 Initializing WandB: {self.run_name}")
        wandb.init(
            project=self.cfg.project_name, 
            name=self.run_name, 
            config=vars(self.cfg),
            notes=self.cfg.wandb_note,
            reinit=True,
            dir=self.cfg.log_dir,
            id=self.cfg.resume_wandb_id, 
            resume="allow",
        )

    def _setup_data(self):
        print(f"[DEBUG] 📂 Searching for data in: {self.cfg.root_dir}")
        
        # Helper to scan a folder
        def scan_split(split_name):
            split_dir = os.path.join(self.cfg.root_dir, split_name)
            if not os.path.exists(split_dir): return []
            return sorted([
                os.path.join(split_name, d) # Store as relative path 'train/1ABA005'
                for d in os.listdir(split_dir) 
                if os.path.isdir(os.path.join(split_dir, d))
            ])

        train_candidates = scan_split("train")
        val_candidates = scan_split("val")
        
        # Logic for 'subjects' (Single Image Optimization)
        if self.cfg.subjects:
            print(f"[DEBUG] 🎯 Filtering specific subjects: {self.cfg.subjects}")
            # Filter candidates that end with the requested ID
            # e.g., if requested '1ABA005', match 'train/1ABA005'
            self.train_subjects = [c for c in train_candidates + val_candidates if os.path.basename(c) in self.cfg.subjects]
            self.val_subjects = self.train_subjects # Validate on the same subject for overfitting
        else:
            # Standard Mode: Use the existing splits
            self.train_subjects = train_candidates
            self.val_subjects = val_candidates
        
        print(f"[DEBUG] 📊 Data Split - Train: {len(self.train_subjects)} | Val: {len(self.val_subjects)}")

        if self.cfg.analyze_shapes:
            shapes = []
            for s in tqdm(self.train_subjects[:30], desc="Analyzing Shapes (Sample)"):
                try:
                    p = get_subject_paths(self.cfg.root_dir, s)
                    sh = nib.load(p['mri']).header.get_data_shape()
                    shapes.append(sh)
                except Exception: pass
            
            if shapes:
                avg_shape = np.mean(np.array(shapes), axis=0).astype(int)
                print(f"📊 Mean Volume Shape: {tuple(int(x) for x in avg_shape)}")
        
        # 3. Helper to create paths
        def _make_subj_list(subjs):
            return [tio.Subject(
                mri=tio.ScalarImage(p['mri']), 
                ct=tio.ScalarImage(p['ct']),
                subj_id=os.path.basename(s) # Extract just ID for logging
            ) for s in subjs for p in [get_subject_paths(self.cfg.root_dir, s)]]

        # 5. Train Loader (Queue)
        train_objs = _make_subj_list(self.train_subjects)
        # use_safety = (self.cfg.model_type.lower() == "cnn" and self.cfg.enable_safety_padding)
        use_safety = self.cfg.enable_safety_padding
        
        preprocess = DataPreprocessing(
            patch_size=self.cfg.patch_size, 
            enable_safety_padding=use_safety, 
            res_mult=self.cfg.res_mult,
            use_weighted_sampler=self.cfg.use_weighted_sampler
        )
        transforms = tio.Compose([preprocess, get_augmentations()]) if self.cfg.augment else preprocess
        
        train_ds = tio.SubjectsDataset(train_objs, transform=transforms) 
        
        if self.cfg.use_weighted_sampler:
            sampler = tio.WeightedSampler(patch_size=self.cfg.patch_size, probability_map='prob_map')
        else:
            sampler = tio.UniformSampler(patch_size=self.cfg.patch_size)
        
        queue = tio.Queue(
            subjects_dataset=train_ds,
            samples_per_volume=self.cfg.patches_per_volume,
            max_length=max(self.cfg.patches_per_volume, self.cfg.data_queue_max_length),
            sampler=sampler,
            num_workers=self.cfg.data_queue_num_workers,
            shuffle_patches=True,
            shuffle_subjects=True
        )
        self.train_loader = tio.SubjectsLoader(queue, batch_size=self.cfg.batch_size, num_workers=0)
        
        # Create infinite iterator
        def _inf_gen(loader):
            while True:
                for batch in loader: yield batch
        self.train_iter = _inf_gen(self.train_loader)

        # 6. Val Loader (Full Volume)
        val_objs = _make_subj_list(self.val_subjects)
        val_preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult)
        val_ds = tio.SubjectsDataset(val_objs, transform=val_preprocess) 
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        total_val = len(self.val_subjects)
        num_viz = min(self.cfg.viz_limit, total_val)
        rng = random.Random(self.cfg.seed) 
        self.val_viz_indices = set(rng.sample(range(total_val), num_viz))
        
        print(f"[DEBUG] 🖼️  Fixed Validation Viz Indices: {self.val_viz_indices}")

    def _setup_models(self):
        # 1. Anatomix (Feature Extractor)
        print(f"[DEBUG] 🏗️ Building Anatomix ({self.cfg.anatomix_weights})...")
        if self.cfg.anatomix_weights == "v1":
            self.cfg.res_mult = 16 
            self.feat_extractor = Unet(3, 1, 16, 4, 16).to(self.device)
            # ckpt = os.path.join(self.cfg.root_dir, "anatomix", "model-weights", "anatomix.pth")
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
        elif self.cfg.anatomix_weights == "v2":
            self.cfg.res_mult = 32
            self.feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(self.device)
            # Optimize inference speed
            self.feat_extractor = torch.compile(self.feat_extractor, mode="default")
            # ckpt = os.path.join(self.cfg.root_dir, "anatomix", "model-weights", "best_val_net_G.pth")
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G.pth"
        else:
            raise ValueError("Invalid anatomix_weights")
            
        if os.path.exists(ckpt):
            self.feat_extractor.load_state_dict(torch.load(ckpt, map_location=self.device), strict=True)
            print(f"[DEBUG] Loaded Anatomix weights from {ckpt}")
        else:
            print(f"[WARNING] ⚠️ Anatomix weights NOT FOUND at {ckpt}")

        if not self.cfg.finetune_feat_extractor:
            for p in self.feat_extractor.parameters(): p.requires_grad = False
            self.feat_extractor.eval()
        
        # 2. Unet Translator
        print(f"[DEBUG] 🏗️ Building Unet Translator (Anatomix v1)...")
        model = Unet(
            dimension=3,
            input_nc=16, 
            output_nc=1,
            num_downs=4,           
            ngf=16, 
            final_act="sigmoid",   
        ).to(self.device)

        if self.cfg.model_compile_mode:
            print(f"[DEBUG] 🚀 Compiling model with mode: {self.cfg.model_compile_mode}")
            self.model = torch.compile(model, mode=self.cfg.model_compile_mode)
        else:
            print(f"[DEBUG] 🐢 specific compile mode not set or None. Skipping compilation.")
            self.model = model

    def _setup_opt(self):
        params = [{'params': self.model.parameters(), 'lr': self.cfg.lr}]
        if self.cfg.finetune_feat_extractor:
            params.append({'params': self.feat_extractor.parameters(), 'lr': self.cfg.lr_feat_extractor})
            
        self.optimizer = torch.optim.Adam(params)
        
        # Auto-Pilot Scheduler
        # patience=10: Waits 10 validation checks (10 * 5 = 50 epochs) before dropping.
        # factor=0.5: When stuck, cuts LR in half.
        if self.cfg.use_scheduler:
            print("[DEBUG] 📉 Initializing Scheduler (ReduceLROnPlateau)")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=10, 
                # patience=5,
                min_lr=1e-6,
            )
        else:
            print("[DEBUG] 🛑 Scheduler DISABLED. Using fixed LR.")
            self.scheduler = None

        self.loss_fn = CompositeLoss(weights={
            "l1": self.cfg.l1_w, "l2": self.cfg.l2_w, 
            "ssim": self.cfg.ssim_w, "perceptual": self.cfg.perceptual_w
        }).to(self.device)
        # self.scaler = torch.cuda.amp.GradScaler()
        self.scaler = torch.amp.GradScaler('cuda')
        
    def _load_resume(self):
        if not self.cfg.resume_wandb_id: return
        
        print(f"[RESUME] 🕵️ Searching for Run ID: {self.cfg.resume_wandb_id}")
        run_folders = glob(os.path.join(self.cfg.log_dir, "wandb", f"run-*-{self.cfg.resume_wandb_id}"))
        if not run_folders:
            print("[RESUME] ❌ Run folder not found.")
            return

        all_ckpts = []
        for f in run_folders:
            ckpts = glob(os.path.join(f, "files", "*.pt"))
            all_ckpts.extend(ckpts)
            
        if not all_ckpts:
            print("[RESUME] ⚠️ No checkpoints found inside run folder.")
            return

        resume_path = max(all_ckpts, key=os.path.getmtime)
        print(f"[RESUME] 📥 Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.cfg.override_lr:
                print(f"[RESUME] 🔧 Forcing new Learning Rate: {self.cfg.lr}")
                for param_group in self.optimizer.param_groups:
                    if len(param_group['params']) == len(list(self.feat_extractor.parameters())):
                         param_group['lr'] = self.cfg.lr_feat_extractor
                    else:
                         param_group['lr'] = self.cfg.lr
        
        # if 'optimizer_state_dict' in checkpoint:
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        else:
            self.global_step = self.start_epoch * self.cfg.steps_per_epoch
            if self.start_epoch > 0:
                print(f"[RESUME] ⚠️ Global step not found. Estimating: {self.global_step}")
    
    def save_checkpoint(self, epoch, is_final=False):
        filename = f"{self.cfg.model_type}_{'FINAL' if is_final else f'epoch{epoch:05d}_{datetime.datetime.now():%Y%m%d_%H%M}'}.pt"
        save_dir = wandb.run.dir if self.cfg.wandb else os.path.join(self.cfg.root_dir, "results", "models")
        os.makedirs(save_dir, exist_ok=True)
        
        save_dict = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.cfg)
        }
        
        if self.cfg.finetune_feat_extractor:
             save_dict['feat_extractor_state_dict'] = self.feat_extractor.state_dict()
             
        path = os.path.join(save_dir, filename)
        torch.save(save_dict, path)
        print(f"[SAVE] 💾 Checkpoint saved: {path}")
        if self.cfg.wandb:
            wandb.log({"info/checkpoint_path": path}, step=self.global_step)
            
    # ==========================================
    # VISUALIZATION
    # ==========================================
    def _log_aug_viz(self, step):
        try:
            # 1. Get Data
            subj_id = self.val_subjects[0] 
            paths = get_subject_paths(self.cfg.root_dir, subj_id)
            
            subj = tio.Subject(mri=tio.ScalarImage(paths['mri']), ct=tio.ScalarImage(paths['ct']))
            prep = DataPreprocessing(patch_size=self.cfg.patch_size, res_mult=self.cfg.res_mult)
            subj = prep(subj)

            # 2. Augment
            aug = get_augmentations()(subj)
            hist_str = " | ".join([t.name for t in aug.history])

            # 3. Slice & Plot
            z = subj['mri'].shape[-1] // 2
            
            # NOTE: If aug changes shape, this line will crash.
            orig_sl = subj['mri'].data[0, ..., z].numpy()
            aug_sl = aug['mri'].data[0, ..., z].numpy()

            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(np.rot90(orig_sl), cmap='gray', vmin=0, vmax=1); ax[0].set_title(f"Original")
            ax[1].imshow(np.rot90(aug_sl), cmap='gray', vmin=0, vmax=1); ax[1].set_title(f"Augmented\n{hist_str}")
            
            # The simple Diff you wanted
            ax[2].imshow(np.rot90(aug_sl - orig_sl), cmap='seismic', vmin=-0.5, vmax=0.5); ax[2].set_title("Diff")
            
            wandb.log({"val/aug_viz": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Aug Viz failed: {e}")

    
    @torch.no_grad()
    def _visualize_full(self, pred, ct, mri, feats_mri, subj_id, shape, step, epoch, idx, offset=0, save_path=None):
        """
        Full 8-column visualization with PCA, Cosine Sim, and Residuals.
        """
        # 1. Extract Features for Comparison
        def extract_np(vol_tensor):
            inp = vol_tensor.to(self.device)
            if inp.ndim == 4: inp = inp.unsqueeze(0) # Handle missing batch dim
            return self.feat_extractor(inp).squeeze(0).cpu().numpy()

        feats_gt = extract_np(ct)
        feats_pred = extract_np(pred)
        # feats_mri is already extracted, just convert to numpy
        feats_mri_np = feats_mri.squeeze(0).cpu().numpy()
        
        # 2. Unpad Volumes
        w, h, d = shape
        gt_ct = unpad(ct, shape, offset).cpu().numpy().squeeze()
        gt_mri = unpad(mri, shape, offset).cpu().numpy().squeeze()
        pred_ct = unpad(pred, shape, offset).cpu().numpy().squeeze()
        
        # 3. Unpad Features (C, W, H, D)
        feats_gt = feats_gt[..., offset:offset+w, offset:offset+h, offset:offset+d]
        feats_pred = feats_pred[..., offset:offset+w, offset:offset+h, offset:offset+d]
        feats_mri_np = feats_mri_np[..., offset:offset+w, offset:offset+h, offset:offset+d]

        C, H, W, D_dim = feats_gt.shape
        
        # 4. Define Items
        items = [
            (gt_mri, "GT MRI", "gray", (0,1)),
            (gt_ct, "GT CT", "gray", (0,1)),
            (pred_ct, "Pred CT", "gray", (0,1)),
            (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5)),
        ]

        # 5. PCA Logic
        if self.cfg.viz_pca:
            def sample_vox(f, max_v=200_000):
                X = f.reshape(C, -1).T
                if X.shape[0] > max_v: X = X[np.random.choice(X.shape[0], max_v, replace=False)]
                return X
            
            X_all = np.concatenate([sample_vox(feats_mri_np), sample_vox(feats_gt), sample_vox(feats_pred)], axis=0)
            pca = PCA(n_components=3, svd_solver="randomized").fit(X_all)
            
            def proj(f):
                Y = pca.transform(f.reshape(C, -1).T)
                Y = (Y - Y.min(0, keepdims=True)) / (Y.max(0, keepdims=True) - Y.min(0, keepdims=True) + 1e-8)
                return Y.reshape(H, W, D_dim, 3)

            items.extend([
                (proj(feats_mri_np), "PCA (MRI)", None, None),
                (proj(feats_gt), "PCA (GT CT)", None, None),
                (proj(feats_pred), "PCA (Pred)", None, None),
            ])

        # 6. Cosine Similarity
        gt_t = torch.from_numpy(feats_gt).unsqueeze(0)
        pred_t = torch.from_numpy(feats_pred).unsqueeze(0)
        cos_sim = F.cosine_similarity(gt_t, pred_t, dim=1).squeeze(0).numpy()
        cos_sim_n = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        items.append((cos_sim_n, "Cosine Sim", "plasma", (0,1)))

        # 7. Plotting
        num_cols = len(items)
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)
        
        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(4 * num_cols, 3.5 * len(slice_indices)))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        
        # Handle single row edge case
        if len(slice_indices) == 1: axes = axes.reshape(1, -1)

        for i, z_slice in enumerate(slice_indices):
            for j, (data, title, cmap, clim) in enumerate(items):
                ax = axes[i, j]
                if data.ndim == 3: # (H, W, D)
                    im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
                    if title == "Residual": res_im = im
                    if title == "Cosine Sim": cos_im = im
                else: # (H, W, D, 3) RGB
                    ax.imshow(data[:, :, z_slice, :])
                
                if i == 0: ax.set_title(title)
                ax.axis("off")

        # Colorbars
        if 'res_im' in locals():
            cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
            cbar.set_label("Residual Error")
        
        cbar2 = fig.colorbar(cos_im, ax=axes[:, num_cols-1], fraction=0.04, pad=0.01)
        cbar2.set_label("Cosine Similarity")

        title_str = f"Subject: {subj_id} | Epoch {epoch} | Step {step}"
        caption = f"Subject: {subj_id}"
        if save_path: caption += f"\nSaved to: {save_path}"
        fig.suptitle(title_str, fontsize=16, y=0.99)
        
        if self.cfg.wandb:
            wandb.log({f"viz/{'train' if idx==-1 else ('val_'+ str(idx))}": wandb.Image(fig, caption=caption)}, step=step)
        plt.close(fig)

    @torch.no_grad()
    def _visualize_lite(self, pred, ct, mri, subj_id, shape, step, epoch, idx, offset=0, save_path=None):
        """
        Lightweight visualization: MRI, GT, Pred, Residual. 
        """
        # 1. Unpad Volumes
        w, h, d = shape
        gt_ct = unpad(ct, shape, offset).cpu().numpy().squeeze()
        gt_mri = unpad(mri, shape, offset).cpu().numpy().squeeze()
        pred_ct = unpad(pred, shape, offset).cpu().numpy().squeeze()
        
        # 2. Define Items (Standard 4-column view)
        items = [
            (gt_mri, "GT MRI", "gray", (0,1)),
            (gt_ct, "GT CT", "gray", (0,1)),
            (pred_ct, "Pred CT", "gray", (0,1)),
            (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5)),
        ]

        # 3. Plotting
        D_dim = gt_ct.shape[-1]
        num_cols = len(items)
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)
        
        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(3 * num_cols, 3.5 * len(slice_indices)))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        
        if len(slice_indices) == 1: axes = axes.reshape(1, -1)

        for i, z_slice in enumerate(slice_indices):
            for j, (data, title, cmap, clim) in enumerate(items):
                ax = axes[i, j]
                im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
                
                if title == "Residual": res_im = im
                if i == 0: ax.set_title(title)
                ax.axis("off")

        if 'res_im' in locals():
            cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
            cbar.set_label("Residual Error")
        
        title_str = f"Subject: {subj_id} | Epoch {epoch} | Step {step}"
        caption = f"Subject: {subj_id}"
        if save_path: caption += f"\nSaved to: {save_path}"
        fig.suptitle(title_str, fontsize=16, y=0.99)
        
        if self.cfg.wandb:
            wandb.log({f"viz/{('val_'+ str(idx))}": wandb.Image(fig, caption=caption)}, step=step)
        plt.close(fig)
        
    # Added method to visualize training patches
    @torch.no_grad()
    def _log_training_patch(self, mri, ct, pred, step, batch_idx):
        """
        Visualizes MRI, Prediction, and CT (GT) for the first patch in the batch.
        """
        # 1. Prepare Data (Batch 0, Channel 0)
        img_in = mri[0, 0].detach().cpu().float().numpy()
        img_gt = ct[0, 0].detach().cpu().float().numpy()
        img_pred = pred[0, 0].detach().cpu().float().numpy()
        
        # Center indices
        cx, cy, cz = np.array(img_in.shape) // 2

        # 3 Rows (MRI, Pred, CT), 3 Cols (Axial, Sagittal, Coronal)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        # Helper to plot a row
        def plot_row(row_idx, vol, title_prefix, vmin=None, vmax=None):
            # Axial
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"{title_prefix} Ax")
            # Sagittal
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"{title_prefix} Sag")
            # Coronal
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 2].set_title(f"{title_prefix} Cor")
            
            # Add stats text to the left of the row
            axes[row_idx, 0].text(-5, 10, f"Min: {vol.min():.2f}\nMax: {vol.max():.2f}", 
                                  fontsize=8, color='white', backgroundcolor='black')

        # Row 1: MRI (Auto-scaled intensity)
        plot_row(0, img_in, "MRI")

        # Row 2: Prediction (Fixed 0-1 range to match CT)
        plot_row(1, img_pred, "Pred", vmin=0, vmax=1)

        # Row 3: CT (Fixed 0-1 range)
        plot_row(2, img_gt, "GT CT", vmin=0, vmax=1)

        # Cleanup
        for ax in axes.flatten():
            ax.axis('off')
        
        plt.tight_layout()
        
        wandb.log({f"train/patch_viz": wandb.Image(fig)}, step=step)
        plt.close(fig)
        
    # ==========================================
    # CORE LOGIC
    # ==========================================
    def train_epoch(self, epoch):
        self.model.train()
        if self.cfg.finetune_feat_extractor: self.feat_extractor.train()
        else: self.feat_extractor.eval()
        
        total_loss = 0.0
        total_grad = 0.0
        comp_accum = {}
        
        pbar = tqdm(range(self.cfg.steps_per_epoch), desc=f"Train Ep {epoch}", leave=False, dynamic_ncols=True)
        
        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            step_t_data = 0.0
            step_t_fwd = 0.0
            step_t_bwd = 0.0
            t_step_start = 0.0
            
            for _ in range(self.cfg.accum_steps):
                t0 = time.time() 
                batch = next(self.train_iter)
                t1 = time.time()
                step_t_data += (t1 - t0)
                
                mri = batch['mri'][tio.DATA].to(self.device, non_blocking=True)
                ct = batch['ct'][tio.DATA].to(self.device, non_blocking=True)

                torch.cuda.synchronize()
                t2 = time.time()

                # Use 'dtype=torch.bfloat16' for Ampere+ GPUs (3090, 4090, A100, A6000)
                # Use 'dtype=torch.float16' for older GPUs (2080, V100, Titan)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.cfg.finetune_feat_extractor:
                        features = self.feat_extractor(mri)
                    else:
                        with torch.no_grad(): features = self.feat_extractor(mri)
                    
                    pred = self.model(features)
                    
                    if self.cfg.wandb and step_idx == 0:
                        self._log_training_patch(mri, ct, pred, self.global_step, step_idx)
                    
                    fe_ref = self.feat_extractor if self.cfg.perceptual_w > 0 else None
                    loss, comps = self.loss_fn(pred, ct, feat_extractor=fe_ref)
                    
                    for k, v in comps.items():
                        comp_accum[k] = comp_accum.get(k, 0.0) + (v / self.cfg.accum_steps)
                    
                    loss = loss / self.cfg.accum_steps

                    torch.cuda.synchronize()
                    t3 = time.time()
                    step_t_fwd += (t3 - t2)
                    
                    self.scaler.scale(loss).backward()
                    step_loss += loss.item()

                    torch.cuda.synchronize()
                    t4 = time.time()
                    step_t_bwd += (t4 - t3)

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + (list(self.feat_extractor.parameters()) if self.cfg.finetune_feat_extractor else []),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += step_loss
            total_grad += grad_norm.item()
            
            self.global_step += 1

            t_step_end = time.time()
            step_t_total = t_step_end - t_step_start
            avg_batch_data = step_t_data / self.cfg.accum_steps
            avg_batch_fwd = step_t_fwd / self.cfg.accum_steps
            avg_batch_bwd = step_t_bwd / self.cfg.accum_steps
            
            # pbar.set_postfix({"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"})

            pbar.set_postfix({
                "loss": f"{step_loss:.4f}", 
                "gn": f"{grad_norm.item():.2f}",
                "dt": f"{avg_batch_data:.3f}", 
                "fwd": f"{avg_batch_fwd:.3f}",
                "bwd": f"{avg_batch_bwd:.3f}"
            })
            
            if self.cfg.wandb:
                log_dict = {
                    "train/loss_step": step_loss,
                    "train/grad_norm": grad_norm.item(),
                    "info/time_data": avg_batch_data,
                    "info/time_forward": avg_batch_fwd,
                    "info/time_backward": avg_batch_bwd,
                    "info/time_step_total": step_t_total,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                }
                # Log components step-wise
                for k, v in comps.items():
                    log_dict[k.replace("loss_", "train_loss_step/")] = v
                
                wandb.log(log_dict, step=self.global_step)
            
        return total_loss / self.cfg.steps_per_epoch, \
               {k: v / self.cfg.steps_per_epoch for k, v in comp_accum.items()}, \
               total_grad / self.cfg.steps_per_epoch

    @torch.no_grad()
    def validate(self, epoch):
        gc.collect()
        torch.cuda.empty_cache()

        self.model.eval()
        val_metrics = defaultdict(list)
        
        # 1. Validation Loop
        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mri = batch['mri'][tio.DATA].to(self.device)
            ct = batch['ct'][tio.DATA].to(self.device)
            orig_shape = batch['original_shape'][0].tolist()
            subj_id = batch['subj_id'][0]
            pad_offset = batch['pad_offset'][0].item() if 'pad_offset' in batch else 0
            
            # Sliding Window (Lite) vs Full Volume (Standard)
            feats = None
            if self.cfg.val_sliding_window:
                def combined_forward(x):
                    return self.model(self.feat_extractor(x))

                pred = sliding_window_inference(
                    inputs=mri, 
                    roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size), 
                    sw_batch_size=self.cfg.val_sw_batch_size, 
                    predictor=combined_forward,
                    overlap=self.cfg.val_sw_overlap,
                    mode="gaussian",
                    device=self.device
                )
            else:
                feats = self.feat_extractor(mri)
                pred = self.model(feats)
            
            # Metrics
            pred_unpad = unpad(pred, orig_shape, pad_offset)
            ct_unpad = unpad(ct, orig_shape, pad_offset)
            met = compute_metrics(pred_unpad, ct_unpad)
            
            # Loss (Composite)
            l_val, _ = self.loss_fn(pred, ct, feat_extractor=self.feat_extractor, use_sliding_window = self.cfg.val_sliding_window)
            met['loss'] = l_val.item()
            
            for k, v in met.items():
                val_metrics[k].append(v)
            
            # Viz & Save
            if i in self.val_viz_indices:
                save_path = None
                if self.cfg.save_val_volumes:
                    save_dir = os.path.join(self.cfg.prediction_dir, self.run_name, f"epoch_{epoch}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Denormalize [0, 1] -> [-1024, 1024]
                    pred_np = pred_unpad.float().cpu().numpy().squeeze()
                    pred_hu = (pred_np * 2048.0) - 1024.0
                    affine = batch['ct']['affine'][0].cpu().numpy()
                    
                    nii = nib.Nifti1Image(pred_hu, affine)
                    save_path = os.path.join(save_dir, f"pred_{subj_id}.nii.gz")
                    nib.save(nii, save_path)

                if self.cfg.wandb:
                    if self.cfg.val_sliding_window:
                        self._visualize_lite(pred, ct, mri, subj_id, orig_shape, self.global_step, epoch, idx=i, offset=pad_offset, save_path=save_path)
                    else:
                        self._visualize_full(pred, ct, mri, feats, subj_id, orig_shape, self.global_step, epoch, idx=i, offset=pad_offset, save_path=save_path)

            del mri, ct, pred, pred_unpad, ct_unpad
            if 'feats' in locals(): del feats
            torch.cuda.empty_cache()
            
        # 2. Augmentation Viz
        if self.cfg.wandb and self.cfg.augment:
             self._log_aug_viz(self.global_step)

        # 3. Log
        avg_met = {k: np.mean(v) for k, v in val_metrics.items()}
        if self.cfg.wandb:
            wandb.log({f"val/{k}": v for k, v in avg_met.items()}, step=self.global_step)

        gc.collect()
        torch.cuda.empty_cache()
        
        return avg_met

    def train(self):
        print(f"[DEBUG] 🏁 Starting Loop: Ep {self.start_epoch} -> {self.cfg.total_epochs}")
        self.global_start_time = time.time()
        
        if self.cfg.sanity_check and not self.cfg.resume_wandb_id:
            print("[DEBUG] running sanity check...")
            avg_met = self.validate(0)
            tqdm.write(
                    f"Ep -1 | Train: 0.0000 | Val: {avg_met.get('loss',0):.4f} | "
                    f"SSIM: {avg_met.get('ssim',0):.4f} | PSNR: {avg_met.get('psnr',0):.2f}"
                )

        global_pbar = tqdm(
            range(self.start_epoch, self.cfg.total_epochs),
            desc="🚀 Total Progress",
            initial=self.start_epoch,
            total=self.cfg.total_epochs,
            dynamic_ncols=True,
            unit="ep"
        )
            
        for epoch in global_pbar:
            ep_start = time.time()
            
            loss, comps, gn = self.train_epoch(epoch)
            
            if epoch % self.cfg.val_interval == 0 or (epoch+1) == self.cfg.total_epochs:
                avg_met = self.validate(epoch)
                val_loss = avg_met.get('loss', 0)

                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                
                tqdm.write(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {val_loss:.4f} | "
                    f"SSIM: {avg_met.get('ssim',0):.4f} | PSNR: {avg_met.get('psnr',0):.2f} | "
                    f"Bone: {avg_met.get('bone_dice',0):.4f}"
                )

            ep_duration = time.time() - ep_start
            cumulative_time = time.time() - self.global_start_time

            if self.cfg.wandb:
                current_lr = self.optimizer.param_groups[0]['lr']
                log = {
                    "train_loss/total": loss, 
                    "info/grad_norm": gn, 
                    "info/epoch_duration": ep_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/lr": current_lr 
                }
                for k, v in comps.items(): log[k.replace("loss_", "train_loss/")] = v
                wandb.log(log, step=self.global_step)
                
            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)
                
        self.save_checkpoint(self.cfg.total_epochs, is_final=True)
        if self.cfg.wandb: wandb.finish()
        print(f"✅ Training Complete. Total Time: {time.time() - self.global_start_time:.1f}s")
