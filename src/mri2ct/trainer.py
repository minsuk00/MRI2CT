import os
import sys
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
# from sklearn.decomposition import PCA
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio
from monai.inferers import sliding_window_inference
from anatomix.model.network import Unet
from anatomix.segmentation.segmentation_utils import load_model_v1_1

# Add CADS to path (Optional now, but kept for legacy)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "CADS")))

from mri2ct.config import Config
from mri2ct.utils import set_seed, cleanup_gpu, unpad, compute_metrics, get_ram_info, count_parameters
from mri2ct.data import DataPreprocessing, get_augmentations, get_subject_paths, get_region_key
from mri2ct.loss import CompositeLoss

class Trainer:
    def __init__(self, config_dict):
        # 1. Config Setup
        self.cfg = Config(config_dict)
        set_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        print(f"[DEBUG] 🚀 Initializing Trainer on {self.device}")

        # Default run name
        self.run_name = f"{datetime.datetime.now():%Y%m%d_%H%M}_{self.cfg.anatomix_weights.upper()}_Train"

        # 1.5. Stage Data to Local NVMe
        self._stage_data_local()

        # 2. Setup Components
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()
        
        # 3. State Tracking
        self.start_epoch = 0
        self.global_step = 0
        self.samples_seen = 0
        self.global_start_time = None 
        self._load_resume()

    def _stage_data_local(self):
        """Copies dataset to local NVMe RAID for blazing fast I/O."""
        user_id = os.environ.get("USER", "default")
        local_root = os.path.join("/tmp_data", f"mri2ct_dataset_{user_id}")
        
        if not os.path.exists("/tmp_data") or not os.access("/tmp_data", os.W_OK):
            print("[Trainer] ⚠️ Local storage not available. Staying on GPFS.")
            return

        print(f"[Trainer] 🚚 Staging data to local NVMe: {local_root}")
        os.makedirs(local_root, exist_ok=True)
        
        # We only need train/val subfolders
        # rsync is efficient: only copies if different
        for split in ["train", "val"]:
            src = os.path.join(self.cfg.root_dir, split)
            dst = os.path.join(local_root, split)
            if os.path.exists(src):
                print(f"  - Syncing {split}...")
                os.system(f"rsync -am --include='*/' --include='ct.nii.gz' --include='registration_output/' --include='moved_mr.nii.gz' --include='cads_ct_seg.nii.gz' --exclude='*' {src}/ {dst}/")

        self.cfg.root_dir = local_root
        print(f"[Trainer] ✅ Data staged. New root: {self.cfg.root_dir}")

    def _setup_wandb(self):
        if not self.cfg.wandb: return
        
        # Consistent run name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")                          
        self.run_name = f"{timestamp}_{self.cfg.anatomix_weights.upper()}_Train{len(self.train_subjects)}"
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        
        print(f"[DEBUG] 📡 Initializing WandB: {self.run_name}")
        wandb_id = None if self.cfg.diverge_wandb_branch else self.cfg.resume_wandb_id
        
        wandb.init(
            project=self.cfg.project_name, 
            name=self.run_name, 
            config=vars(self.cfg),
            notes=self.cfg.wandb_note,
            reinit=True,
            dir=self.cfg.log_dir,
            id=wandb_id, 
            resume="allow" if not self.cfg.diverge_wandb_branch else None,
        )

    def _setup_data(self):
        print(f"[DEBUG] 📂 Searching for data in: {self.cfg.root_dir}")
        
        # Helper to scan a folder
        def scan_split(split_name):
            split_dir = os.path.join(self.cfg.root_dir, split_name)
            if not os.path.exists(split_dir): return []
            
            valid_subjs = []
            for d in sorted(os.listdir(split_dir)):
                subj_path = os.path.join(split_dir, d)
                if not os.path.isdir(subj_path): continue
                
                # Check for required files (must have both CT and Registered MRI)
                has_ct = os.path.exists(os.path.join(subj_path, "ct.nii.gz"))
                has_mr = os.path.exists(os.path.join(subj_path, "registration_output", "moved_mr.nii.gz"))
                
                if has_ct and has_mr:
                    valid_subjs.append(os.path.join(split_name, d))
            
            return valid_subjs

        train_candidates = scan_split("train")
        val_candidates = scan_split("val")
        
        # Logic for 'subjects' (Single Image Optimization)
        if self.cfg.subjects:
            print(f"[DEBUG] Filtering specific subjects: {self.cfg.subjects}")
            # Filter candidates that end with the requested ID
            # e.g., if requested '1ABA005', match 'train/1ABA005'
            self.train_subjects = [c for c in train_candidates + val_candidates if os.path.basename(c) in self.cfg.subjects]
            self.val_subjects = self.train_subjects # Validate on the same subject for overfitting
        else:
            # Standard Mode: Use the existing splits
            self.train_subjects = train_candidates
            self.val_subjects = val_candidates
        
        print(f"[DEBUG] Data Split - Train: {len(self.train_subjects)} | Val: {len(self.val_subjects)}")

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
        def _make_subj_list(subjs, load_seg=False):
            subj_list = []
            for s in subjs:
                paths = get_subject_paths(self.cfg.root_dir, s)
                kwargs = {
                    'mri': tio.ScalarImage(paths['mri']), 
                    'ct': tio.ScalarImage(paths['ct']),
                    'subj_id': os.path.basename(s)
                }
                # Conditionally load segmentation
                if load_seg:
                    seg_path = os.path.join(self.cfg.root_dir, s, "cads_ct_seg.nii.gz")
                    if os.path.exists(seg_path):
                        kwargs['seg'] = tio.LabelMap(seg_path)
                    else:
                        raise FileNotFoundError(f"Segmentation missing for {s} but dice_w > 0.")
                        
                subj_list.append(tio.Subject(**kwargs))
            return subj_list

        # 5. Train Loader (Queue)
        # Only load seg if Dice weight > 0
        load_seg = getattr(self.cfg, 'dice_w', 0) > 0
        train_objs = _make_subj_list(self.train_subjects, load_seg=load_seg)
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
        # Load seg for validation if we are using Dice loss (to measure semantic consistency)
        val_objs = _make_subj_list(self.val_subjects, load_seg=load_seg)
        val_preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult)
        val_ds = tio.SubjectsDataset(val_objs, transform=val_preprocess) 
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

        # Stratified Validation Sampling (2 per region)
        total_val = len(self.val_subjects)
        rng = random.Random(self.cfg.seed)
        
        region_to_indices = defaultdict(list)
        for idx, subj_path in enumerate(self.val_subjects):
            subj_id = os.path.basename(subj_path)
            region = get_region_key(subj_id)
            region_to_indices[region].append(idx)
        
        viz_indices = []
        for region, indices in region_to_indices.items():
            # Pick up to 2 subjects per region
            num_to_pick = min(len(indices), 2)
            viz_indices.extend(rng.sample(indices, num_to_pick))
        
        self.val_viz_indices = set(viz_indices)
        
        # print(f"[DEBUG] 🖼️  Stratified Val Viz Indices: {self.val_viz_indices}")
        for region, indices in region_to_indices.items():
            picked = [i for i in viz_indices if i in indices]
            # print(f"   - {region:10}: picked {len(picked)}/{len(indices)} (indices: {picked})")

    def _setup_models(self):
        # 1. Anatomix (Feature Extractor)
        print(f"[DEBUG] 🏗️ Building Anatomix ({self.cfg.anatomix_weights})...")
        if self.cfg.anatomix_weights == "v1":
            self.cfg.res_mult = 16 
            self.feat_extractor = Unet(3, 1, 16, 4, 16).to(self.device)
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
        elif self.cfg.anatomix_weights == "v2":
            self.cfg.res_mult = 32
            self.feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(self.device)
            # Optimize inference speed
            self.feat_extractor = torch.compile(self.feat_extractor, mode="default")
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
            
        tot, train = count_parameters(self.feat_extractor)
        print(f"[Model] Anatomix Feat Extractor Params: Total={tot:,} | Trainable={train:,}")
        
        # 2. Unet Translator (Generator)
        print(f"[DEBUG] 🏗️ Building Unet Translator (Anatomix v1)...")
        model = Unet(
            dimension=3,
            input_nc=16, 
            output_nc=1,
            num_downs=4,           
            ngf=16, 
            final_act="sigmoid",   
        ).to(self.device)

        # Non-recursive compilation logic
        # If 'full' mode, we compile the step, so we leave the models raw.
        # If 'model' mode, we compile the models individually.
        should_compile_models = self.cfg.compile_mode == "model"
        
        if should_compile_models:
            print(f"[DEBUG] 🚀 Compiling Generator (mode=default)...")
            self.model = torch.compile(model, mode="default")
        else:
            self.model = model
            
        tot, train = count_parameters(self.model)
        print(f"[Model] Unet Translator Params: Total={tot:,} | Trainable={train:,}")
            
        # 3. Teacher Model (Baby U-Net) for Dice Loss
        self.teacher_model = None
        if getattr(self.cfg, 'dice_w', 0) > 0:
            print("[DEBUG] 👨‍🏫 Initializing Baby U-Net Teacher for Dice Loss...")
            try:
                # Load Baby U-Net (12 classes: 11 organs + Brain)
                self.teacher_model = load_model_v1_1(
                    pretrained_ckpt=self.cfg.teacher_weights_path, 
                    n_classes=self.cfg.n_classes - 1, 
                    device=self.device,
                    compile_model=False 
                )
                
                # Freeze Teacher
                self.teacher_model.to(device=self.device, dtype=torch.bfloat16)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                
                if should_compile_models:
                    print(f"[DEBUG] 🚀 Compiling Teacher with mode: default")
                    self.teacher_model = torch.compile(self.teacher_model, mode="default")
                
                tot, train = count_parameters(self.teacher_model)
                print(f"[Model] Teacher Params: Total={tot:,} | Trainable={train:,} | Dtype=BFloat16")
                print("[DEBUG] ✅ Teacher initialized.")
            except Exception as e:
                print(f"[ERROR] ❌ Failed to init Teacher Model: {e}")
                if self.cfg.dice_w > 0: raise e

        # 4. Step Compilation
        if self.cfg.compile_mode == "full":
            print("[SegTrainer] 🚀 Compiling Training Step (mode=default)...")
            self.train_step = torch.compile(self._train_step, mode="default")
        else:
            self.train_step = self._train_step

    def _train_step(self, mri, ct, seg=None):
        # Note: zero_grad is handled in the outer loop for accumulation
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.cfg.finetune_feat_extractor:
                features = self.feat_extractor(mri)
            else:
                # Feat extractor is usually frozen, so we can wrap in no_grad even inside compile
                with torch.no_grad(): features = self.feat_extractor(mri)
            
            pred = self.model(features)
            
            # Dice Loss Calculation
            pred_probs = None
            if self.teacher_model is not None and seg is not None:
                pred_probs = self.teacher_model(pred)
            
            loss, comps = self.loss_fn(
                pred, ct, 
                pred_probs=pred_probs, 
                target_mask=seg
            )
            
            loss = loss / self.cfg.accum_steps
        
        self.scaler.scale(loss).backward()
        return pred, loss, comps, pred_probs

    def _setup_opt(self):
        params = [{'params': self.model.parameters(), 'lr': self.cfg.lr}]
        if self.cfg.finetune_feat_extractor:
            params.append({'params': self.feat_extractor.parameters(), 'lr': self.cfg.lr_feat_extractor})
            
        self.optimizer = torch.optim.Adam(params)
        
        # 3. Scheduler Setup
        if self.cfg.scheduler_type == "cosine":
            print(f"[DEBUG] 📉 Initializing Scheduler (CosineAnnealingLR) T_max={self.cfg.scheduler_t_max}, min_lr={self.cfg.scheduler_min_lr}")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.scheduler_t_max,
                eta_min=self.cfg.scheduler_min_lr
            )
        else:
            print("[DEBUG] 🛑 Scheduler DISABLED. Using fixed LR.")
            self.scheduler = None

        self.loss_fn = CompositeLoss(weights={
            "l1": self.cfg.l1_w, "l2": self.cfg.l2_w, 
            "ssim": self.cfg.ssim_w, 
            "dice_w": getattr(self.cfg, "dice_w", 0.0)
        }).to(self.device)
        # self.scaler = torch.cuda.amp.GradScaler()
        self.scaler = torch.amp.GradScaler('cuda')
        
    def _load_resume(self):
        if not self.cfg.resume_wandb_id: return
        
        print(f"[RESUME] 🕵️ Searching for Run ID: {self.cfg.resume_wandb_id}")
        run_folders = glob(os.path.join(self.cfg.log_dir, "wandb", f"run-*-{self.cfg.resume_wandb_id}"))
        if not run_folders:
            print(f"[RESUME] ❌ Run folder not found for ID: {self.cfg.resume_wandb_id}")
            return

        all_ckpts = []
        for f in run_folders:
            ckpts = glob(os.path.join(f, "files", "*.pt"))
            all_ckpts.extend(ckpts)
            
        if not all_ckpts:
            print("[RESUME] ⚠️ No checkpoints found inside run folder.")
            return

        if self.cfg.resume_epoch is not None:
            epoch_str = f"epoch{self.cfg.resume_epoch:05d}"
            target_ckpts = sorted([c for c in all_ckpts if epoch_str in os.path.basename(c)])
            if not target_ckpts:
                print(f"[RESUME] ❌ Could not find checkpoint for epoch {self.cfg.resume_epoch}")
                return
            resume_path = target_ckpts[-1] # Take the latest one for that epoch
        else:
            resume_path = max(all_ckpts, key=os.path.getmtime)
            
        print(f"[RESUME] 📥 Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        # Helper to load into potentially compiled model
        def load_state(m, state):
            getattr(m, "_orig_mod", m).load_state_dict(state)
            # m.load_state_dict(state)

        load_state(self.model, checkpoint['model_state_dict'])
        
        if 'feat_extractor_state_dict' in checkpoint:
            print("[RESUME] 📥 Loading Feature Extractor state...")
            load_state(self.feat_extractor, checkpoint['feat_extractor_state_dict'])

        if checkpoint.get('scheduler_state_dict') is not None and self.scheduler is not None:
            print("[RESUME] 📥 Loading Scheduler state...")
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.cfg.override_lr:
                print(f"[RESUME] 🔧 Forcing new Learning Rate: {self.cfg.lr}")
                feat_params = list(self.feat_extractor.parameters())
                for param_group in self.optimizer.param_groups:
                    # Identify by parameter matching instead of just length if possible, 
                    # but length is a good fallback for different models.
                    is_feat = any(any(p is fp for fp in feat_params) for p in param_group['params'])
                    
                    if is_feat:
                         param_group['lr'] = self.cfg.lr_feat_extractor
                    else:
                         param_group['lr'] = self.cfg.lr
                
                # Update scheduler base_lrs to match new optimizer LRs
                if self.scheduler is not None:
                    self.scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if 'scaler_state_dict' in checkpoint:
            print("[RESUME] 📥 Loading GradScaler state...")
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

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
        
        # Helper to get clean state dict (handles torch.compile)
        def get_state_dict(m):
            return getattr(m, "_orig_mod", m).state_dict()

        save_dict = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': get_state_dict(self.model),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'config': vars(self.cfg)
        }
        
        if self.cfg.finetune_feat_extractor:
             save_dict['feat_extractor_state_dict'] = get_state_dict(self.feat_extractor)
             
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
            from sklearn.decomposition import PCA
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
    def _log_training_patch(self, mri, ct, pred, step, batch_idx, seg=None, pred_probs=None):
        """
        Visualizes MRI, Prediction, and CT (GT) for the first patch in the batch.
        Optionally overlays Segmentation (GT) and Predicted Probs (Argmax).
        """
        # 1. Prepare Data (Batch 0, Channel 0)
        img_in = mri[0, 0].detach().cpu().float().numpy()
        img_gt = ct[0, 0].detach().cpu().float().numpy()
        img_pred = pred[0, 0].detach().cpu().float().numpy()
        
        img_seg = None
        img_pred_seg = None
        
        if seg is not None:
            img_seg = seg[0, 0].detach().cpu().float().numpy()
        
        if pred_probs is not None:
            # pred_probs: [B, C, X, Y, Z] -> Argmax -> [X, Y, Z]
            img_pred_seg = torch.argmax(pred_probs[0], dim=0).detach().cpu().float().numpy()
        
        # Center indices
        cx, cy, cz = np.array(img_in.shape) // 2

        # 3 Rows (MRI, Pred, CT), 3 Cols (Axial, Sagittal, Coronal)
        # If seg exists, add 4th row (Seg GT, Seg Pred, Overlay?)
        
        nrows = 4 if img_seg is not None else 3
        fig, axes = plt.subplots(nrows, 3, figsize=(10, 3.5*nrows))
        
        # Helper to plot a row
        def plot_row(row_idx, vol, title_prefix, vmin=None, vmax=None, cmap='gray'):
            # Axial
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"{title_prefix} Ax")
            # Sagittal
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"{title_prefix} Sag")
            # Coronal
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap=cmap, vmin=vmin, vmax=vmax)
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
        
        # Row 4: Segmentation (if available)
        if img_seg is not None:
            # Determine common vmin/vmax for consistent coloring
            seg_vmin = 0
            seg_vmax = img_seg.max()
            if img_pred_seg is not None:
                seg_vmax = max(seg_vmax, img_pred_seg.max())

            # GT Seg
            axes[3, 0].imshow(np.rot90(img_seg[:, :, cz]), cmap='tab20', vmin=seg_vmin, vmax=seg_vmax, interpolation='nearest')
            axes[3, 0].set_title("GT Seg Ax")
            
            # Pred Seg (if available)
            if img_pred_seg is not None:
                axes[3, 1].imshow(np.rot90(img_pred_seg[:, :, cz]), cmap='tab20', vmin=seg_vmin, vmax=seg_vmax, interpolation='nearest')
                axes[3, 1].set_title("Pred Seg Ax")
                
                # Overlay on Pred (Axial) - Changed from GT on GT to Pred on Pred
                axes[3, 2].imshow(np.rot90(img_pred[:, :, cz]), cmap='gray', vmin=0, vmax=1)
                axes[3, 2].imshow(np.rot90(img_pred_seg[:, :, cz]), cmap='tab20', vmin=seg_vmin, vmax=seg_vmax, alpha=0.3, interpolation='nearest')
                axes[3, 2].set_title("Pred Overlay")
            else:
                axes[3, 1].axis('off')
                axes[3, 2].axis('off')

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

            if self.cfg.enable_profiling:
                step_t_data = 0.0
                step_t_fwd = 0.0
                step_t_bwd = 0.0
                t_step_start = time.time()
            
            for _ in range(self.cfg.accum_steps):
                if self.cfg.enable_profiling: t0 = time.time()
                batch = next(self.train_iter)
                if self.cfg.enable_profiling:
                    t1 = time.time()
                    step_t_data += (t1 - t0)
                
                # Extract tensors (torchio uses [tio.DATA])
                mri = batch['mri'][tio.DATA].to(self.device, non_blocking=True)
                ct = batch['ct'][tio.DATA].to(self.device, non_blocking=True)
                seg = batch['seg'][tio.DATA].to(self.device, non_blocking=True) if 'seg' in batch else None

                if self.cfg.enable_profiling: torch.cuda.synchronize(); t2 = time.time()

                # Call the training step (compiled or eager)
                # Note: self.train_step handles mixed precision context internally
                pred, loss, comps, pred_probs = self.train_step(mri, ct, seg)

                if self.cfg.enable_profiling:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    step_t_fwd += (t3 - t2)
                    step_t_bwd += 0 # Backward is inside train_step
                
                step_loss += loss.item()
                
                # Log patch (using the returned predictions)
                if self.cfg.wandb and step_idx == 0:
                    self._log_training_patch(mri, ct, pred, self.global_step, step_idx, seg, pred_probs)
                
                for k, v in comps.items():
                    comp_accum[k] = comp_accum.get(k, 0.0) + (v / self.cfg.accum_steps)

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + (list(self.feat_extractor.parameters()) if self.cfg.finetune_feat_extractor else []),
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Step scheduler per iteration for cosine
            if self.scheduler is not None and self.cfg.scheduler_type == "cosine":
                if self.scheduler.last_epoch < self.scheduler.T_max:
                    self.scheduler.step()

            total_loss += step_loss
            total_grad += grad_norm.item()
            
            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.accum_steps

            pbar_dict = {"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"}
            
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.cfg.enable_profiling:
                t_step_end = time.time()
                step_t_total = t_step_end - t_step_start
                avg_batch_data = step_t_data / self.cfg.accum_steps * 1000
                avg_batch_fwd = step_t_fwd / self.cfg.accum_steps * 1000
                avg_batch_bwd = step_t_bwd / self.cfg.accum_steps * 1000
                pbar_dict.update({
                    "dt(ms)": f"{avg_batch_data:.3f}", 
                    "fwd(ms)": f"{avg_batch_fwd:.3f}",
                    "bwd(ms)": f"{avg_batch_bwd:.3f}",
                    "lr": f"{current_lr:.2e}"
                })
            
            pbar.set_postfix(pbar_dict)
            
            if self.cfg.wandb:
                log_dict = {"info/samples_seen": self.samples_seen}
                if self.cfg.enable_profiling:
                    log_dict.update({
                        "info/time_data(ms)": avg_batch_data,
                        "info/time_forward(ms)": avg_batch_fwd,
                        "info/time_backward(ms)": avg_batch_bwd,
                        "info/time_step_total(s)": step_t_total,
                    })
                wandb.log(log_dict, step=self.global_step)

            # ---------------------------------------------------------
            # RAM 모니터링 (Optional)
            # ---------------------------------------------------------
            if self.cfg.wandb and self.cfg.enable_profiling and step_idx % 100 == 0:
                sys_percent, app_gb = get_ram_info()
                
                queue = self.train_loader.dataset
                curr_patches = len(queue.patches_list)
                
                wandb.log({
                    "perf/ram_system_percent": sys_percent,
                    "perf/ram_app_total_gb": app_gb,
                    "perf/queue_curr_patches": curr_patches
                }, step=self.global_step)
                
                if sys_percent > 90:
                    tqdm.write(f"[WARNING] ⚠️ RAM usage critical: {sys_percent}% (App: {app_gb:.2f} GB)")
            # ---------------------------------------------------------
            
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

                # Optimization: AMP for faster inference
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = sliding_window_inference(
                        inputs=mri, 
                        roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size), 
                        # Optimization: Increase batch size
                        sw_batch_size=max(self.cfg.val_sw_batch_size, 16), 
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
            l_val, _ = self.loss_fn(pred, ct)
            met['loss'] = l_val.item()
            
            # Validation Dice (Semantic Consistency)
            # Only run if enabled in config, as Teacher inference is heavy
            if getattr(self.cfg, "validate_dice", False) and self.teacher_model is not None and seg is not None:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Always use sliding window for teacher on full volumes to prevent OOM
                    pred_probs = sliding_window_inference(
                        inputs=pred, 
                        roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size), 
                        sw_batch_size=max(self.cfg.val_sw_batch_size, 16), 
                        predictor=self.teacher_model,
                        overlap=self.cfg.val_sw_overlap,
                        mode="gaussian",
                        device=self.device
                    )
                        
                val_dice_loss = self.loss_fn.soft_dice_loss(pred_probs, seg)
                met['dice'] = 1.0 - val_dice_loss.item()
            
            for k, v in met.items():
                val_metrics[k].append(v)
            
            # Viz & Save
            if i in self.val_viz_indices:
                save_path = None
                # Optimization: Only save volumes if visualized
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
                    "train/total": loss, 
                    "info/grad_norm": gn, 
                    "info/epoch_duration": ep_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/lr": current_lr,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                }
                for k, v in comps.items(): log[k.replace("loss_", "train/")] = v
                wandb.log(log, step=self.global_step)
                
            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)
                
        self.save_checkpoint(self.cfg.total_epochs, is_final=True)
        if self.cfg.wandb:
            wandb.log({"info/samples_seen_total": self.samples_seen}) # Log total samples at the end
        if self.cfg.wandb: wandb.finish()
        print(f"✅ Training Complete. Total Time: {time.time() - self.global_start_time:.1f}s")
