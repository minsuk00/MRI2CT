from types import SimpleNamespace

import torch

from common.labels import BONE_CLASS_INDICES, CADS_35_CLASS_NAMES


class Config(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            setattr(self, key, value)


DEFAULT_CONFIG = {
    # Data Augmentation (Training only)
    "use_cutout": False,
    "cutout_prob": 0.5,
    "cutout_alpha": 1.0,
    # System
    "root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked",
    "log_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs",
    "prediction_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/predictions/1.5x1.5x1.5mm_registered",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    "project_name": "mri2ct",
    # Data
    "split_file": "splits/center_wise_split.txt",
    # Upfront NIfTI rsync GPFS->/tmp is now redundant: PersistentDataset caches
    # preprocessed tensors on /tmp during epoch 0 (~3-4 min in workers, ~55 GB
    # at fp16 for ~600 subjects). Old staging duplicated raw NIfTIs (~18 GB).
    "stage_data": False,
    "augment": True,
    "patch_size": 128,
    "patches_per_volume": 1,
    "data_queue_max_length": 100,
    "data_queue_num_workers": 4,
    "anatomix_weights": "v1_3",  # "v1", "v1_2", "v1_3"
    "teacher_weights_path": "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_cads_35_center_wise_di54npq3_epoch_1000.pth",
    "res_mult": 32,
    "analyze_shapes": True,
    "enable_profiling": False,
    # Monitoring (RAM/VRAM/timings logged to wandb under monitoring/)
    "monitor_resources": True,
    "monitor_interval": 10,  # log every N steps; covers worker children via psutil PSS
    "use_float16_storage": True,  # Halves PersistentDataset cache size (~104GB->55GB) and warmup time. mri/ct in [0,1] don't need fp32.
    "enforce_ras": True,
    "mri_norm": "minmax",  # "minmax" or "percentile" (0.0–99.5, same as MAISI)
    # Training
    "lr": 3e-4,
    "scheduler_type": "cosine",  # "plateau", "cosine", None
    "scheduler_min_lr": 0.0,
    "val_interval": 1,
    "sanity_check": True,
    "accum_steps": 2,
    "model_save_interval": 200,
    "viz_limit": 2,
    "viz_interval": 25,  # Log train/patch_viz figure every N epochs (was every epoch)
    "viz_force_include": ["1THB011"],  # Always visualize these val subjects (in addition to stratified pick)
    "viz_pca": False,
    "steps_per_epoch": 1000,
    "finetune_feat_extractor": False,
    "feat_norm": "instance",  # norm layer for feat extractor: "instance", "batch"
    "lr_feat_extractor": 1e-5,
    "override_lr": False,
    # Model Choice
    "model_type": "anatomix_translator",
    "compile_mode": "model",  # Options: None, "model", "full"
    "total_epochs": 5001,
    "dropout": 0,
    # CNN Specifics
    "batch_size": 4,  # volumes per batch in MONAI trainers; effective patch batch = batch_size * patches_per_volume
    "final_activation": "sigmoid",
    "use_weighted_sampler": True,
    "pass_mri_to_translator": False,
    "n_classes": 35,  # CADS 35-class grouping (incl. background); see common.labels.CADS_35_CLASS_NAMES
    "class_names": CADS_35_CLASS_NAMES,  # per-class Dice logging keys (dice_score_{idx}_{name})
    "seg_filename": "cads_grouped_35_labels_seg.nii.gz",  # GT seg label map for teacher Dice (must match teacher label space)
    # Sliding Window & Viz Options
    "val_patch_size": 256,
    "val_sw_batch_size": 2,
    "val_sw_overlap": 0.25,
    "validate_dice": False,
    "val_body_mask": True,  # If True, also log val_body/ metrics computed on body voxels only
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 0.1,
    "dice_w": 0.0,
    "dice_bone_w": 0.0,
    # Bone-family class indices in the CADS 35-class map: Skull, Bone-other,
    # Limb & girdle bones, Spine, Thoracic cage. All get dice_bone_w (others get dice_w).
    "dice_bone_indices": BONE_CLASS_INDICES,
    "perceptual_w": 0.0,  # Anatomix v1_4 perceptual loss weight (0 = off)
    "perceptual_layers": None,  # comma-separated decoder layer indices; None -> [38,45,52,65]
    "perceptual_metric": "ncc",  # "ncc" (normalized cross-correlation, default) or "l1"
    "perceptual_separable": True,  # LNCC box-sum via separable 1-D convs (exact, ~3.5x faster); False = dense
    "perceptual_fused": False,  # use the fused_lncc CUDA kernel for the ncc metric (drop-in for the Python box-conv)
    "wandb_tags": [],
    "wandb_note": "test_run",
    "resume_wandb_id": None,
    "resume_epoch": None,
    "diverge_wandb_branch": False,
    # Validation Saving
    "save_val_volumes": True,
    "val_save_interval": 0,  # Save epoch-stamped predictions every N epochs (0 = disabled)
    "full_val": True,
    # DRR Validation
    "val_drr": False,
    "val_drr_angles": 4,
    "val_drr_res": 256,
}
