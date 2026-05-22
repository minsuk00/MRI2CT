"""Paper-faithful data pipeline for MC-IDDPM on SynthRAD.

Mirrors the notebook's active `CustomDataset` `Compose`:
    LoadImaged -> AddChanneld -> ResizeWithPadOrCrop(constant_values=-1)
    -> RandSpatialCropSamplesd -> ToTensord

Adapted for our SynthRAD layout: CT/MR are separate NIfTIs under one root dir,
discovered via the project's split-file utilities. Normalization follows the
paper exactly:
  CT clipped to [-1024, 1650] HU -> [-1, 1]
  MRI per-volume minmax -> [-1, 1]
No augmentation beyond uniform random crop (paper has none).
"""
import os

import torch
from monai.data import Dataset, PersistentDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandSpatialCropSamplesd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)

# Project utilities — we only reuse path/subject discovery; normalization is paper-native.
from common.data import (
    RecordAffineD,
    RecordOriginalShapeD,
    StripMetaD,
    build_data_dicts,
    default_monai_cache_dir,
    get_split_subjects,
)

# Paper preset (prostate variant, larger volumes — our thorax data is larger still).
PATCH = (128, 128, 4)  # RAS-ordered (R, A, S=axial); S is 4-slice slab per paper
CT_CLIP = (-1024, 1650)  # paper clip range


def build_cached_xform(load_body_mask: bool = True):
    """Deterministic CPU pipeline cached by PersistentDataset.

    Output dict keys after the transform:
      mri:            (1, D, H, W) float32 in [-1, 1], padded to >= PATCH
      ct:             (1, D, H, W) float32 in [-1, 1], padded to >= PATCH
      body_mask:      (1, D, H, W) uint8, padded (optional)
      original_shape: long tensor [3]   — pre-pad spatial dims for `unpad` at eval
      ct_affine:      (4, 4) float32    — for NIfTI export aligned with GT
      ct_spacing:     (3,)  float32
      subj_id:        str
    """
    img_keys = ["mri", "ct"]
    mask_keys = ["body_mask"] if load_body_mask else []
    spatial_keys = img_keys + mask_keys
    xforms = [
        LoadImaged(keys=spatial_keys, image_only=True),
        EnsureChannelFirstd(keys=spatial_keys),
        Orientationd(keys=spatial_keys, axcodes="RAS"),
        # Paper CT normalization: clip [-1024, 1650] HU, scale to [-1, 1].
        ScaleIntensityRanged(keys=["ct"], a_min=CT_CLIP[0], a_max=CT_CLIP[1],
                             b_min=-1.0, b_max=1.0, clip=True),
        # Paper MRI normalization: per-volume minmax to [-1, 1].
        ScaleIntensityd(keys=["mri"], minv=-1.0, maxv=1.0),
        # Snapshot pre-pad shape + affine for downstream unpad + NIfTI export.
        RecordOriginalShapeD(ref_key="ct"),
        RecordAffineD(ref_key="ct", key_prefix="ct"),
        # Pad mri/ct with -1 (= background after [-1,1] scaling, paper convention).
        SpatialPadd(keys=img_keys, spatial_size=PATCH, method="end",
                    mode="constant", constant_values=-1.0),
    ]
    if mask_keys:
        # Pad body_mask with 0 (no body in padding).
        xforms.append(
            SpatialPadd(keys=mask_keys, spatial_size=PATCH, method="end",
                        mode="constant", constant_values=0)
        )
        from monai.transforms import CastToTyped
        xforms.append(CastToTyped(keys=mask_keys, dtype=torch.uint8))
    xforms.append(StripMetaD(keys=spatial_keys))
    return Compose(xforms)


def build_crop(num_samples: int, has_body_mask: bool = True):
    """Uniform random crop — paper does NOT use a weighted sampler.

    With num_samples > 1, each __getitem__ returns a list of N dicts; MONAI's
    default `list_data_collate` flattens these, so a DataLoader with batch_size=B
    yields B*N patches per step (notebook: B=4, N=2 -> 8 patches per step).
    """
    keys = ["mri", "ct"] + (["body_mask"] if has_body_mask else [])
    return RandSpatialCropSamplesd(
        keys=keys,
        roi_size=PATCH,
        num_samples=num_samples,
        random_size=False,
    )


def build_datasets(cfg):
    """Return (train_ds, val_ds) for the MC-IDDPM Trainer.

    train_ds: random-cropped patches of shape PATCH.
    val_ds:   full padded volumes (sliding-window inference at eval time).
    """
    train_subj = get_split_subjects(cfg.split_file, "train")
    val_subj = get_split_subjects(cfg.split_file, "val")
    train_dicts = build_data_dicts(cfg.root_dir, train_subj, load_body_mask=True)
    val_dicts = build_data_dicts(cfg.root_dir, val_subj, load_body_mask=True)

    cache_dir = default_monai_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[MCDDPM-data] 💾 MONAI cache dir: {cache_dir}")
    print(f"[MCDDPM-data] Split — train={len(train_dicts)} val={len(val_dicts)}")

    cached = build_cached_xform(load_body_mask=True)
    crop = build_crop(num_samples=cfg.patches_per_volume, has_body_mask=True)

    train_base = PersistentDataset(data=train_dicts, transform=cached, cache_dir=cache_dir)
    train_ds = Dataset(data=train_base, transform=Compose([crop, ToTensord(keys=["mri", "ct"])]))

    val_ds = PersistentDataset(data=val_dicts, transform=cached, cache_dir=cache_dir)
    return train_ds, val_ds
