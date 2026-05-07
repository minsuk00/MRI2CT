"""Data pipeline for MRI->CT training.

Three transform stages, each doing one job:

  1. `get_cached_transforms(...)` — deterministic CPU preproc cached on disk
     via `monai.data.PersistentDataset`. Output: full padded volumes.
       load -> RAS -> normalize (CT clip / MRI minmax|percentile) ->
       record pre-pad shape -> pad-end to multiple of res_mult -> uint8 masks.

  2. `get_random_crop(...)` — random patch crop. Run on CPU in the DataLoader
     workers by wrapping the PersistentDataset with `monai.data.Dataset`.
     Output: uniform-shape patches that collate with default `list_data_collate`.
     Skip this stage entirely for full-volume training (e.g. MAISI).

  3. `get_gpu_transforms(...)` — random photometric + spatial augmentations
     applied to the **whole batch at once** via `gpu_augment_batch`, using
     `batchaug` (independent per-batch-element parameters, fused grid_sample
     for spatial ops via lazy=True). Replaced the previous MONAI per-item
     loop, which lives at `src/_archive/data_monai.py`.
       Order: Flip -> Rotate90 -> LowRes -> GaussNoise -> BiasField
       -> Gibbs -> Contrast -> Smooth -> Sharpen -> Affine -> 3DElastic
       -> RandConv -> ScaleIntensity.
"""

import os
import sys
from glob import glob

import numpy as np
import torch
from monai.transforms import (
    CastToTyped,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandSpatialCropSamplesd,
    RandWeightedCropd,
    ScaleIntensityd as MonaiScaleIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
)

# ── batchaug import workaround ──────────────────────────────────────────────
# The cloned batchaug repo lives at /home/minsukc/MRI2CT/batchaug/, which
# collides with the editable-installed package via cwd-resolved namespace
# packages when scripts run from MRI2CT root. Strip those entries from
# sys.path so the proper site-packages finder wins.
_BAD = {"", "/home/minsukc/MRI2CT"}
sys.path = [p for p in sys.path if p not in _BAD]
import batchaug as B  # noqa: E402


# ============================================================================
# Subject discovery — file-system helpers (no transforms / no I/O of volumes)
# ============================================================================
def get_subject_paths(root, relative_path):
    """Return {ct, mri, [body_mask]} paths for a single subject directory."""
    subj_dir = os.path.join(root, relative_path)

    ct_path = os.path.join(subj_dir, "ct.nii.gz")
    if not os.path.exists(ct_path):
        ct_path = os.path.join(subj_dir, "ct.nii")

    mr_candidates = sorted(glob(os.path.join(subj_dir, "moved_mr*.nii*")))
    if not mr_candidates:
        raise FileNotFoundError(f"No registered MRI (moved_mr*.nii*) found in {subj_dir}")
    mr_path = mr_candidates[0]

    if not os.path.exists(ct_path) or not os.path.exists(mr_path):
        raise FileNotFoundError(f"Missing files in {subj_dir}")

    paths = {"ct": ct_path, "mri": mr_path}

    body_mask_path = os.path.join(subj_dir, "mask.nii.gz")
    if not os.path.exists(body_mask_path):
        body_mask_path = os.path.join(subj_dir, "mask.nii")
    if os.path.exists(body_mask_path):
        paths["body_mask"] = body_mask_path

    return paths


def get_split_subjects(split_file, split_name):
    """Read `SPLIT SUBJECT_ID` lines and return all subject IDs for `split_name`."""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    valid = []
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == split_name:
                valid.append(parts[1])
    return sorted(valid)


def get_region_key(subj_id):
    """Map a subject ID like '1ABA005' to a region label ('abdomen' here)."""
    mapping = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if not subj_id or len(subj_id) < 2:
        return "abdomen"
    code_2 = subj_id[1:3].upper()
    code_1 = subj_id[1:2].upper()
    if code_2 in mapping:
        return mapping[code_2]
    if code_1 in mapping:
        return mapping[code_1]
    return "abdomen"


# ============================================================================
# Build per-subject dicts for MONAI Dataset / PersistentDataset
# ============================================================================
def build_data_dicts(root_dir, subjects, load_seg=False, load_body_mask=True):
    """Build [{mri, ct, [body_mask], [seg], subj_id}, ...] of file paths.

    Set `load_body_mask=False` for eval splits where masks may be absent
    (mask is required for training because of weighted sampling / val_body).
    """
    items = []
    for s in subjects:
        try:
            paths = get_subject_paths(root_dir, s)
        except Exception as e:
            print(f"  [ERROR] Failed to resolve paths for {s}: {e}")
            continue

        if load_body_mask and "body_mask" not in paths:
            print(f"  [ERROR] Missing body mask for {s}. Skipping.")
            continue

        d = {"mri": paths["mri"], "ct": paths["ct"], "subj_id": s}
        if load_body_mask and "body_mask" in paths:
            d["body_mask"] = paths["body_mask"]

        if load_seg:
            seg_path = os.path.join(root_dir, s, "ct_seg.nii")
            if not os.path.exists(seg_path):
                seg_path_gz = os.path.join(root_dir, s, "ct_seg.nii.gz")
                seg_path = seg_path_gz if os.path.exists(seg_path_gz) else None
            if seg_path is None:
                print(f"  [WARNING] Segmentation missing for {s}. Skipping.")
                continue
            d["seg"] = seg_path

        items.append(d)
    return items


# ============================================================================
# One small custom transform (snapshot pre-pad shape for `unpad`)
# ============================================================================
class RecordOriginalShapeD(MapTransform):
    """Snapshot the spatial shape of `ref_key` into `original_shape` (used by `common.utils.unpad`)."""

    def __init__(self, ref_key: str = "ct"):
        super().__init__([ref_key], allow_missing_keys=False)
        self.ref_key = ref_key

    def __call__(self, data):
        d = dict(data)
        d["original_shape"] = torch.tensor(list(d[self.ref_key].shape[1:]))
        return d


class RecordAffineD(MapTransform):
    """Snapshot `ref_key`'s MetaTensor.affine (4x4) and per-axis spacing as plain tensor keys.

    Required because `PersistentDataset` uses `torch.load(weights_only=True)` on cache read,
    which strips the MetaTensor subclass — so `.affine` is unavailable on cache hits (epoch ≥ 1).
    Stores `<ref>_affine` (4,4) for NIfTI export and `<ref>_spacing` (3,) for MAISI's spacing input.
    """

    def __init__(self, ref_key: str = "ct"):
        super().__init__([ref_key], allow_missing_keys=False)
        self.ref_key = ref_key
        self.affine_key = f"{ref_key}_affine"
        self.spacing_key = f"{ref_key}_spacing"

    def __call__(self, data):
        d = dict(data)
        ref = d[self.ref_key]
        affine = getattr(ref, "affine", None)
        if affine is None:
            d[self.affine_key] = torch.eye(4, dtype=torch.float32)
            d[self.spacing_key] = torch.ones(3, dtype=torch.float32)
        else:
            aff = affine.float() if hasattr(affine, "float") else torch.as_tensor(affine, dtype=torch.float32)
            d[self.affine_key] = aff.clone()
            d[self.spacing_key] = torch.linalg.norm(aff[:3, :3], dim=0)
        return d


class StripMetaD(MapTransform):
    """Drop the MetaTensor subclass on `keys` (returns plain torch.Tensor).

    Why: PersistentDataset's `torch.load(weights_only=True)` returns plain tensors on
    cache hit, while a cache miss runs the live transform pipeline which yields MetaTensor.
    With `batch_size > 1`, MONAI's `collate_meta_tensor_fn` chokes when a batch mixes the
    two states ('Tensor' object has no attribute 'meta'). Stripping at the end of the cached
    pipeline makes both states uniform — affine/spacing are already preserved out-of-band
    by `RecordAffineD` as separate plain-tensor keys.
    """

    def __init__(self, keys):
        super().__init__(keys, allow_missing_keys=True)

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            v = d[k]
            if hasattr(v, "as_tensor"):
                d[k] = v.as_tensor()
        return d


# ============================================================================
# Pipeline factories
# ============================================================================
def get_cached_transforms(
    *,
    patch_size: int,
    res_mult: int,
    enforce_ras: bool = True,
    mri_norm: str = "minmax",  # "minmax" | "percentile"
    ct_range: tuple = (-1024, 1024),
    mri_percentile_range: tuple = (0.0, 99.5),
    load_seg: bool = False,
    load_body_mask: bool = True,
    use_float16_storage: bool = False,
):
    """Deterministic CPU pipeline for PersistentDataset caching.

    Normalization presets:
      - amix / unet: ct_range=(-1024, 1024), mri_norm="minmax"
      - maisi:       ct_range=(-1000, 1000), mri_norm="percentile" (0.0–99.5)
    """
    spatial_keys = ["mri", "ct"] + (["body_mask"] if load_body_mask else [])
    if load_seg:
        spatial_keys.append("seg")

    transforms = [
        LoadImaged(keys=spatial_keys, image_only=True),
        EnsureChannelFirstd(keys=spatial_keys),
    ]
    if enforce_ras:
        transforms.append(Orientationd(keys=spatial_keys, axcodes="RAS"))

    # CT: clip-and-scale to [0, 1].
    transforms.append(ScaleIntensityRanged(keys=["ct"], a_min=ct_range[0], a_max=ct_range[1], b_min=0.0, b_max=1.0, clip=True))

    # MRI: per-volume minmax OR percentile clip-and-scale.
    if mri_norm == "percentile":
        transforms.append(
            ScaleIntensityRangePercentilesd(
                keys=["mri"],
                lower=mri_percentile_range[0],
                upper=mri_percentile_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        transforms.append(MonaiScaleIntensityd(keys=["mri"], minv=0.0, maxv=1.0))

    # Snapshot pre-pad shape, then pad at end up to patch_size and to next multiple of res_mult.
    transforms.extend(
        [
            RecordOriginalShapeD(ref_key="ct"),
            RecordAffineD(ref_key="ct"),
            SpatialPadd(keys=spatial_keys, spatial_size=(patch_size,) * 3, method="end", mode="constant"),
            DivisiblePadd(keys=spatial_keys, k=res_mult, method="end", mode="constant"),
        ]
    )

    # Mask dtype hygiene (matches legacy uint8 RAM optimization).
    mask_keys = (["body_mask"] if load_body_mask else []) + (["seg"] if load_seg else [])
    if mask_keys:
        transforms.append(CastToTyped(keys=mask_keys, dtype=torch.uint8))

    if use_float16_storage:
        transforms.append(CastToTyped(keys=["mri", "ct"], dtype=torch.float16))

    # Final: drop MetaTensor subclass so cache-hit (plain) and cache-miss (MetaTensor)
    # batches collate uniformly. RecordAffineD already saved the affine/spacing.
    transforms.append(StripMetaD(keys=spatial_keys))

    return Compose(transforms)


def get_random_crop(
    *,
    patch_size: int,
    use_weighted_sampler: bool = True,
    has_seg: bool = False,
    num_samples: int = 1,
):
    """CPU random patch crop, applied per-item in DataLoader workers.

    Wrap a PersistentDataset with `monai.data.Dataset(base, transform=this)`.
    With `num_samples > 1` each __getitem__ returns a list of N dicts; MONAI's
    default `list_data_collate` flattens lists, so the batch yields B*N patches.

    Skip this stage for full-volume training (MAISI).
    """
    spatial_keys = ["mri", "ct", "body_mask"] + (["seg"] if has_seg else [])
    if use_weighted_sampler:
        return RandWeightedCropd(
            keys=spatial_keys,
            w_key="body_mask",
            spatial_size=(patch_size,) * 3,
            num_samples=num_samples,
        )
    return RandSpatialCropSamplesd(
        keys=spatial_keys,
        roi_size=(patch_size,) * 3,
        num_samples=num_samples,
        random_size=False,
    )


def get_gpu_transforms(
    *,
    augment: bool = True,
    has_seg: bool = False,
    aug_prob: float = 0.33,
    elastic_sigma_range: tuple = (5, 8),
    elastic_magnitude_range: tuple = (100, 500),
    rotate_range_rad: float = np.pi / 4,
    scale_range: float = 0.2,
    shear_range: float = 0.2,
    bias_coeff_range: tuple = (-0.3, 0.3),
    bias_field_order: int = 3,
    gibbs_alpha: tuple = (0.1, 0.5),
    smooth_sigma_range: tuple = (0.25, 1.0),
    lowres_zoom_range: tuple = (0.25, 1.0),
    contrast_gamma_range: tuple = (0.5, 2.0),
    randconv_prob: float = 0.5,
    randconv_kernel_sizes: tuple = (1, 3, 5),
):
    """Random GPU-side augmentation pipeline using `batchaug` (batched, fused).

    Operates on a **5D batched dict** `{mri: (B,1,H,W,D), ct: (B,1,H,W,D),
    [seg: (B,1,H,W,D)]}`. Each item in the batch gets independent random
    parameters. Spatial ops (flip, rotate90, affine, elastic) are fused into
    a single grid_sample call via `lazy=True`. Photometric ops apply only
    to "mri".

    Note: batchaug splits MONAI's `Rand3DElasticd` into separate `RandAffined`
    + `Rand3DElasticd`. Lazy fusion means the cost is the same as one call.

    body_mask is intentionally NOT augmented here: it's only used for
    weighted cropping upstream and is not consumed after GPU aug.
    """
    spatial_keys = ["mri", "ct"] + (["seg"] if has_seg else [])

    # Per-key interp for the fused grid_sample at Compose level.
    mode_dict = {"mri": "bilinear", "ct": "bilinear"}
    if has_seg:
        mode_dict["seg"] = "nearest"

    if not augment:
        return B.Compose(transforms=[], lazy=True, mode=mode_dict)

    transforms = [
        B.RandAxisFlipd(keys=spatial_keys, prob=aug_prob),
        B.RandRotate90d(keys=spatial_keys, prob=aug_prob, max_k=3, spatial_axes=(0, 1)),
        B.RandSimulateLowResolutiond(keys=["mri"], prob=aug_prob, zoom_range=lowres_zoom_range),
        B.RandGaussianNoised(keys=["mri"], prob=aug_prob),
        B.RandBiasFieldd(
            keys=["mri"],
            prob=aug_prob,
            degree=bias_field_order,
            coeff_range=bias_coeff_range,
        ),
        B.RandGibbsNoised(keys=["mri"], prob=aug_prob, alpha=gibbs_alpha),
        B.RandAdjustContrastd(keys=["mri"], prob=aug_prob, gamma=contrast_gamma_range),
        B.RandGaussianSmoothd(
            keys=["mri"],
            prob=aug_prob,
            sigma_x=smooth_sigma_range,
            sigma_y=smooth_sigma_range,
            sigma_z=smooth_sigma_range,
        ),
        B.RandGaussianSharpend(keys=["mri"], prob=aug_prob),
        B.RandAffined(
            keys=spatial_keys,
            prob=aug_prob,
            rotate_range=(rotate_range_rad,) * 3,
            scale_range=(scale_range,) * 3,
            shear_range=(shear_range,) * 3,
            padding_mode="zeros",
        ),
        B.Rand3DElasticd(
            keys=spatial_keys,
            prob=aug_prob,
            sigma_range=elastic_sigma_range,
            magnitude_range=elastic_magnitude_range,
            padding_mode="zeros",
        ),
        B.RandConvd(
            keys=["mri"],
            prob=randconv_prob,
            kernel_sizes=randconv_kernel_sizes,
            mixing=True,
        ),
        B.ScaleIntensityd(keys=["mri", "ct"]),
    ]

    return B.Compose(transforms=transforms, lazy=True, mode=mode_dict)


# ============================================================================
# Train-loop helpers
# ============================================================================
def gpu_augment_batch(batch, gpu_transforms, device):
    """Apply the batched GPU aug pipeline to a 5D batched dict.

    Replaces the old decollate->loop->recollate path. batchaug operates on
    the whole batch in one call with independent per-element randomness.

    Tensors are moved to `device` if needed. `seg` (typically uint8 from
    cached pipeline) is cast to float32 for the duration of the spatial
    grid_sample, then cast back to its source dtype.

    MetaTensor stripping: batchaug ops do `tensor[0]` indexing internally;
    on batched MetaTensors this triggers MONAI's `_handle_batched -> decollate_batch`
    path which iterates batched meta dicts and fails with `TypeError: iteration
    over a 0-d array` when any meta value is a numpy scalar. Affine/spacing are
    already preserved out-of-band by RecordAffineD as plain-tensor keys, so we
    can drop the MetaTensor subclass safely here.
    """
    if gpu_transforms is None:
        return batch

    out = {}
    seg_dtype = None
    for k, v in batch.items():
        if torch.is_tensor(v):
            if hasattr(v, "as_tensor"):  # MetaTensor -> plain Tensor
                v = v.as_tensor()
            if v.device != device:
                v = v.to(device, non_blocking=True)
            # batchaug grid_sample needs float; seg from cached pipeline is uint8.
            if k == "seg" and not v.is_floating_point():
                seg_dtype = v.dtype
                v = v.float()
        out[k] = v

    out = gpu_transforms(out)

    if seg_dtype is not None and "seg" in out:
        out["seg"] = out["seg"].to(seg_dtype)
    return out


def default_monai_cache_dir() -> str:
    """Local NVMe cache dir shared across all trainers.

    Different trainers (different mri_norm/ct_range/res_mult) produce different
    transform-spec hashes, so PersistentDataset writes them to separate files
    in this single dir without collision.
    """
    user_id = os.environ.get("USER", "default")
    return os.path.join("/tmp", f"mri2ct_{user_id}_monai_cache")
