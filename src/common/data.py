import os
from glob import glob

import torch
import torchio as tio

from common.utils import anatomix_normalize


class DataPreprocessing(tio.Transform):
    def __init__(self, patch_size=96, enable_safety_padding=False, res_mult=32, use_weighted_sampler=False, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.enable_safety_padding = enable_safety_padding
        self.res_mult = res_mult
        self.use_weighted_sampler = use_weighted_sampler

    def apply_transform(self, subject):
        subject["ct"].set_data(anatomix_normalize(subject["ct"].data, clip_range=(-1024, 1024)))
        subject["mri"].set_data(anatomix_normalize(subject["mri"].data))

        subject["original_shape"] = torch.tensor(subject["ct"].spatial_shape)
        pad_offset = 0

        # Keys that must be transformed spatially
        spatial_keys = ["ct", "mri"]
        if "prob_map" in subject:
            spatial_keys.append("prob_map")
        if "seg" in subject:
            spatial_keys.append("seg")

        if self.enable_safety_padding:
            pad_val = self.patch_size // 2
            subject = tio.Pad(pad_val, padding_mode=0, include=spatial_keys)(subject)
            pad_offset = pad_val

        subject["pad_offset"] = torch.tensor(pad_offset)
        current_shape = subject["ct"].spatial_shape
        padding_params = []
        for dim in current_shape:
            target = max(self.patch_size, (int(dim) + self.res_mult - 1) // self.res_mult * self.res_mult)
            padding_params.extend([0, int(target - dim)])

        if any(p > 0 for p in padding_params):
            subject = tio.Pad(padding_params, padding_mode=0, include=spatial_keys)(subject)

        if self.use_weighted_sampler and "prob_map" not in subject:
            prob = (subject["ct"].data > 0.01).to(torch.uint8)
            subject.add_image(tio.LabelMap(tensor=prob, affine=subject["mri"].affine), "prob_map")

        # Final RAM optimization: Ensure masks are uint8 (1 byte)
        if "prob_map" in subject:
            subject["prob_map"].set_data(subject["prob_map"].data.to(torch.uint8))
        if "seg" in subject:
            subject["seg"].set_data(subject["seg"].data.to(torch.uint8))

        return subject


# class Float16Storage(tio.Transform):
#     """
#     RAM optimization: Casts MRI and CT to float16.
#     MUST be the last transform in the pipeline to avoid SimpleITK errors.
#     """

#     def apply_transform(self, subject):
#         subject["mri"].set_data(subject["mri"].data.to(torch.float16))
#         subject["ct"].set_data(subject["ct"].data.to(torch.float16))
#         return subject


def get_augmentations():
    return tio.Compose(
        [
            tio.RandomAffine(scales=(0.95, 1.1), degrees=7, translation=4, default_pad_value="minimum", p=0.8),
            tio.RandomFlip(axes=(0, 1, 2), p=0.5),
            tio.Clamp(0, 1),
            tio.Compose(
                [
                    tio.RandomBiasField(coefficients=0.5, order=2, p=0.4),
                    tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.4),
                    tio.RandomNoise(std=(0, 0.02), p=0.25),
                ],
                include=["mri"],
            ),
            tio.Clamp(0, 1),
        ]
    )


def get_subject_paths(root, relative_path):
    subj_dir = os.path.join(root, relative_path)

    # Check for ct.nii or ct.nii.gz
    ct_path = os.path.join(subj_dir, "ct.nii.gz")
    if not os.path.exists(ct_path):
        ct_path = os.path.join(subj_dir, "ct.nii")

    # Search for moved_mr*.nii or moved_mr*.nii.gz directly in subj_dir
    mr_candidates = sorted(glob(os.path.join(subj_dir, "moved_mr*.nii*")))
    if not mr_candidates:
        raise FileNotFoundError(f"No registered MRI (moved_mr*.nii*) found in {subj_dir}")
    mr_path = mr_candidates[0]

    if not os.path.exists(ct_path) or not os.path.exists(mr_path):
        raise FileNotFoundError(f"Missing files in {subj_dir}")

    paths = {"ct": ct_path, "mri": mr_path}

    # Optional files (body mask for weighted sampling)
    body_mask_path = os.path.join(subj_dir, "mask.nii.gz")
    if not os.path.exists(body_mask_path):
        body_mask_path = os.path.join(subj_dir, "mask.nii")

    if os.path.exists(body_mask_path):
        paths["body_mask"] = body_mask_path

    return paths


def get_split_subjects(split_file, split_name):
    """Returns a list of subject IDs for a given split from a text file."""
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    valid_subjs = []
    with open(split_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                s_name, subj_id = parts[0], parts[1]
                if s_name == split_name:
                    valid_subjs.append(subj_id)
    return sorted(valid_subjs)


def get_region_key(subj_id):
    """Determines region key from subject ID (e.g., 1ABA005 -> abdomen)."""
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


def build_tio_subjects(root_dir, subjects, use_weighted_sampler=False, load_seg=False):
    """
    Converts a list of subject IDs into torchio.Subject objects.
    """
    tio_subjects = []
    for s in subjects:
        try:
            paths = get_subject_paths(root_dir, s)
            kwargs = {
                "mri": tio.ScalarImage(paths["mri"]),
                "ct": tio.ScalarImage(paths["ct"]),
                "subj_id": os.path.basename(s),
            }

            # Load body_mask as prob_map if it exists and we want to use weighted sampler
            if use_weighted_sampler and "body_mask" in paths:
                kwargs["prob_map"] = tio.LabelMap(paths["body_mask"])

            # Conditionally load segmentation
            if load_seg:
                seg_path = os.path.join(root_dir, s, "ct_seg.nii")
                if os.path.exists(seg_path):
                    kwargs["seg"] = tio.LabelMap(seg_path)
                else:
                    # In some cases we might want to skip, but here we keep it strict
                    # if the caller asked for segs, they must exist.
                    print(f"  [WARNING] Segmentation missing for {s}. Skipping.")
                    continue

            tio_subjects.append(tio.Subject(**kwargs))
        except Exception as e:
            print(f"  [ERROR] Failed to build subject {s}: {e}")
            continue

    return tio_subjects


def stage_data_to_local(gpfs_root, subjects, cfg, prefix="Trainer"):
    """
    Copies a list of subject directories from GPFS to local NVMe RAID for faster I/O.
    Returns the new local root path.
    """
    if not getattr(cfg, "stage_data", True):
        print(f"[{prefix}] ⏩ Skipping data staging (staying on GPFS).")
        return gpfs_root

    user_id = os.environ.get("USER", "default")
    # Extract resolution string (e.g., '1.5x1.5x1.5mm') from GPFS path to differentiate cache
    res_str = os.path.basename(gpfs_root.rstrip("/"))
    local_root = os.path.join("/tmp", f"mri2ct_{user_id}_{res_str}")

    if not os.path.exists("/tmp") or not os.access("/tmp", os.W_OK):
        print(f"[{prefix}] ⚠️ Local storage not available. Staying on GPFS.")
        return gpfs_root

    if os.path.exists(local_root):
        print(f"[{prefix}] ♻️  Local cache found at {local_root}. Syncing updates...")
    else:
        print(f"[{prefix}] 🚚 Staging data to local NVMe: {local_root}")
        os.makedirs(local_root, exist_ok=True)

    # Construct rsync includes dynamically
    includes = ["--include='*/'", "--include='ct.nii*'", "--include='moved_mr*.nii*'"]

    # Only sync masks/segs if needed
    if getattr(cfg, "use_weighted_sampler", False):
        includes.append("--include='mask.nii*'")
    if getattr(cfg, "dice_w", 0) > 0 or getattr(cfg, "validate_dice", False):
        includes.append("--include='ct_seg.nii*'")

    includes.append("--exclude='*'")
    inc_str = " ".join(includes)

    # Sync only the subjects we actually need
    print(f"  - Syncing {len(subjects)} unique subjects to local storage...")
    for subj_id in subjects:
        src = os.path.join(gpfs_root, subj_id)
        dst = os.path.join(local_root, subj_id)
        if os.path.exists(src):
            os.system(f"rsync -am {inc_str} {src}/ {dst}/")

    return local_root
