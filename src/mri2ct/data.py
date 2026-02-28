import os
from glob import glob

import torch
import torchio as tio

from mri2ct.utils import anatomix_normalize


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
            prob = (subject["ct"].data > 0.01).to(torch.float32)
            subject.add_image(tio.LabelMap(tensor=prob, affine=subject["mri"].affine), "prob_map")

        return subject


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


def get_subject_paths(root, relative_path, use_registered=True):
    subj_dir = os.path.join(root, relative_path)
    ct_path = os.path.join(subj_dir, "ct.nii.gz")

    if use_registered:
        # Search for moved_mr*.nii or moved_mr*.nii.gz directly in subj_dir
        mr_candidates = sorted(glob(os.path.join(subj_dir, "moved_mr*.nii*")))
        if not mr_candidates:
            raise FileNotFoundError(f"No registered MRI (moved_mr*.nii*) found in {subj_dir}")
        mr_path = mr_candidates[0]
    else:
        mr_path = os.path.join(subj_dir, "mr.nii.gz")

    if not os.path.exists(ct_path) or not os.path.exists(mr_path):
        raise FileNotFoundError(f"Missing files in {subj_dir} (Registered: {use_registered})")

    paths = {"ct": ct_path, "mri": mr_path}

    # Optional files
    body_mask_path = os.path.join(subj_dir, "body_mask.nii.gz")
    if os.path.exists(body_mask_path):
        paths["body_mask"] = body_mask_path

    return paths


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
