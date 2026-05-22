"""SynthRAD MR->CT dataset for cWDM.

Thin wrapper around the project's MONAI cached pipeline (`src/common/data.py`).
Reuses build_data_dicts / get_cached_transforms / default_monai_cache_dir so
that CT/MR normalization and padding match the amix / unet / maisi baselines
exactly (fair head-to-head comparison).

Output per item:
  {
    'mri':            tensor (1, D, H, W)     # padded to mult-of-res_mult
    'ct':             tensor (1, D, H, W)     # ditto
    'body_mask':      tensor (1, D, H, W)     # uint8, optional
    'subj_id':        str
    'original_shape': tensor[3]               # pre-pad spatial dims (D, H, W)
    'ct_affine':      tensor (4, 4)           # for NIfTI export at val time
  }

No random crop, no augmentation. Full padded volumes -- cWDM is bs=1, FCN.
"""
import os
import sys

import torch
from monai.data import PersistentDataset

# Make `src.common.*` importable when running from anywhere under /home/minsukc/MRI2CT
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.common.data import (  # noqa: E402
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_split_subjects,
)


class SynthRADVolumes(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split_file,
        split_name="train",
        patch_size=128,
        res_mult=32,
        ct_range=(-1024, 1024),
        mri_norm="minmax",
        subjects_filter=None,
    ):
        subjects = get_split_subjects(split_file, split_name)
        if subjects_filter is not None:
            wanted = set(subjects_filter)
            subjects = [s for s in subjects if s in wanted]
            if not subjects:
                raise ValueError(
                    f"subjects_filter={subjects_filter} matched nothing in split '{split_name}' of {split_file}"
                )
        dicts = build_data_dicts(root_dir, subjects, load_seg=False, load_body_mask=True)
        if not dicts:
            raise RuntimeError(f"No usable subjects for split '{split_name}' under {root_dir}")

        xform = get_cached_transforms(
            patch_size=patch_size,
            res_mult=res_mult,
            enforce_ras=True,
            mri_norm=mri_norm,
            ct_range=ct_range,
            load_seg=False,
            load_body_mask=True,
        )
        self.ds = PersistentDataset(
            data=dicts,
            transform=xform,
            cache_dir=default_monai_cache_dir(),
        )
        self.subjects = [d["subj_id"] for d in dicts]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]

        def _plain(x):
            return x.as_tensor() if hasattr(x, "as_tensor") else x

        out = {
            "ct": _plain(item["ct"]).float(),
            "mri": _plain(item["mri"]).float(),
            "subj_id": item.get("subj_id", f"subj_{i}"),
            "original_shape": item["original_shape"],
            "ct_affine": item.get("ct_affine", torch.eye(4, dtype=torch.float32)),
        }
        if "body_mask" in item:
            out["body_mask"] = _plain(item["body_mask"])
        return out
