"""Apply body mask to CT and MR volumes, copy seg and mask as-is."""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np


def process_subject(subject_dir: Path, out_root: Path) -> str:
    out_dir = out_root / subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    mask = nib.load(subject_dir / "mask.nii")
    mask_data = mask.get_fdata(dtype=np.float32) > 0

    for fname, bg_val in (("ct.nii", -1024.0), ("moved_mr.nii", 0.0)):
        img = nib.load(subject_dir / fname)
        data = img.get_fdata(dtype=np.float32)
        data[~mask_data] = bg_val
        nib.save(nib.Nifti1Image(data, img.affine, img.header), out_dir / fname)

    shutil.copy2(subject_dir / "ct_seg.nii", out_dir / "ct_seg.nii")
    shutil.copy2(subject_dir / "mask.nii", out_dir / "mask.nii")

    return subject_dir.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="dataset/1.5mm_registered_flat")
    parser.add_argument("--dst", default="dataset/1.5mm_registered_flat_masked")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    subjects = sorted([d for d in src.iterdir() if d.is_dir() and (d / "ct.nii").exists()])
    print(f"Processing {len(subjects)} subjects → {dst}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_subject, s, dst): s for s in subjects}
        for f in as_completed(futures):
            name = f.result()
            done += 1
            if done % 50 == 0 or done == len(subjects):
                print(f"  {done}/{len(subjects)}  {name}")

    print("Done.")


if __name__ == "__main__":
    main()
