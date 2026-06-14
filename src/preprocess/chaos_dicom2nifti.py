"""Convert CHAOS MR DICOM files to NIfTI and resample to 1.5mm isotropic.

Output structure:
  {out_root}/{split}/MR/{subject_id}/T1DUAL/inphase.nii.gz
  {out_root}/{split}/MR/{subject_id}/T1DUAL/outphase.nii.gz
  {out_root}/{split}/MR/{subject_id}/T2SPIR/t2spir.nii.gz
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk
import dicom2nifti
import dicom2nifti.settings as settings

TARGET_SPACING = (1.5, 1.5, 1.5)


def resample_to_isotropic(img: sitk.Image, spacing=TARGET_SPACING) -> sitk.Image:
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / spacing[i]))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(img)


def convert_and_resample(dicom_dir: Path, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "_tmp_dcm2nii"
    tmp_dir.mkdir(exist_ok=True)
    try:
        settings.disable_validate_slice_increment()
        dicom2nifti.convert_directory(str(dicom_dir), str(tmp_dir), compression=True, reorient=True)
        niftis = list(tmp_dir.glob("*.nii.gz"))
        if len(niftis) != 1:
            print(f"    ERROR: expected 1 NIfTI, got {len(niftis)} in {tmp_dir}")
            return False
        img = sitk.ReadImage(str(niftis[0]))
        img_resampled = resample_to_isotropic(img)
        sitk.WriteImage(img_resampled, str(out_path))
        return True
    finally:
        for f in tmp_dir.glob("*"):
            f.unlink()
        tmp_dir.rmdir()


def process_subject_mr(subj_dir: Path, out_base: Path) -> list[tuple[str, bool | None]]:
    results = []
    for phase, fname in [("InPhase", "inphase.nii.gz"), ("OutPhase", "outphase.nii.gz")]:
        dicom_dir = subj_dir / "T1DUAL" / "DICOM_anon" / phase
        out_path = out_base / "T1DUAL" / fname
        if out_path.exists():
            results.append((str(out_path), None))
        elif dicom_dir.exists():
            results.append((str(out_path), convert_and_resample(dicom_dir, out_path)))
        else:
            results.append((str(out_path), False))
    dicom_dir = subj_dir / "T2SPIR" / "DICOM_anon"
    out_path = out_base / "T2SPIR" / "t2spir.nii.gz"
    if out_path.exists():
        results.append((str(out_path), None))
    elif dicom_dir.exists():
        results.append((str(out_path), convert_and_resample(dicom_dir, out_path)))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chaos_root", default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/CHAOS")
    parser.add_argument("--out_root", default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/CHAOS/nifti")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["Train_Sets", "Test_Sets"])
    args = parser.parse_args()

    chaos_root = Path(args.chaos_root)
    out_root = Path(args.out_root)
    total, ok_count, skip_count, fail_count = 0, 0, 0, 0

    for split in args.splits:
        split_dir = chaos_root / split
        if not split_dir.exists():
            print(f"Skipping {split} (not found)")
            continue
        mod_dir = split_dir / "MR"
        if not mod_dir.exists():
            continue
        subj_ids = sorted(d.name for d in mod_dir.iterdir() if d.is_dir())
        if args.subjects:
            subj_ids = [s for s in subj_ids if s in args.subjects]

        for subj_id in subj_ids:
            subj_dir = mod_dir / subj_id
            out_base = out_root / split / "MR" / subj_id
            print(f"  [{split}/MR/{subj_id}]")
            for out_path, status in process_subject_mr(subj_dir, out_base):
                total += 1
                if status is None:
                    skip_count += 1
                    print(f"    SKIP  {out_path}")
                elif status:
                    ok_count += 1
                    print(f"    OK    {out_path}")
                else:
                    fail_count += 1
                    print(f"    FAIL  {out_path}")

    print(f"\nDone: {ok_count} converted, {skip_count} skipped, {fail_count} failed / {total} total")
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
