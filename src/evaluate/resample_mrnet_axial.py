import os

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def resample_mrnet_to_axial_space(subject_id="0016", target_iso=1.5):
    """
    Resamples and re-orients Coronal and Sagittal volumes into Axial space
    at 1.5mm isotropic resolution.
    """
    base_path = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/MRNet/mrnetkneemris/MRNet-v1.0/train/"
    output_dir = f"MRNet_Knee_1.5mm/subject{subject_id}"

    # 1. Physical Parameters (3T Protocol)
    fov_mm = 150.0
    pixels = 256.0
    px_mm = fov_mm / pixels  # 0.5859 mm

    # Spacings [Slice, H, W]
    spacings = {"axial": [3.3, px_mm, px_mm], "coronal": [2.5, px_mm, px_mm], "sagittal": [2.5, px_mm, px_mm]}

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for plane in ["axial", "coronal", "sagittal"]:
        file_path = os.path.join(base_path, plane, f"{subject_id}.npy")
        if not os.path.exists(file_path):
            continue

        # Load (Slices, H, W)
        data = np.load(file_path).astype(np.float32)
        curr_spacing = spacings[plane]

        print(f"Processing {plane.upper()}...")

        # 2. Re-orient to Axial Space (Z, Y, X)
        if plane == "axial":
            # (Slices=Z, H=Y, W=X)
            reoriented = data
            reoriented_spacing = curr_spacing
        elif plane == "coronal":
            # Coronal Slices are Y. Input is (Y, Z, X)
            # Fix Y-flip for axial alignment
            reoriented = data.transpose(1, 0, 2)
            reoriented = np.flip(reoriented, axis=1)
            reoriented_spacing = [curr_spacing[1], curr_spacing[0], curr_spacing[2]]
        elif plane == "sagittal":
            # Sagittal Slices are X. Input is (X, Z, Y)
            reoriented = data.transpose(1, 2, 0)
            reoriented_spacing = [curr_spacing[1], curr_spacing[2], curr_spacing[0]]

        # 3. Resample
        zoom_factors = [s / target_iso for s in reoriented_spacing]
        resampled = zoom(reoriented, zoom_factors, order=3)
        resampled = np.clip(resampled, 0, 255).astype(np.uint8)

        # 4. Save as NIfTI (X, Y, Z for nibabel)
        nifti_data = resampled.transpose(2, 1, 0)
        affine = np.eye(4)
        np.fill_diagonal(affine, [target_iso, target_iso, target_iso, 1])

        nifti_img = nib.Nifti1Image(nifti_data, affine)
        out_path = os.path.join(output_dir, f"{plane}.nii.gz")
        nib.save(nifti_img, out_path)

        print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else "0016"
    resample_mrnet_to_axial_space(sid)
