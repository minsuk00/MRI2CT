
import os
import nibabel as nib
import numpy as np
import glob

def analyze():
    # 1. Setup paths
    root_dir = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
    
    # 2. Discover all subjects in the flat directory
    subject_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"🔍 Found {len(subject_dirs)} subjects in {root_dir}")

    shapes = []
    padded_shapes = []
    found_count = 0

    for s in subject_dirs:
        # Check for ct.nii.gz or ct.nii
        subj_path = os.path.join(root_dir, s)
        p = os.path.join(subj_path, "ct.nii.gz")
        if not os.path.exists(p):
            p = os.path.join(subj_path, "ct.nii")
        
        if os.path.exists(p):
            try:
                # Use nib.load().header.get_data_shape() to avoid loading the full image data into RAM
                img = nib.load(p)
                shape = img.header.get_data_shape()
                shapes.append(shape)
                
                # Calculate what it would be after padding to multiple of 32
                padded = tuple(int(np.ceil(d / 32) * 32) for d in shape)
                padded_shapes.append(padded)
                found_count += 1
            except Exception as e:
                print(f"Error reading {s}: {e}")
        
        if found_count % 100 == 0 and found_count > 0:
            print(f"  Processed {found_count}...")

    shapes = np.array(shapes)
    padded_shapes = np.array(padded_shapes)

    if len(shapes) == 0:
        print("No valid CT shapes found.")
        return

    print(f"\n✅ Successfully analyzed {len(shapes)} volumes.")

    # 3. Statistics
    for i, dim in enumerate(['X', 'Y', 'Z']):
        d_data = shapes[:, i]
        p_data = padded_shapes[:, i]
        
        print(f"\n--- Dimension {dim} ---")
        print(f"  Min: {d_data.min()} | Max: {d_data.max()} | Mean: {d_data.mean():.1f}")
        
        # Bin by 32
        # Use a fixed start at 0, and go up to max padded
        max_val = int(p_data.max())
        bins = np.arange(0, max_val + 64, 32)
        hist, _ = np.histogram(d_data, bins=bins)
        
        print(f"  Histogram (Bin Size 32):")
        for j in range(len(hist)):
            count = hist[j]
            if count > 0:
                b_start = bins[j]
                b_end = bins[j+1] - 1
                print(f"    {b_start:3d}-{b_end:3d}: {count:4d}")

    # 4. Overall Max Padded Volume
    max_padded = tuple(padded_shapes.max(axis=0))
    print(f"\n🚀 Overall Max Padded Size (Multiple of 32): {max_padded}")
    
    # Peak VRAM calculation for float32 (128 channels)
    # Vol * Channels * 4 bytes
    voxels = np.prod(max_padded)
    peak_gb = (voxels * 128 * 4) / (1024**3)
    print(f"  Estimated Peak VAE Activation Size (Single Pass): {peak_gb:.2f} GB")
    print(f"  (Note: A40 has 44.42 GB usable VRAM)")

if __name__ == "__main__":
    analyze()
