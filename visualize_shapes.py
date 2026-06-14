
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def visualize():
    root_dir = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
    subject_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"Gathering shapes for {len(subject_dirs)} subjects...")
    shapes = []
    for s in subject_dirs:
        subj_path = os.path.join(root_dir, s)
        p = os.path.join(subj_path, "ct.nii.gz")
        if not os.path.exists(p): p = os.path.join(subj_path, "ct.nii")
        
        if os.path.exists(p):
            try:
                img = nib.load(p)
                shapes.append(img.header.get_data_shape())
            except: continue

    shapes = np.array(shapes)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ['#ff9999','#66b3ff','#99ff99']
    dims = ['X (Width)', 'Y (Height)', 'Z (Depth)']

    for i, ax in enumerate(axes):
        data = shapes[:, i]
        ax.hist(data, bins=30, color=colors[i], edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution of {dims[i]}', fontsize=14)
        ax.set_xlabel('Voxels', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add vertical lines for multiples of 32 for reference
        for val in range(0, int(data.max()) + 32, 32):
            ax.axvline(val, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    save_path = "dataset_dimensions.png"
    plt.savefig(save_path)
    print(f"✅ Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize()
