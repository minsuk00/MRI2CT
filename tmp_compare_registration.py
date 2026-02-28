import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
TEST_CASES = [
    ("1THB034", "moved_mr_g4_hw4_l1.25_ga2_icTrue_median_test_3.nii", "disp_mr_g4_hw4_l1.25_ga2_icTrue_median_test_3.nii.gz")
]

BASE_DATA = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/native_masked"
TMP_DIR = "tmp_reg_test"

def normalize(arr, min_val=None, max_val=None):
    if min_val is not None and max_val is not None:
        arr = np.clip(arr, min_val, max_val)
    denom = arr.max() - arr.min()
    if denom == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / denom

def create_rgb(r, g):
    rgb = np.zeros((*r.shape, 3))
    rgb[..., 0] = r
    rgb[..., 1] = g
    return rgb

def create_comparison():
    for SUBJ_ID, WARPED_FILE, DISP_FILE in TEST_CASES:
        print(f"🧐 Visualizing {SUBJ_ID}...")
        
        FIXED_PATH = os.path.join(BASE_DATA, SUBJ_ID, "ct.nii")
        MOVING_PATH = os.path.join(BASE_DATA, SUBJ_ID, "mr.nii")
        WARPED_PATH = os.path.join(TMP_DIR, WARPED_FILE)
        DISP_PATH = os.path.join(TMP_DIR, DISP_FILE)
        
        fix_img = nib.load(FIXED_PATH)
        mov_img = nib.load(MOVING_PATH)
        wrp_img = nib.load(WARPED_PATH)
        disp_img = nib.load(DISP_PATH)
        
        sx, sy, sz = fix_img.header.get_zooms()
        fix = fix_img.get_fdata()
        mov = mov_img.get_fdata()
        wrp = wrp_img.get_fdata()
        disp = disp_img.get_fdata()
        
        if disp.ndim == 5:
            disp = disp.squeeze()
        axis_ch = -1 if disp.shape[-1] == 3 else 0
        disp_mag = np.linalg.norm(disp, axis=axis_ch)

        X, Y, Z = fix.shape
        
        orientations = [
            ("Axial", Z // 2, sy / sx),
            ("Coronal", Y // 2, sz / sx),
            ("Sagittal", X // 2, sz / sy)
        ]

        for name, s_idx, aspect in orientations:
            def get_sl(vol):
                if name == "Axial":
                    return np.rot90(vol[:, :, s_idx])
                elif name == "Coronal":
                    return np.rot90(vol[:, s_idx, :])
                else: # Sagittal
                    return np.rot90(vol[s_idx, :, :])

            f_sl = normalize(get_sl(fix), -450, 450)
            m_sl = normalize(get_sl(mov))
            w_sl = normalize(get_sl(wrp))
            d_sl = get_sl(disp_mag)

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            plt.suptitle(f"{SUBJ_ID} - {name} View (Median Test: grid_sp=4, ga=2)", fontsize=20)

            axes[0, 0].imshow(f_sl, cmap="gray", aspect=aspect)
            axes[0, 0].set_title("Fixed (CT)")
            axes[0, 1].imshow(m_sl, cmap="gray", aspect=aspect)
            axes[0, 1].set_title("Moving (Orig MRI)")
            axes[0, 2].imshow(w_sl, cmap="gray", aspect=aspect)
            axes[0, 2].set_title("Warped (Reg MRI)")

            axes[1, 0].imshow(create_rgb(f_sl, m_sl), aspect=aspect)
            axes[1, 0].set_title("Pre-Reg Overlay (R=CT, G=MRI)")
            axes[1, 1].imshow(create_rgb(f_sl, w_sl), aspect=aspect)
            axes[1, 1].set_title("Post-Reg Overlay (R=CT, G=Warped)")
            
            im = axes[1, 2].imshow(d_sl, cmap="jet", aspect=aspect)
            axes[1, 2].set_title("Displacement Magnitude")
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

            for ax in axes.flatten():
                ax.axis("off")

            save_name = f"reg_comp_{SUBJ_ID}_{name.lower()}.png"
            plt.tight_layout()
            plt.savefig(save_name, dpi=150)
            plt.close()
            
        print(f"✅ Saved visualizations for {SUBJ_ID}")

if __name__ == "__main__":
    create_comparison()
