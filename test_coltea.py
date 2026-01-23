import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

# Standard Metrics (CPU)
from skimage.metrics import structural_similarity as ssim

# Model Imports
from anatomix.model.network import Unet
# from models import CNNTranslator
from main_coltea import CNNTranslator

# --- CONFIGURATION ---
class TestConfig:
    def __init__(self):
        self.root_dir = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/Coltea-Lung-CT-100W/data_nifti_3mm"
        self.anatomix_weights = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G.pth"
        # self.cnn_weights = "/home/minsukc/MRI2CT/results/models/checkpoints/cnn_epoch100_20260121_0920.pt" #NOTE: change
        self.cnn_weights = "/home/minsukc/MRI2CT/results/models/checkpoints/cnn_epoch150_20260122_0352.pt" #NOTE: change # 150 epochs using new weights
        # self.cnn_weights = "/home/minsukc/MRI2CT/results/models/cnn_best.pt"
        
        self.output_dir = "./results/test_evaluation"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # --- TRAINING NORMALIZATION SETTINGS ---
        # self.min_clip = -1000 
        # self.max_clip = 1000
        self.min_clip = -450 
        self.max_clip = 450
        
        self.patch_size = 96
        # self.cnn_depth = 5
        # self.cnn_hidden = 64
        self.cnn_depth = 9
        self.cnn_hidden = 128
        self.final_activation = "sigmoid"

# --- HELPER FUNCTIONS ---
def minmax(arr, min_val, max_val):
    """Normalize [min_val, max_val] to [0, 1] for Model Input"""
    arr = np.clip(arr, min_val, max_val)
    return (arr - min_val) / (max_val - min_val)

def denormalize(arr, min_val, max_val):
    """Restore [0, 1] to [min_val, max_val] HU"""
    return arr * (max_val - min_val) + min_val

def to_author_space(arr_hu):
    """
    Convert Raw HU values -> Author's Metric Space.
    Formula: x / 1000 - 1
    """
    arr = np.copy(arr_hu)
    # arr = np.copy(arr_hu) + 1024
    # The author clipped values < 0 before scaling
    arr[arr < 0] = 0  
    arr = arr / 1000.0
    arr = arr - 1.0
    return arr

def load_test_subject(subject_path, cfg):
    """Loads Native (Input) and contrast (GT)."""
    native_path = glob.glob(os.path.join(subject_path, "native.nii*"))[0]
    # contrast_path = glob.glob(os.path.join(subject_path, "contrast.nii*"))[0]
    contrast_path = glob.glob(os.path.join(subject_path, "arterial.nii*"))[0]

    native_img = sitk.ReadImage(native_path)
    contrast_img = sitk.ReadImage(contrast_path)
    
    # Raw HU Data
    native_arr = sitk.GetArrayFromImage(native_img)
    contrast_arr = sitk.GetArrayFromImage(contrast_img)

    # Normalize for Model Input using Config Range
    native_in = minmax(native_arr, cfg.min_clip, cfg.max_clip)
    
    # # Handle Shape Mismatch
    # if native_in.shape != contrast_arr.shape:
    #     target_shape = [max(n, v) for n, v in zip(native_in.shape, contrast_arr.shape)]
    #     def resize(arr):
    #         pad = [(0, t-s) for t, s in zip(target_shape, arr.shape)]
    #         return np.pad(arr, pad, mode='constant', constant_values=0)
    #     native_in = resize(native_in)
    #     native_arr = resize(native_arr)
    #     contrast_arr = resize(contrast_arr)

    orig_shape = native_in.shape

    # Pad for Model (Multiple of 32)
    target_shape = [max(cfg.patch_size, (d + 31) // 32 * 32) for d in orig_shape]
    pad_width = [(0, t - o) for t, o in zip(target_shape, orig_shape)]
    
    native_pad = np.pad(native_in, pad_width, mode='constant', constant_values=0)

    return native_pad, native_arr, contrast_arr, orig_shape, os.path.basename(subject_path)

def calculate_metrics(native_hu, contrast_hu, pred_hu):
    """
    Computes metrics exactly like the authors: 
    1. Convert to Author Space (x/1000 - 1)
    2. Iterate over 2D slices
    3. Average the results
    """
    # 1. Convert ALL to Author Space
    native_norm = to_author_space(native_hu)
    contrast_norm = to_author_space(contrast_hu)
    pred_norm = to_author_space(pred_hu)

    # 2. Iterate Slices (Z-axis is index 0 in numpy from SimpleITK)
    mae_pre_list, rmse_pre_list, ssim_pre_list = [], [], []
    mae_post_list, rmse_post_list, ssim_post_list = [], [], []

    depth = native_norm.shape[0]
    
    # We infer data_range from the Ground Truth max-min per slice or globally.
    # Authors likely used default or global. Let's use global for consistency across slices.
    # dr_pre = native_norm.max() - native_norm.min()
    # dr_post = contrast_norm.max() - contrast_norm.min()

    for z in range(depth):
        # 2D Slices
        n_slice = native_norm[z]
        v_slice = contrast_norm[z]
        p_slice = pred_norm[z]

        dr_pre = max(n_slice.max() - n_slice.min(), v_slice.max() - v_slice.min())
        dr_post = max(p_slice.max() - p_slice.min(), v_slice.max() - v_slice.min())
        if dr_pre == 0: dr_pre = 1.0 
        if dr_post == 0: dr_post = 1.0

        # --- PRE (Baseline) ---
        mae_pre_list.append(np.mean(np.abs(n_slice - v_slice)))
        rmse_pre_list.append(np.sqrt(np.mean((n_slice - v_slice)**2)))
        ssim_pre_list.append(ssim(n_slice, v_slice, data_range=dr_pre))
        # ssim_pre_list.append(ssim(n_slice, v_slice))

        # --- POST (Model) ---
        mae_post_list.append(np.mean(np.abs(p_slice - v_slice)))
        rmse_post_list.append(np.sqrt(np.mean((p_slice - v_slice)**2)))
        ssim_post_list.append(ssim(p_slice, v_slice, data_range=dr_post))
        # ssim_post_list.append(ssim(p_slice, v_slice))

    # 3. Average
    return (
        np.mean(mae_pre_list), np.mean(rmse_pre_list), np.mean(ssim_pre_list),
        np.mean(mae_post_list), np.mean(rmse_post_list), np.mean(ssim_post_list)
    )

def save_visualization(native, contrast, pred, subj_id, save_dir, vmin, vmax):
    """
    Saves a visual comparison of the middle slice.
    Uses vmin/vmax to ensure all images share the exact same grayscale mapping.
    """
    z = native.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # --- VISUALIZATION SETTINGS ---
    # Global: -1000 (Air) to 1000 (Bone)
    # Soft Tissue: -160 to 240
    # Lungs: -1000 to -500
    # VMIN, VMAX = -1000, 1000 
    # VMIN, VMAX = -450, 450 

    # 1. Input (Native)
    axes[0].imshow(native[z], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Input (Native)\n{subj_id}")
    
    # 2. Ground Truth (contrast)
    axes[1].imshow(contrast[z], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth (Contrast)")
    
    # 3. Prediction
    axes[2].imshow(pred[z], cmap='gray', vmin=vmin, vmax=vmax)
    axes[2].set_title("Prediction")
    
    # 4. Difference Map (Absolute Error)
    # We fix the range (0 to 200 HU) so a small error looks distinct from a large error
    diff = np.abs(contrast[z] - pred[z])
    im = axes[3].imshow(diff, cmap='inferno', vmin=0, vmax=200)
    axes[3].set_title("Diff Error (GT Contrast - Pred. 0-200 HU)")
    
    # Add colorbar only to the diff map
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    for ax in axes: ax.axis('off')
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{subj_id}_compare.png"), dpi=150)
    plt.close()

# --- MAIN LOOP ---
@torch.no_grad()
def run_test():
    cfg = TestConfig()
    print(f"🚀 Starting Test (Author Metrics) on {cfg.device}")
    print(f"ℹ️  Input Normalization Range: [{cfg.min_clip}, {cfg.max_clip}]")
    
    # 1. Models
    anatomix = Unet(dimension=3, input_nc=1, output_nc=16, num_downs=5, ngf=20, norm="instance", interp="trilinear", pooling="Avg").to(cfg.device)
    anatomix = torch.compile(anatomix, mode="default")
    anatomix.load_state_dict(torch.load(cfg.anatomix_weights, map_location=cfg.device))
    anatomix.eval()
    
    decoder = CNNTranslator(in_channels=16, hidden_channels=cfg.cnn_hidden, depth=cfg.cnn_depth, final_activation=cfg.final_activation, dropout=0).to(cfg.device)
    if os.path.exists(cfg.cnn_weights):
        decoder.load_state_dict(torch.load(cfg.cnn_weights, map_location=cfg.device))
    else:
        print(f"❌ Weights not found: {cfg.cnn_weights}")
        return
    decoder.eval()

    # 2. Data
    test_dir = os.path.join(cfg.root_dir, "test")
    subjects = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    
    results = {'mae_pre': [], 'mae_post': [], 'rmse_pre': [], 'rmse_post': [], 'ssim_pre': [], 'ssim_post': []}

    for subj in tqdm(subjects, desc="Eval"):
        subj_path = os.path.join(test_dir, subj)
        try:
            # native_in: Padded & Normalized (0-1 based on Config Range)
            # native_hu, contrast_hu: Raw HU values
            native_in, native_hu, contrast_hu, orig_shape, _ = load_test_subject(subj_path, cfg)
        except Exception as e:
            print(f"Skip {subj}: {e}")
            continue

        # Inference
        native_t = torch.from_numpy(native_in).float().unsqueeze(0).unsqueeze(0).to(cfg.device)
        feats = anatomix(native_t)
        pred_t = decoder(feats)
        pred_np = pred_t.squeeze().cpu().numpy()

        # Unpad
        z, y, x = orig_shape
        pred_cropped = pred_np[:z, :y, :x]
        
        # Denormalize Output to HU
        # IMPORTANT: We denormalize based on the range we put IN to the model
        pred_hu = denormalize(pred_cropped, cfg.min_clip, cfg.max_clip)

        # Calculate Metrics (Author Style)
        # This function takes HU values and converts them to "Author Space" internally
        m_pre, r_pre, s_pre, m_post, r_post, s_post = calculate_metrics(native_hu, contrast_hu, pred_hu)
        
        results['mae_pre'].append(m_pre)
        results['rmse_pre'].append(r_pre)
        results['ssim_pre'].append(s_pre)
        
        results['mae_post'].append(m_post)
        results['rmse_post'].append(r_post)
        results['ssim_post'].append(s_post)

        save_visualization(native_hu, contrast_hu, pred_hu, subj, cfg.output_dir, cfg.min_clip, cfg.max_clip)

    # 4. Final Report & Save
    result_file = os.path.join(cfg.output_dir, "final_metrics.txt")
    
    print(f"\n📝 Saving results to: {result_file}")
    
    with open(result_file, "w") as f:
        # Helper to print AND write to file
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        log("\n" + "="*60)
        log("📊 COMPARISON RESULTS (Author's Metric Space)")
        log("="*60)
        log(f"{'Metric':<10} | {'Baseline (Pre)':<15} | {'Model (Post)':<15} | {'Improvement'}")
        log("-" * 60)
        
        def log_row(name, key_pre, key_post):
            m_pre = np.mean(results[key_pre])
            m_post = np.mean(results[key_post])
            
            if name == "SSIM":
                imp = m_post - m_pre # Higher is better
                symbol = "↑"
            else:
                imp = m_pre - m_post # Lower is better
                symbol = "↓"
                
            log(f"{name:<10} | {m_pre:.4f}          | {m_post:.4f}          | {symbol} {abs(imp):.4f}")

        log_row("MAE", 'mae_pre', 'mae_post')
        log_row("RMSE", 'rmse_pre', 'rmse_post')
        log_row("SSIM", 'ssim_pre', 'ssim_post')
        log("="*60)

if __name__ == "__main__":
    run_test()