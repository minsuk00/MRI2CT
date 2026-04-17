import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Parse arguments for testing
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=5000, help="Number of training steps")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
parser.add_argument("--log_interval", type=int, default=500, help="Steps between logging images")
parser.add_argument("--use_body_mask", action="store_true", help="Zero out non-body regions in MRI before feature extraction")
parser.add_argument("--feat_norm", type=str, default=None, choices=["minmax", "instance"], help="Normalize anatomix features before translator: minmax (global) or instance (per-channel)")
args = parser.parse_args()

sys.path.append("/home/minsukc/MRI2CT")
from anatomix.model.network import Unet
from monai.inferers import sliding_window_inference


# ----------------- UTILS -----------------
def anatomix_normalize(tensor):
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    v_min = tensor.min()
    v_max = tensor.max()
    denom = v_max - v_min
    if denom == 0:
        return torch.zeros_like(tensor)
    return (tensor - v_min) / denom


def normalize_ct(tensor):
    tensor = torch.as_tensor(tensor, dtype=torch.float32)
    tensor = torch.clamp(tensor, -1024.0, 1024.0)
    return (tensor + 1024.0) / 2048.0


def normalize_features(feat, mode):
    if mode == "minmax":
        f_min = feat.amin(dim=(2, 3, 4), keepdim=True)
        f_max = feat.amax(dim=(2, 3, 4), keepdim=True)
        denom = (f_max - f_min).clamp(min=1e-8)
        return (feat - f_min) / denom
    elif mode == "instance":
        mean = feat.mean(dim=(2, 3, 4), keepdim=True)
        std = feat.std(dim=(2, 3, 4), keepdim=True).clamp(min=1e-8)
        return (feat - mean) / std
    return feat


def clean_state_dict(state_dict):
    clean_dict = {}
    for k, v in state_dict.items():
        clean_dict[k.replace("module.", "")] = v
    return clean_dict


# ----------------- CONFIG -----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
patch_size = 128
max_steps = args.steps
batch_size = args.batch_size

# ----------------- DATA LOADING -----------------
TRAIN_SUBJECTS = ["1THA001", "1THA002", "1THA003", "1THA004", "1THA005"]
print(f"Loading training subjects into RAM: {TRAIN_SUBJECTS}")

train_data = []
for sub in TRAIN_SUBJECTS:
    mr = nib.load(os.path.join(data_root, sub, "moved_mr.nii")).get_fdata()
    ct = nib.load(os.path.join(data_root, sub, "ct.nii")).get_fdata()
    mr_t = anatomix_normalize(torch.from_numpy(mr).float())
    ct_t = normalize_ct(torch.from_numpy(ct).float())
    if args.use_body_mask:
        mask = torch.from_numpy(nib.load(os.path.join(data_root, sub, "mask.nii.gz")).get_fdata()).float()
        mr_t = mr_t * mask
    train_data.append({"mr": mr_t, "ct": ct_t})


def sample_batch(data_list, batch_size, patch_size):
    mr_batch, ct_batch = [], []
    for _ in range(batch_size):
        subject = random.choice(data_list)
        mr, ct = subject["mr"], subject["ct"]
        x = random.randint(0, mr.shape[0] - patch_size)
        y = random.randint(0, mr.shape[1] - patch_size)
        z = random.randint(0, mr.shape[2] - patch_size)
        mr_batch.append(mr[x: x + patch_size, y: y + patch_size, z: z + patch_size].unsqueeze(0))
        ct_batch.append(ct[x: x + patch_size, y: y + patch_size, z: z + patch_size].unsqueeze(0))
    return torch.stack(mr_batch), torch.stack(ct_batch)


# ----------------- MODELS -----------------
class SimpleTranslator(nn.Module):
    def __init__(self, input_nc=16):
        super().__init__()
        # Extremely simple CNN translator: just features -> CT
        self.net = nn.Sequential(
            nn.Conv3d(input_nc, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


print("\nInitializing Feature Extractors and Translators...")
# V1: BatchNorm
feat_v1 = Unet(3, 1, 16, 4, 16, norm="batch").to(device)
ckpt_v1 = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
feat_v1.load_state_dict(clean_state_dict(torch.load(ckpt_v1, map_location=device)), strict=False)
feat_v1.eval()
for p in feat_v1.parameters():
    p.requires_grad = False

# V1.2: InstanceNorm
feat_v12 = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg", use_bias=True).to(device)
ckpt_v12 = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth"
feat_v12.load_state_dict(clean_state_dict(torch.load(ckpt_v12, map_location=device)), strict=False)
feat_v12.eval()
for p in feat_v12.parameters():
    p.requires_grad = False

# Note: Translators take ONLY the features (16 channels) as input
trans_v1 = SimpleTranslator(input_nc=16).to(device)
trans_v12 = SimpleTranslator(input_nc=16).to(device)

opt_v1 = optim.Adam(trans_v1.parameters(), lr=1e-4)
opt_v12 = optim.Adam(trans_v12.parameters(), lr=1e-4)
criterion = nn.L1Loss()


# ----------------- FULL VOLUME EVALUATION LOGIC -----------------
test_subject = "1THB011"
mr_test_nib = nib.load(os.path.join(data_root, test_subject, "moved_mr.nii"))
ct_test_nib = nib.load(os.path.join(data_root, test_subject, "ct.nii"))

mr_test = mr_test_nib.get_fdata()
ct_test = ct_test_nib.get_fdata()

mr_test_t = anatomix_normalize(torch.from_numpy(mr_test).float())
ct_test_t = normalize_ct(torch.from_numpy(ct_test).float())
if args.use_body_mask:
    mask_test = torch.from_numpy(nib.load(os.path.join(data_root, test_subject, "mask.nii.gz")).get_fdata()).float()
    mr_test_t = mr_test_t * mask_test


def evaluate_and_plot(step, save_nifti=False):
    print(f"\n[Step {step}] Evaluating on Full Volume {test_subject}...")
    trans_v1.eval()
    trans_v12.eval()

    def infer_full(feat, trans, mr_tensor):
        def _inferer(x):
            return trans(normalize_features(feat(x), args.feat_norm))

        mr_tensor_in = mr_tensor.unsqueeze(0).unsqueeze(0).to(device)
        out = sliding_window_inference(inputs=mr_tensor_in, roi_size=(128, 128, 128), sw_batch_size=args.batch_size, predictor=_inferer, overlap=0.25)
        return out.squeeze().cpu().numpy()

    with torch.no_grad():
        pv1_norm_full = infer_full(feat_v1, trans_v1, mr_test_t)
        pv12_norm_full = infer_full(feat_v12, trans_v12, mr_test_t)

    # Denormalize [0, 1] -> [-1024, 1024]
    pv1_full_hu = (pv1_norm_full * 2048.0) - 1024.0
    pv12_full_hu = (pv12_norm_full * 2048.0) - 1024.0

    if save_nifti:
        mask_tag = "masked" if args.use_body_mask else "unmasked"
        print(f"Saving NIfTI volumes for step {step}...")
        nib.save(nib.Nifti1Image(mr_test_t.numpy(), mr_test_nib.affine), f"tests/research/ab_mr_input_{mask_tag}.nii.gz")
        nib.save(nib.Nifti1Image(pv1_full_hu, mr_test_nib.affine), "tests/research/ab_v1_full.nii.gz")
        nib.save(nib.Nifti1Image(pv12_full_hu, mr_test_nib.affine), "tests/research/ab_v12_full.nii.gz")
        print("Saved NIfTI volumes.")

    # Generate full slice visualization
    mid_z = mr_test.shape[2] // 2
    slice_mr = mr_test_t.numpy()[:, :, mid_z]
    slice_ct = (ct_test_t.numpy()[:, :, mid_z] * 2048.0) - 1024.0
    slice_pv1 = pv1_full_hu[:, :, mid_z]
    slice_pv12 = pv12_full_hu[:, :, mid_z]

    _, axes = plt.subplots(1, 4, figsize=(22, 6))
    plt.suptitle(f"BatchNorm vs InstanceNorm (Step {step})", fontsize=18)

    axes[0].imshow(slice_mr, cmap="gray")
    axes[0].set_title("Input MRI")

    axes[1].imshow(slice_ct, cmap="gray", vmin=-1000, vmax=200)
    axes[1].set_title("Ground Truth CT")

    axes[2].imshow(slice_pv1, cmap="gray", vmin=-1000, vmax=200)
    axes[2].set_title("V1 (BatchNorm) Translator")

    axes[3].imshow(slice_pv12, cmap="gray", vmin=-1000, vmax=200)
    axes[3].set_title("V1.2 (InstanceNorm) Translator")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plot_path = f"tests/research/ab_full_slice_results_step_{step}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved visual proof to {plot_path}")

    # Set back to train
    trans_v1.train()
    trans_v12.train()


# ----------------- TRAINING LOOP -----------------
print(f"\nStarting A/B Micro-Training for {max_steps} steps (Batch Size: {batch_size})...")
pbar = tqdm(range(1, max_steps + 1))
for step in pbar:
    mr_b, ct_b = sample_batch(train_data, batch_size, patch_size)
    mr_b = mr_b.to(device)
    ct_b = ct_b.to(device)

    with torch.no_grad():
        f_v1 = normalize_features(feat_v1(mr_b), args.feat_norm)
        f_v12 = normalize_features(feat_v12(mr_b), args.feat_norm)

    # V1 Train Step
    opt_v1.zero_grad()
    out_v1 = trans_v1(f_v1)
    loss_v1 = criterion(out_v1, ct_b)
    loss_v1.backward()
    opt_v1.step()

    # V1.2 Train Step
    opt_v12.zero_grad()
    out_v12 = trans_v12(f_v12)
    loss_v12 = criterion(out_v12, ct_b)
    loss_v12.backward()
    opt_v12.step()

    pbar.set_postfix({"V1": f"{loss_v1.item():.4f}", "V1.2": f"{loss_v12.item():.4f}"})

    if step % args.log_interval == 0:
        evaluate_and_plot(step, save_nifti=False)

evaluate_and_plot(max_steps, save_nifti=True)

print("Training Complete.")
