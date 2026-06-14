"""One-off: run CADS task 559 on 1ABA058 and save to /tmp/cads_1ABA058/."""
import os, sys, shutil, torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CADS"))

from cads.utils.libs import setup_nnunet_env, get_model_weights_dir, check_or_download_model_weights
from cads.utils.inference import predict

TASK_ID = 559
CT_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5x1.5x1.5mm/test/1ABA058/ct.nii.gz"
OUT_DIR = "/tmp/cads_1ABA058"
WEIGHTS_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/cads_weights"

os.environ["CADS_WEIGHTS_PATH"] = WEIGHTS_PATH
setup_nnunet_env()
model_folder = get_model_weights_dir()
check_or_download_model_weights(TASK_ID)

tmp_in = os.path.join(OUT_DIR, "input")
tmp_out = os.path.join(OUT_DIR, "output")
os.makedirs(tmp_in, exist_ok=True)
os.makedirs(tmp_out, exist_ok=True)

ct_link = os.path.join(tmp_in, "1ABA058.nii.gz")
if os.path.exists(ct_link):
    os.remove(ct_link)
os.symlink(CT_PATH, ct_link)

predict(
    files_in=[ct_link],
    folder_out=tmp_out,
    model_folder=model_folder,
    task_ids=[TASK_ID],
    folds="all",
    use_cpu=not torch.cuda.is_available(),
    preprocess_cads=True,
    postprocess_cads=True,
    save_all_combined_seg=False,
    verbose=True,
)

raw_seg = os.path.join(tmp_out, "1ABA058", f"1ABA058_part_{TASK_ID}.nii.gz")
final_seg = os.path.join(OUT_DIR, "cads_ct_seg_559.nii.gz")
if os.path.exists(raw_seg):
    shutil.copy(raw_seg, final_seg)
    print(f"Saved: {final_seg}")
else:
    print(f"Warning: expected output not found at {raw_seg}")
    print("Contents of tmp_out:")
    for root, dirs, files in os.walk(tmp_out):
        for f in files:
            print(" ", os.path.join(root, f))
