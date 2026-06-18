"""Demo: download all 9 CADS task weights, then run task='all' on one CT.

Outputs land in cads_demo_1ABA058/ inside the project dir:
  <pid>_part_551.nii.gz ... _part_559.nii.gz  (one per task, original geometry)
  <pid>_combined.nii.gz                        (CADS global 167-label merge)
  <pid>_snapshot.png                           (overview, best-effort)
"""
import os, sys, torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CADS"))

WEIGHTS_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/cads_weights"
os.environ["CADS_WEIGHTS_PATH"] = WEIGHTS_PATH

from cads.utils.libs import setup_nnunet_env, get_model_weights_dir, check_or_download_model_weights
setup_nnunet_env()
from cads.utils.inference import predict

ALL_TASKS = [551, 552, 553, 554, 555, 556, 557, 558, 559]
CT_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5x1.5x1.5mm/test/1ABA058/ct.nii.gz"
OUT_DIR = os.path.join(PROJECT_ROOT, "cads_demo_1ABA058")

model_folder = get_model_weights_dir()

# 1) Download all weights (idempotent; skips already-present)
print("=== Ensuring all task weights present ===", flush=True)
for t in ALL_TASKS:
    check_or_download_model_weights(t)
print("=== Weights ready ===", flush=True)

# 2) Stage input as <pid>.nii.gz so patient_id parses cleanly
tmp_in = os.path.join(OUT_DIR, "_input")
os.makedirs(tmp_in, exist_ok=True)
ct_link = os.path.join(tmp_in, "1ABA058.nii.gz")
if os.path.exists(ct_link):
    os.remove(ct_link)
os.symlink(CT_PATH, ct_link)

# 3) Run all tasks
predict(
    files_in=[ct_link],
    folder_out=OUT_DIR,
    model_folder=model_folder,
    task_ids=ALL_TASKS,
    folds="all",
    use_cpu=not torch.cuda.is_available(),
    preprocess_cads=True,
    postprocess_cads=True,
    save_all_combined_seg=True,   # also write the global 167-label merge for comparison
    snapshot=True,                # best-effort overview PNG
    save_separate_targets=False,
    verbose=True,
)
print("=== DONE ===", flush=True)
print("Outputs in:", os.path.join(OUT_DIR, "1ABA058"), flush=True)
