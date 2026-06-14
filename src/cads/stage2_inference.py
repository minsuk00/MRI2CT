"""Stage 2 (GPU): run all 9 CADS tasks on preprocessed CTs.

Reads PREP_DIR/<subj>.nii.gz, writes SEG_PREP_DIR/<subj>/<subj>_part_55X.nii.gz
(in preprocessed 1.5mm space; geometry restored later in stage 3).
Postprocessing (outlier removal + head/H&N 557/558 logic) runs here.

  python src/cads/stage2_inference.py --shard 0/4
Resumable: skips subjects that already have all 9 part files.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

os.environ["CADS_WEIGHTS_PATH"] = C.WEIGHTS_PATH

import torch
from cads.utils.libs import setup_nnunet_env, get_model_weights_dir, check_or_download_model_weights
setup_nnunet_env()
from cads.utils.inference import predict_preprocessed_images


def is_done(subj):
    d = os.path.join(C.SEG_PREP_DIR, subj)
    return all(os.path.exists(os.path.join(d, f"{subj}_part_{t}.nii.gz")) for t in C.ALL_TASKS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="'i/N' strided shard")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(C.SEG_PREP_DIR, exist_ok=True)

    all_prep = sorted(glob.glob(os.path.join(C.PREP_DIR, "*.nii.gz")))
    shard_files = C.take_shard(all_prep, args.shard)
    todo = [f for f in shard_files if not is_done(os.path.basename(f)[:-7])]
    print(f"[stage2] shard {args.shard}: {len(shard_files)} in shard, "
          f"{len(todo)} to run (rest already done)", flush=True)
    if not todo:
        print("[stage2] nothing to do", flush=True)
        return

    model_folder = get_model_weights_dir()
    for t in C.ALL_TASKS:
        check_or_download_model_weights(t)

    predict_preprocessed_images(
        todo, C.SEG_PREP_DIR, model_folder, list(C.ALL_TASKS),
        folds="all",
        use_cpu=not torch.cuda.is_available(),
        postprocess_cads=True,
        num_threads_preprocessing=args.threads,
        nr_threads_saving=args.threads,
        mode="auto",
        verbose=False,
    )
    print("[stage2] done", flush=True)


if __name__ == "__main__":
    main()
