"""Stage 1 (CPU): reorient->RAS + resample to 1.5mm, write preprocessed CT + metadata.

Reads <DATA_DIR>/<subj>/ct.nii (uncompressed, all identically named -> we pass an
explicit patient_id, which the upstream run_01 CLI cannot do).
Writes PREP_DIR/<subj>.nii.gz and META_DIR/<subj>_metadata.pkl.

  python src/cads/stage1_preprocess.py --shard 0/4
Idempotent: skips subjects whose preprocessed image already exists.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

from cads.dataset_utils.preprocessing import preprocess_nifti_and_save_to_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="'i/N' strided shard")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(C.PREP_DIR, exist_ok=True)
    os.makedirs(C.META_DIR, exist_ok=True)

    subjects = C.take_shard(C.list_subjects(), args.shard)
    print(f"[stage1] shard {args.shard}: {len(subjects)} subjects", flush=True)

    t0 = time.time()
    n_done = n_skip = n_fail = 0
    for i, subj in enumerate(subjects):
        out_img = os.path.join(C.PREP_DIR, f"{subj}.nii.gz")
        out_meta = os.path.join(C.META_DIR, f"{subj}_metadata.pkl")
        if os.path.exists(out_img) and os.path.exists(out_meta):
            n_skip += 1
            continue
        ct = os.path.join(C.DATA_DIR, subj, C.CT_NAME)
        try:
            preprocess_nifti_and_save_to_dir(
                ct, C.PREP_DIR, C.META_DIR, subj,
                spacing=1.5, num_threads_preprocessing=args.threads,
            )
            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"[stage1] FAILED {subj}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"[stage1] {i + 1}/{len(subjects)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"[stage1] done: {n_done} new, {n_skip} skipped, {n_fail} failed "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
