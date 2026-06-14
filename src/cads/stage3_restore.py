"""Stage 3 (CPU): restore each part segmentation to the original CT geometry.

Reads SEG_PREP_DIR/<subj>/<subj>_part_55X.nii.gz + META_DIR/<subj>_metadata.pkl,
writes SEG_DIR/<subj>/<subj>_part_55X.nii.gz on the original (LPS, native-shape)
1.5mm grid -- i.e. voxel-aligned with the dataset's ct.nii / mask.nii.

  python src/cads/stage3_restore.py --shard 0/4
Idempotent: skips part files already restored.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C

from cads.dataset_utils.preprocessing import restore_seg_in_orig_format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="'i/N' strided shard")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(C.SEG_DIR, exist_ok=True)

    subjects = sorted(
        d for d in os.listdir(C.SEG_PREP_DIR)
        if os.path.isdir(os.path.join(C.SEG_PREP_DIR, d))
    )
    subjects = C.take_shard(subjects, args.shard)
    print(f"[stage3] shard {args.shard}: {len(subjects)} subjects", flush=True)

    t0 = time.time()
    n_restored = n_skip = n_fail = 0
    for i, subj in enumerate(subjects):
        meta_path = os.path.join(C.META_DIR, f"{subj}_metadata.pkl")
        if not os.path.exists(meta_path):
            n_fail += 1
            print(f"[stage3] FAILED {subj}: missing metadata {meta_path}", flush=True)
            continue
        with open(meta_path, "rb") as f:
            metadata_orig = pickle.load(f)

        in_dir = os.path.join(C.SEG_PREP_DIR, subj)
        out_dir = os.path.join(C.SEG_DIR, subj)
        os.makedirs(out_dir, exist_ok=True)
        for seg_file in sorted(os.listdir(in_dir)):
            if not seg_file.endswith(".nii.gz"):  # skip *_ERROR.log etc.
                continue
            out_path = os.path.join(out_dir, seg_file)
            if os.path.exists(out_path):
                n_skip += 1
                continue
            try:
                restore_seg_in_orig_format(
                    os.path.join(in_dir, seg_file), out_path, metadata_orig,
                    num_threads_preprocessing=args.threads,
                )
                n_restored += 1
            except Exception as e:
                n_fail += 1
                print(f"[stage3] FAILED {subj}/{seg_file}: {e}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"[stage3] {i + 1}/{len(subjects)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"[stage3] done: {n_restored} restored, {n_skip} skipped, {n_fail} failed "
          f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
