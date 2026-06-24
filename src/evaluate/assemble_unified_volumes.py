"""Assemble one unified per-model volume tree for full_eval.

For amix/unet/maisi/mcddpm/cwdm: symlink each model's per-subject validate output
(`raw/<model>/[shard_*/]<subj>/{sample,target}.nii.gz`) into `volumes/<model>/<subj>/`.

For koalAI (no GPU rerun): convert the pre-generated fold-0 HU predictions
(`.../fold_0/validation_revert_norm/<subj>.mha`) and the SynthRAD origin ground truth
(`origin/<ds>/TARGET_IMAGES/<subj>_0000.mha`) to NIfTI via SimpleITK (geometry preserved),
so all six models load identically. koalAI HU is stored as-is (full range, bone >1024 kept).

Usage:
    python src/evaluate/assemble_unified_volumes.py \
        --eval_root /gpfs/.../evaluation_results/full_eval_20260601
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import get_region_key, get_split_subjects  # noqa: E402

NNSYN = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/nnsyn_workspace"
KOALAI_DS_ID = {"abdomen": 960, "thorax": 962, "head_neck": 964, "brain": 966, "pelvis": 968}
KOALAI_TRAINER = "nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres"

# Models whose validate output is a flat per-subject tree.
FLAT_MODELS = ["amix", "unet", "maisi", "amix_bw4", "unet_bw4"]
# Models whose validate output is sharded (raw/<model>/shard_*_of_*/<subj>/...).
SHARDED_MODELS = ["mcddpm", "cwdm"]


def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def symlink_model(model, raw_dir, vol_dir, sharded):
    """Symlink sample/target for every subject found under raw_dir into vol_dir."""
    if sharded:
        subj_dirs = sorted(glob.glob(os.path.join(raw_dir, "shard_*_of_*", "*")))
    else:
        subj_dirs = sorted(glob.glob(os.path.join(raw_dir, "*")))
    found = {}
    for sd in subj_dirs:
        if not os.path.isdir(sd):
            continue
        subj = os.path.basename(sd)
        sample = os.path.join(sd, "sample.nii.gz")
        target = os.path.join(sd, "target.nii.gz")
        if not (os.path.exists(sample) and os.path.exists(target)):
            continue
        out = os.path.join(vol_dir, subj)
        os.makedirs(out, exist_ok=True)
        _link(sample, os.path.join(out, "sample.nii.gz"))
        _link(target, os.path.join(out, "target.nii.gz"))
        found[subj] = sd
    return found


def convert_koalai(vol_dir, subjects):
    import SimpleITK as sitk
    found = {}
    for subj in subjects:
        region = get_region_key(subj)
        ds_id = KOALAI_DS_ID[region]
        ds = f"synthrad2025_task1_mri2ct_{region}"
        pred = os.path.join(NNSYN, "results", f"Dataset{ds_id}_{ds}", KOALAI_TRAINER,
                            "fold_0", "validation_revert_norm", f"{subj}.mha")
        target = os.path.join(NNSYN, "origin", ds, "TARGET_IMAGES", f"{subj}_0000.mha")
        if not (os.path.exists(pred) and os.path.exists(target)):
            continue
        out = os.path.join(vol_dir, subj)
        os.makedirs(out, exist_ok=True)
        dst_sample = os.path.join(out, "sample.nii.gz")
        dst_target = os.path.join(out, "target.nii.gz")
        if not os.path.exists(dst_sample):
            sitk.WriteImage(sitk.ReadImage(pred), dst_sample)
        if not os.path.exists(dst_target):
            sitk.WriteImage(sitk.ReadImage(target), dst_target)
        found[subj] = pred
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--split_file", default="splits/center_wise_split.txt")
    ap.add_argument("--split_name", default="val")
    args = ap.parse_args()

    subjects = get_split_subjects(args.split_file, args.split_name)
    n = len(subjects)
    print(f"[assemble] {n} {args.split_name} subjects from {args.split_file}")

    raw = os.path.join(args.eval_root, "raw")
    vols = os.path.join(args.eval_root, "volumes")

    coverage = {}
    for model in FLAT_MODELS + SHARDED_MODELS:
        raw_dir = os.path.join(raw, model)
        vol_dir = os.path.join(vols, model)
        os.makedirs(vol_dir, exist_ok=True)
        if not os.path.isdir(raw_dir):
            print(f"[assemble] {model}: raw dir missing ({raw_dir}) — skipping")
            coverage[model] = set()
            continue
        found = symlink_model(model, raw_dir, vol_dir, sharded=model in SHARDED_MODELS)
        coverage[model] = set(found)
        print(f"[assemble] {model}: linked {len(found)}/{n}")

    vol_dir = os.path.join(vols, "koalAI")
    os.makedirs(vol_dir, exist_ok=True)
    found = convert_koalai(vol_dir, subjects)
    coverage["koalAI"] = set(found)
    print(f"[assemble] koalAI: converted {len(found)}/{n}")

    print("\n[assemble] coverage summary:")
    ok = True
    subj_set = set(subjects)
    for model, have in coverage.items():
        missing = sorted(subj_set - have)
        flag = "OK" if not missing else f"MISSING {len(missing)}: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        print(f"   {model:8} {len(have):3}/{n}  {flag}")
        if missing:
            ok = False
    print(f"\n[assemble] {'ALL COMPLETE' if ok else 'INCOMPLETE — some models still generating or failed'}")


if __name__ == "__main__":
    main()
