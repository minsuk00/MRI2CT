"""koalAI / nnsyn OOD inference for the distribution analysis.

For each OOD dataset, stage the already-saved masked MR + body mask (written by
ood_distribution_gen.py at 1.5 mm RAS) into the nnsyn predict input/mask layout
as .mha, run `nnsyn_predict` with the matching region's synthesis model
(fold 0 = center-wise / OOD design, MAP-loss trainer), then convert the
--revert_norm HU output back to sct_koalai_<seq>.nii.gz on the same grid and
merge per-volume stats into stats.json.

Region -> dataset id (nnsyn): brain=966, pelvis=968, abdomen=960.

Must run in the `koalai` env with the nnsyn env vars exported:
    source sbatch/koalai_env.sh   (or rely on this script's os.environ defaults)
    micromamba run -n koalai python src/evaluate/ood_koalai_gen.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
OUT_ROOT = os.path.join(GPFS, "external_inference", "ood_distribution")
WS = os.path.join(GPFS, "nnsyn_workspace")
STAGE = os.path.join(OUT_ROOT, "_koalai_stage")

os.environ.setdefault("nnUNet_raw", os.path.join(WS, "raw"))
os.environ.setdefault("nnUNet_preprocessed", os.path.join(WS, "preprocessed"))
os.environ.setdefault("nnUNet_results", os.path.join(WS, "results"))
os.environ.setdefault("NNSYN_ORIGIN_ROOT", os.path.join(WS, "origin"))

# dataset -> (region, nnsyn dataset id, aligned)
DATASETS = {
    "cfb_gbm": ("brain", 966, True),
    "gold_atlas": ("pelvis", 968, True),
    "learn2reg": ("abdomen", 960, False),
}


def safe_name(dataset, subj, seq):
    return f"{dataset}_{subj}_{seq}".replace("-", "").replace(".", "")


def stage_region(dataset):
    """Write {name}_0000.mha (MR) + {name}.mha (mask) for every saved volume in this dataset.
    Returns list of (name, subj, seq, ref_nii_path)."""
    in_dir = os.path.join(STAGE, dataset, "input")
    mask_dir = os.path.join(STAGE, dataset, "mask")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    items = []
    droot = os.path.join(OUT_ROOT, dataset)
    if not os.path.isdir(droot):
        return items
    for subj in sorted(os.listdir(droot)):
        sdir = os.path.join(droot, subj)
        body_p = os.path.join(sdir, "body.nii.gz")
        if not os.path.isdir(sdir) or not os.path.exists(body_p):
            continue
        mask_img = sitk.ReadImage(body_p)
        for f in sorted(os.listdir(sdir)):
            if f.startswith("mr_") and f.endswith(".nii.gz"):
                seq = f[len("mr_"):-len(".nii.gz")]
                name = safe_name(dataset, subj, seq)
                mr_img = sitk.ReadImage(os.path.join(sdir, f))
                sitk.WriteImage(mr_img, os.path.join(in_dir, f"{name}_0000.mha"))
                sitk.WriteImage(sitk.Cast(mask_img, sitk.sitkUInt8),
                                os.path.join(mask_dir, f"{name}.mha"))
                items.append((name, subj, seq, os.path.join(sdir, f)))
    return items


def run_predict(dataset, region, ds_id):
    in_dir = os.path.join(STAGE, dataset, "input")
    mask_dir = os.path.join(STAGE, dataset, "mask")
    out_dir = os.path.join(STAGE, dataset, "out")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "nnsyn_predict", "-d", str(ds_id), "-i", in_dir, "-o", out_dir, "-m", mask_dir,
        "-c", "3d_fullres", "-p", "nnUNetResEncUNetLPlans",
        "-tr", "nnUNetTrainer_nnsyn_loss_map", "-f", "0", "--revert_norm",
    ]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_dir + "_revert_norm"


def collect(dataset, region, items, hu_dir):
    """Convert HU .mha outputs -> sct_koalai_<seq>.nii.gz next to the other sCTs; return stat rows."""
    aligned = DATASETS[dataset][2]
    rows = []
    for name, subj, seq, ref_nii in items:
        hu_p = os.path.join(hu_dir, f"{name}.mha")
        if not os.path.exists(hu_p):
            print(f"    [MISS] {name}")
            continue
        sct_img = sitk.ReadImage(hu_p)
        sdir = os.path.join(OUT_ROOT, dataset, subj)
        out_nii = os.path.join(sdir, f"sct_koalai_{seq}.nii.gz")
        # write back onto the reference (masked MR) geometry to guarantee grid match
        ref = sitk.ReadImage(ref_nii)
        sct_img.CopyInformation(ref) if sct_img.GetSize() == ref.GetSize() else None
        sitk.WriteImage(sct_img, out_nii)

        # stats via nibabel-consistent arrays (load body + gt the same way they were saved)
        import nibabel as nib
        body = np.asarray(nib.load(os.path.join(sdir, "body.nii.gz")).dataobj) > 0
        sct = np.asarray(nib.load(out_nii).dataobj).astype(np.float32)
        sct = np.where(body, sct, -1024.0)
        nib.save(nib.Nifti1Image(np.round(sct).astype(np.int16),
                                 nib.load(out_nii).affine), out_nii)
        inb = sct[body]
        row = dict(model="koalai", label="koalAI / nnsyn (SynthRAD'25 winner)", family="nnsyn",
                   split="fold0_centerwise", dataset=dataset, subject=subj, region=region,
                   seq=seq, aligned=aligned,
                   body_mean_hu=float(inb.mean()), body_std_hu=float(inb.std()),
                   bone_frac=float((inb > 300).mean()),
                   hu_p01=float(np.percentile(inb, 1)), hu_p99=float(np.percentile(inb, 99)))
        gt_p = os.path.join(sdir, "gtCT.nii.gz")
        if os.path.exists(gt_p):
            gt = np.asarray(nib.load(gt_p).dataobj).astype(np.float32)
            gtb = gt[gt > -1000]
            row["gt_body_mean_hu"] = float(gtb.mean())
            row["gt_bone_frac"] = float((gtb > 300).mean())
            if aligned:
                row["mae_hu"] = float(np.abs(sct[body] - gt[body]).mean())
        rows.append(row)
        mae = row.get("mae_hu")
        print(f"    [{dataset}/{subj}/{seq}] body-mean {row['body_mean_hu']:.0f} "
              f"bone-frac {row['bone_frac']:.3f} MAE {('%.0f'%mae) if mae is not None else 'n/a'}")
    return rows


def main():
    only = sys.argv[1:] or list(DATASETS)
    all_rows = []
    for dataset in only:
        region, ds_id, _ = DATASETS[dataset]
        print(f"\n### koalAI {dataset} (region={region}, dataset={ds_id})")
        items = stage_region(dataset)
        print(f"  staged {len(items)} volumes")
        if not items:
            continue
        hu_dir = run_predict(dataset, region, ds_id)
        all_rows += collect(dataset, region, items, hu_dir)

    # merge into stats.json
    rec_path = os.path.join(OUT_ROOT, "stats.json")
    merged = {}
    if os.path.exists(rec_path):
        for r in json.load(open(rec_path)):
            merged[(r["model"], r["dataset"], r["subject"], r["seq"])] = r
    for r in all_rows:
        merged[(r["model"], r["dataset"], r["subject"], r["seq"])] = r
    out = sorted(merged.values(), key=lambda r: (r["dataset"], r["subject"], r["seq"], r["model"]))
    with open(rec_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nmerged {len(all_rows)} koalAI rows, {len(out)} total -> {rec_path}")


if __name__ == "__main__":
    main()
