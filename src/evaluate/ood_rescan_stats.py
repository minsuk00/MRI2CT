"""Recompute stats.json for the OOD distribution analysis from whatever sct_*.nii.gz
volumes exist on disk (robust to partial / interrupted generation runs).

Walks external_inference/ood_distribution/<dataset>/<subject>/sct_<model>_<seq>.nii.gz,
re-derives body-mean HU / bone fraction / MAE-vs-gtCT (where aligned), and writes
stats.json. Pure post-hoc measurement; no GPU, no model loading.
"""
import json
import os

import nibabel as nib
import numpy as np

GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
OUT_ROOT = os.path.join(GPFS, "external_inference", "ood_distribution")

MODEL_META = {  # key -> (label, split)
    "unet_centerwise_new": ("U-Net center-wise (nbn71048)", "center_wise"),
    "unet_centerwise_old": ("U-Net center-wise old (9xmodnhn)", "center_wise"),
    "unet_fulldata": ("U-Net full-data (krdhs2k0)", "all_train"),
    "mcddpm": ("MC-IDDPM diffusion (a3g28rez)", "center_wise"),
    "koalai": ("koalAI / nnsyn (SynthRAD'25 winner)", "fold0_centerwise"),
}
DATASET_META = {  # dataset -> (region, aligned)
    "cfb_gbm": ("brain", True),
    "gold_atlas": ("pelvis", True),
    "learn2reg": ("abdomen", False),
}


def parse_model_seq(fname):
    rest = fname[len("sct_"):-len(".nii.gz")]
    for mk in sorted(MODEL_META, key=len, reverse=True):
        if rest.startswith(mk + "_"):
            return mk, rest[len(mk) + 1:]
    return None, None


def load(p):
    return np.asarray(nib.load(p).dataobj).astype(np.float32)


def main():
    records = []
    for dataset, (region, aligned) in DATASET_META.items():
        droot = os.path.join(OUT_ROOT, dataset)
        if not os.path.isdir(droot):
            continue
        for subj in sorted(os.listdir(droot)):
            sdir = os.path.join(droot, subj)
            body_p = os.path.join(sdir, "body.nii.gz")
            if not os.path.isdir(sdir) or not os.path.exists(body_p):
                continue
            body = load(body_p) > 0
            gt_p = os.path.join(sdir, "gtCT.nii.gz")
            gt = load(gt_p) if os.path.exists(gt_p) else None
            for f in sorted(os.listdir(sdir)):
                if not (f.startswith("sct_") and f.endswith(".nii.gz")):
                    continue
                mk, seq = parse_model_seq(f)
                if mk is None:
                    continue
                sct = load(os.path.join(sdir, f))
                inb = sct[body]
                label, split = MODEL_META[mk]
                row = dict(model=mk, label=label, split=split, dataset=dataset,
                           subject=subj, region=region, seq=seq, aligned=aligned,
                           body_mean_hu=float(inb.mean()), body_std_hu=float(inb.std()),
                           bone_frac=float((inb > 300).mean()),
                           hu_p01=float(np.percentile(inb, 1)),
                           hu_p99=float(np.percentile(inb, 99)))
                if gt is not None:
                    gtb = gt[gt > -1000]
                    row["gt_body_mean_hu"] = float(gtb.mean())
                    row["gt_bone_frac"] = float((gtb > 300).mean())
                    if aligned:
                        row["mae_hu"] = float(np.abs(sct[body] - gt[body]).mean())
                records.append(row)
    records.sort(key=lambda r: (r["dataset"], r["subject"], r["seq"], r["model"]))
    out = os.path.join(OUT_ROOT, "stats.json")
    with open(out, "w") as f:
        json.dump(records, f, indent=2)
    counts = {}
    for r in records:
        counts[r["model"]] = counts.get(r["model"], 0) + 1
    print(f"wrote {len(records)} records -> {out}")
    print("per-model volume counts:", counts)


if __name__ == "__main__":
    main()
