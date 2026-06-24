"""OOD distribution analysis: run several trained MR->CT models on out-of-distribution
datasets (CFB-GBM brain, Gold Atlas pelvis, Learn2Reg abdomen) and dump sCT NIfTIs +
per-(model, subject, sequence) statistics for a downstream HTML report.

Builds on src/evaluate/_family_dispatch.py for model load + inference. Each dataset
loader resamples MR + gtCT onto a common 1.5 mm RAS grid, derives a body mask, and
returns body-masked MR sequences so every model sees the SAME masked input (matching
the flat_masked training regime). Outputs go under GPFS external_inference/ood_distribution/
(gitignored); nothing under the raw dataset trees is modified.

Models (all minmax-MR -> sigmoid/diffusion -> HU):
  unet_centerwise_new   nbn71048  center_wise_split, ep799   (current main U-Net)
  unet_centerwise_old   9xmodnhn  center_wise_split, ep799   (older U-Net)
  unet_fulldata         krdhs2k0  all_train_split,   ep999   (sees brain+pelvis+abd in train)
  mcddpm                a3g28rez  center_wise_split, ep3003  (MC-IDDPM diffusion)

Alignment of MR<->gtCT (whether MAE-vs-gtCT is meaningful):
  cfb_gbm    rigid (T1Gd ref)            -> aligned
  gold_atlas deformable B-spline CT->T2  -> aligned (approx)
  learn2reg  deformable challenge task   -> NOT aligned (MAE not meaningful)

Usage:
  python src/evaluate/ood_distribution_gen.py                 # all datasets, all models
  python src/evaluate/ood_distribution_gen.py --models unet_fulldata mcddpm
  python src/evaluate/ood_distribution_gen.py --datasets cfb_gbm --mcddpm_subjects 2
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from scipy import ndimage as ndi

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from common.utils import unpad  # noqa: E402

from _family_dispatch import (  # noqa: E402
    GPFS,
    MCDDPM_PATCH,
    infer_for_family,
    load_for_family,
    to_hu,
)

OUT_ROOT = os.path.join(GPFS, "external_inference", "ood_distribution")

# name -> (family, ckpt_path, label, train_split)
RUNS = os.path.join(GPFS, "wandb_logs", "runs")
MODELS = {
    "unet_centerwise_new": ("unet", os.path.join(RUNS, "20260611_1957_nbn71048", "checkpoint_last.pt"),
                            "U-Net center-wise (nbn71048)", "center_wise"),
    "unet_centerwise_old": ("unet", os.path.join(RUNS, "20260507_0952_9xmodnhn", "checkpoint_last.pt"),
                            "U-Net center-wise old (9xmodnhn)", "center_wise"),
    "unet_fulldata":       ("unet", os.path.join(RUNS, "20260611_1957_krdhs2k0", "checkpoint_last.pt"),
                            "U-Net full-data (krdhs2k0)", "all_train"),
    "mcddpm":              ("mcddpm", os.path.join(RUNS, "20260515_0836_a3g28rez", "checkpoint_last.pt"),
                            "MC-IDDPM diffusion (a3g28rez)", "center_wise"),
}

SPACING = 1.5


# ─── body mask + resample helpers ───────────────────────────────────────────
def body_from_ct(ct, thr=-300):
    m = ndi.binary_fill_holes(ct > thr)
    lbl, n = ndi.label(m)
    if n > 1:
        m = lbl == (np.argmax(np.bincount(lbl.ravel())[1:]) + 1)
    return m.astype(np.uint8)


def body_from_mr(mr, thr):
    m = ndi.binary_fill_holes(mr > thr)
    lbl, n = ndi.label(m)
    if n > 1:
        m = lbl == (np.argmax(np.bincount(lbl.ravel())[1:]) + 1)
    return m.astype(np.uint8)


# ─── dataset loaders ─────────────────────────────────────────────────────────
def load_cfb(pid_dir, file_prefix, seqs):
    """CFB-GBM brain: rigid-registered, all modalities share the patient grid."""
    src = os.path.join(GPFS, "CFB-GBM", pid_dir, "t0")
    imgs = {"ct": tio.ScalarImage(f"{src}/{file_prefix}_t0_ct.nii.gz")}
    gtv_p = f"{src}/{file_prefix}_t0_gtv.nii.gz"
    if os.path.exists(gtv_p):
        imgs["gtv"] = tio.LabelMap(gtv_p)
    present = []
    for s in seqs:
        p = f"{src}/{file_prefix}_t0_{s}.nii.gz"
        if os.path.exists(p):
            imgs[s] = tio.ScalarImage(p)
            present.append(s)
    subj = tio.ToCanonical()(tio.Resample(SPACING)(tio.Subject(**imgs)))
    ct = subj["ct"].data.numpy()[0].astype(np.float32)
    aff = subj["ct"].affine
    body = body_from_ct(ct)
    mr_seqs = {s: subj[s].data.numpy()[0].astype(np.float32) for s in present}
    return dict(region="brain", aligned=True, affine=aff, body=body,
                gtCT=ct, mr_seqs=mr_seqs)


def load_goldatlas(pid):
    """Gold Atlas pelvis: ct_deformed shares T2 grid; resample T1 + ct_deformed onto a
    common 1.5 mm T2-referenced grid."""
    n = os.path.join(GPFS, "GoldAtlas", "nifti", pid)
    t2 = tio.ScalarImage(f"{n}/mr_T2.nii.gz")
    ref = tio.Resample(SPACING)(tio.Subject(im=t2))["im"]

    def onto_ref(img):
        return tio.Resample(ref)(tio.Subject(im=img))["im"]

    T2 = tio.ToCanonical()(tio.Subject(im=ref))["im"]
    T1 = tio.ToCanonical()(tio.Subject(im=onto_ref(tio.ScalarImage(f"{n}/mr_T1.nii.gz"))))["im"]
    CT = tio.ToCanonical()(tio.Subject(im=onto_ref(tio.ScalarImage(f"{n}/ct_deformed.nii.gz"))))["im"]
    aff = T2.affine
    ct = CT.data.numpy()[0].astype(np.float32)
    body = body_from_ct(ct)
    mr_seqs = {"T1": T1.data.numpy()[0].astype(np.float32),
               "T2": T2.data.numpy()[0].astype(np.float32)}
    return dict(region="pelvis", aligned=True, affine=aff, body=body,
                gtCT=ct, mr_seqs=mr_seqs)


def load_learn2reg(case):
    """Learn2Reg abdomen: MR + CT co-gridded but NOT voxel-aligned (the challenge task).
    MAE-vs-gtCT is not meaningful; we keep separate body masks for MR-input and gtCT."""
    b = os.path.join(GPFS, "Learn2Reg_AbdomenMRCT", "AbdomenMRCT", "imagesTr")
    subj = tio.ToCanonical()(tio.Resample(SPACING)(tio.Subject(
        mr=tio.ScalarImage(f"{b}/AbdomenMRCT_{case}_0000.nii.gz"),
        ct=tio.ScalarImage(f"{b}/AbdomenMRCT_{case}_0001.nii.gz"))))
    mr = subj["mr"].data.numpy()[0].astype(np.float32)
    ct = subj["ct"].data.numpy()[0].astype(np.float32)
    aff = subj["ct"].affine
    body_mr = body_from_mr(mr, 50)
    body_ct = body_from_ct(ct, -300)
    return dict(region="abdomen", aligned=False, affine=aff, body=body_mr,
                gtCT=ct, gt_body=body_ct, mr_seqs={"MR": mr})


DATASET_SUBJECTS = {
    "cfb_gbm": [("001", "1"), ("002", "2"), ("005", "5"), ("010", "10")],
    "gold_atlas": ["1_01_P", "1_02_P", "1_03_P", "2_03_P"],
    "learn2reg": ["0001", "0002", "0003"],
}
CFB_SEQS = ["t1gd"]  # registration reference, present for all patients with CT


def load_subject(dataset, subj):
    if dataset == "cfb_gbm":
        pid_dir, prefix = subj
        return pid_dir, load_cfb(pid_dir, prefix, CFB_SEQS)
    if dataset == "gold_atlas":
        return subj, load_goldatlas(subj)
    if dataset == "learn2reg":
        return subj, load_learn2reg(subj)
    raise ValueError(dataset)


# ─── per-family preprocessing (consistent body-masked input) ─────────────────
def prep_tensor(mr, body, family, device):
    """Body-mask MR, normalize per family, pad. Returns (tensor[1,1,D,H,W], orig_shape)."""
    mr = np.where(body > 0, mr, 0.0).astype(np.float32)
    mn, mx = float(mr.min()), float(mr.max())
    v01 = (mr - mn) / (mx - mn + 1e-8)
    orig = list(mr.shape)
    if family == "mcddpm":
        v = v01 * 2.0 - 1.0  # [-1, 1]
        pad_value = -1.0
        # SpatialPad to at least the patch (128,128,4); sliding window handles the rest.
        tgt = [max(MCDDPM_PATCH[i], orig[i]) for i in range(3)]
    else:  # unet family: minmax [0,1], pad to >=128 and multiple of 16
        v = v01
        pad_value = 0.0
        tgt = [max(128, int(np.ceil(s / 16) * 16)) for s in orig]
    pad = [(0, t - s) for s, t in zip(orig, tgt)]
    v = np.pad(v, pad, mode="constant", constant_values=pad_value)
    x = torch.from_numpy(v)[None, None].to(device)
    return x, orig


def stats(pred_hu, body, gtCT=None, gt_body=None, aligned=False):
    inb = pred_hu[body > 0]
    out = dict(
        body_mean_hu=float(inb.mean()),
        body_std_hu=float(inb.std()),
        bone_frac=float((inb > 300).mean()),
        hu_p01=float(np.percentile(inb, 1)),
        hu_p99=float(np.percentile(inb, 99)),
    )
    if gtCT is not None:
        gb = gt_body if gt_body is not None else body
        gtinb = gtCT[gb > 0]
        out["gt_body_mean_hu"] = float(gtinb.mean())
        out["gt_bone_frac"] = float((gtinb > 300).mean())
    if aligned and gtCT is not None:
        out["mae_hu"] = float(np.abs(pred_hu[body > 0] - gtCT[body > 0]).mean())
    return out


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_SUBJECTS))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--mcddpm_subjects", type=int, default=2,
                    help="cap MC-IDDPM (slow) to the first N subjects per dataset")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip a (model, subject, seq) whose sct NIfTI already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_ROOT, exist_ok=True)
    records = []

    # Load each model once, then sweep all datasets/subjects (avoids reloading).
    for mname in args.models:
        family, ckpt, label, split = MODELS[mname]
        print(f"\n{'#'*70}\nMODEL {mname} ({label}) family={family}\n{'#'*70}")
        family_tag, bundle, cfg, epoch = load_for_family(family, ckpt, device)
        print(f"  loaded epoch={epoch}")
        voxel_sizes = np.array([SPACING, SPACING, SPACING])

        for dataset in args.datasets:
            subjects = DATASET_SUBJECTS[dataset]
            if family == "mcddpm":
                subjects = subjects[: args.mcddpm_subjects]
            for subj in subjects:
                sid, data = load_subject(dataset, subj)
                body = data["body"]
                out_dir = os.path.join(OUT_ROOT, dataset, sid)
                os.makedirs(out_dir, exist_ok=True)
                aff = data["affine"]

                # save shared gtCT/mask/body once (model-independent)
                gt_path = os.path.join(out_dir, "gtCT.nii.gz")
                if not os.path.exists(gt_path) and data["gtCT"] is not None:
                    gb = data.get("gt_body", body)
                    ctm = np.where(gb > 0, np.round(data["gtCT"]), -1024).astype(np.int16)
                    nib.save(nib.Nifti1Image(ctm, aff), gt_path)
                    nib.save(nib.Nifti1Image(body, aff), os.path.join(out_dir, "body.nii.gz"))

                for seq, mr in data["mr_seqs"].items():
                    # save masked MR once for figures
                    mr_path = os.path.join(out_dir, f"mr_{seq}.nii.gz")
                    if not os.path.exists(mr_path):
                        nib.save(nib.Nifti1Image(np.where(body > 0, mr, 0.0).astype(np.float32), aff), mr_path)

                    sct_path = os.path.join(out_dir, f"sct_{mname}_{seq}.nii.gz")
                    if args.skip_existing and os.path.exists(sct_path):
                        print(f"  [{dataset}/{sid}/{seq}] skip (exists)")
                        continue

                    x, orig = prep_tensor(mr, body, family, device)
                    pred = infer_for_family(family, bundle, cfg, x, voxel_sizes, device)
                    pred = unpad(pred.float(), orig)
                    pred_hu = to_hu(family, pred).cpu().numpy().squeeze()
                    pred_hu = np.where(body > 0, pred_hu, -1024.0)

                    nib.save(nib.Nifti1Image(np.round(pred_hu).astype(np.int16), aff), sct_path)

                    row = dict(model=mname, label=label, family=family, split=split,
                               dataset=dataset, subject=sid, region=data["region"],
                               seq=seq, aligned=data["aligned"])
                    row.update(stats(pred_hu, body, data["gtCT"],
                                     data.get("gt_body"), data["aligned"]))
                    records.append(row)
                    mae = row.get("mae_hu")
                    print(f"  [{dataset}/{sid}/{seq}] body-mean {row['body_mean_hu']:.0f} "
                          f"bone-frac {row['bone_frac']:.3f} (gt {row.get('gt_bone_frac', float('nan')):.3f}) "
                          f"MAE {('%.0f' % mae) if mae is not None else 'n/a'}")

        del bundle
        torch.cuda.empty_cache()

    rec_path = os.path.join(OUT_ROOT, "stats.json")
    # merge with any existing records, keyed on (model, dataset, subject, seq)
    merged = {}
    if os.path.exists(rec_path):
        for r in json.load(open(rec_path)):
            merged[(r["model"], r["dataset"], r["subject"], r["seq"])] = r
    for r in records:
        merged[(r["model"], r["dataset"], r["subject"], r["seq"])] = r
    out = sorted(merged.values(), key=lambda r: (r["dataset"], r["subject"], r["seq"], r["model"]))
    with open(rec_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(records)} new records, {len(out)} total -> {rec_path}")


if __name__ == "__main__":
    main()
