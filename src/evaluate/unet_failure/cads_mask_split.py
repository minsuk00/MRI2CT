"""Proper external-vs-internal split of the CADS label-0 (Background) air, using a
tight-body estimate (largest connected component of GT>-300, slice-wise hole-fill)
instead of erosion. Over all 207 subjects. Fixes report 10 section 1."""
import os
import glob
import json
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool
from scipy.ndimage import binary_fill_holes, label

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"
RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def tight_body(gt, body):
    t = (gt > -300) & body
    lab, n = label(t)
    if n == 0:
        return t
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    t = lab == big
    for z in range(t.shape[2]):
        t[:, :, z] = binary_fill_holes(t[:, :, z])
    return t


def work(s):
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception:
        return None
    tb = tight_body(gt, body)
    ae = np.abs(sct - gt)
    lab0 = (seg == 0) & body
    ext = lab0 & ~tb            # label-0 outside the patient (loose mask)
    intr = lab0 & tb            # label-0 inside the patient (internal gas/unsegmented)
    return {
        "subj": s, "n_body": int(body.sum()), "sabs_body": float(ae[body].sum()),
        "n_lab0": int(lab0.sum()), "n_ext": int(ext.sum()), "n_int": int(intr.sum()),
        "sabs_ext": float(ae[ext].sum()), "sabs_int": float(ae[intr].sum()),
        "n_ext_air": int((ext & (gt < -300)).sum()), "n_int_air": int((intr & (gt < -300)).sum()),
    }


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    R = [r for r in Pool(8).map(work, subs) if r]
    df = pd.DataFrame(R)
    df.to_csv(os.path.join(RUN, "cads_mask_split.csv"), index=False)
    tb_body = df.n_body.sum()
    tot_abs = df.sabs_body.sum()
    out = {
        "n_subj": len(df),
        "pct_body_lab0": 100 * df.n_lab0.sum() / tb_body,
        "pct_body_lab0_external": 100 * df.n_ext.sum() / tb_body,
        "pct_body_lab0_internal": 100 * df.n_int.sum() / tb_body,
        "external_share_of_lab0": 100 * df.n_ext.sum() / df.n_lab0.sum(),
        "errmass_external_pct": 100 * df.sabs_ext.sum() / tot_abs,
        "errmass_internal_pct": 100 * df.sabs_int.sum() / tot_abs,
        "mae_external": df.sabs_ext.sum() / max(df.n_ext.sum(), 1),
        "mae_internal": df.sabs_int.sum() / max(df.n_int.sum(), 1),
    }
    json.dump(out, open(os.path.join(RUN, "cads_mask_split.json"), "w"), indent=2)
    for k, v in out.items():
        print(f"  {k}: {round(v,2) if isinstance(v,float) else v}")


if __name__ == "__main__":
    main()
