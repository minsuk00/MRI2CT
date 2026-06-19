"""Model-free test of whether MR APPEARANCE (not just one voxel's intensity, but the
full local 5x5x5 patch) can predict CT HU — for bone vs soft tissue.

Rationale: a CNN can only use information that's in the MR. If we take the FULL local MR
patch around each voxel and ask "do voxels with near-identical MR patches have similar CT
HU?", we are directly estimating how predictable CT HU is from MR appearance, with NO
neural net and NO L1 loss involved. kNN-in-patch-space approximates the best any model
could do. If bone HU is unpredictable from the patch (high kNN error) while soft is
predictable (low error), that is direct, model-free evidence that the MR does not contain
the bone-density information — context included.

Cross-subject: each query's neighbors are drawn from OTHER subjects (no trivial self-match).
MR is z-scored within each subject's body (MR is only per-volume meaningful). Patches are
PCA-reduced to tame the curse of dimensionality (same reduction for bone and soft, so the
comparison is fair).

Writes knn_patch.json + knn_patch.npz (for the figure).
"""
import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
PER_SUBJECT = os.path.join(REPO, "evaluation_results/full_eval_20260617/metrics/per_subject.csv")
R = 2                       # patch radius -> 5x5x5 = 125 voxels
PER_CLASS_PER_SUBJ = 1500   # sampled voxels per class per subject
RNG = np.random.RandomState(0)


def reg(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def sample_subject(args):
    sid, sidx = args
    try:
        gt = canon(f"{DATA}/{sid}/ct.nii")
        mr = canon(f"{DATA}/{sid}/moved_mr.nii")
        body = canon(f"{DATA}/{sid}/mask.nii") > 0
    except Exception:
        return None
    # z-score MR within body (MR is per-volume normalized; this makes patches comparable)
    mb = mr[body]
    mu, sd = mb.mean(), mb.std() + 1e-6
    mrz = (mr - mu) / sd
    mrz = np.pad(mrz, R, mode="edge")  # pad so patch extraction is safe
    bone = body & (gt > 200)
    soft = body & (gt >= -300) & (gt <= 200)

    out = {}
    for name, mask in [("bone", bone), ("soft", soft)]:
        idx = np.argwhere(mask)
        if len(idx) < 200:
            out[name] = None
            continue
        sel = idx[RNG.choice(len(idx), min(PER_CLASS_PER_SUBJ, len(idx)), replace=False)]
        patches = np.empty((len(sel), (2 * R + 1) ** 3), np.float32)
        hu = np.empty(len(sel), np.float32)
        for i, (a, b, c) in enumerate(sel):
            patches[i] = mrz[a:a + 2 * R + 1, b:b + 2 * R + 1, c:c + 2 * R + 1].ravel()
            hu[i] = gt[a, b, c]
        out[name] = (patches, hu, np.full(len(sel), sidx, np.int32))
    return out


def knn_eval(X, hu, subj, k=16):
    """Cross-subject kNN: predict each voxel's HU from the mean HU of its k nearest
    MR-patch neighbors that belong to OTHER subjects. Returns MAE and R^2."""
    nn = NearestNeighbors(n_neighbors=k + 40).fit(X)
    _, nbr = nn.kneighbors(X)
    pred = np.empty(len(X), np.float32)
    for i in range(len(X)):
        cand = nbr[i][subj[nbr[i]] != subj[i]][:k]  # drop same-subject neighbors
        pred[i] = hu[cand].mean() if len(cand) else hu[nbr[i][1:k + 1]].mean()
    mae = float(np.abs(pred - hu).mean())
    ss_res = float(((pred - hu) ** 2).sum())
    ss_tot = float(((hu - hu.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    return mae, r2, pred


def main():
    ps = pd.read_csv(PER_SUBJECT)
    ps = ps[ps.model == "unet"].copy()
    ps["region"] = ps.subj_id.map(reg)
    subs = ps.groupby("region").head(8).subj_id.tolist()   # region-balanced ~40
    jobs = [(s, i) for i, s in enumerate(subs)]
    print(f"[knn] sampling {len(subs)} subjects", flush=True)
    res = [r for r in Pool(8).map(sample_subject, jobs) if r]

    result = {}
    store = {}
    for cls in ["bone", "soft"]:
        parts = [r[cls] for r in res if r.get(cls) is not None]
        X = np.concatenate([p[0] for p in parts])
        hu = np.concatenate([p[1] for p in parts])
        sj = np.concatenate([p[2] for p in parts])
        # subsample to keep kNN tractable
        if len(X) > 25000:
            s = RNG.choice(len(X), 25000, replace=False)
            X, hu, sj = X[s], hu[s], sj[s]
        Xp = PCA(n_components=25, random_state=0).fit_transform(X)        # full patch (context)
        Xc = X[:, [X.shape[1] // 2]]                                      # center voxel intensity only
        mae_patch, r2_patch, pred_patch = knn_eval(Xp, hu, sj)
        mae_int, r2_int, _ = knn_eval(Xc, hu, sj)
        base = float(np.abs(hu - np.median(hu)).mean())                   # predict constant (no MR)
        result[cls] = {"n": int(len(X)), "hu_std": float(hu.std()),
                       "knn_patch_mae": mae_patch, "knn_patch_r2": r2_patch,
                       "knn_intensity_mae": mae_int, "knn_intensity_r2": r2_int,
                       "constant_mae": base}
        store[f"{cls}_hu"] = hu
        store[f"{cls}_pred"] = pred_patch
        print(f"  {cls}: patch kNN MAE {mae_patch:.0f} (R2 {r2_patch:.2f}) | "
              f"intensity-only MAE {mae_int:.0f} | constant {base:.0f} | HU std {hu.std():.0f}", flush=True)

    json.dump(result, open(os.path.join(RUN, "knn_patch.json"), "w"), indent=2)
    np.savez(os.path.join(RUN, "knn_patch.npz"), **store)
    print("[knn] wrote knn_patch.json + knn_patch.npz")


if __name__ == "__main__":
    main()
