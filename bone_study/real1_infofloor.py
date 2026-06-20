"""Real-data confirmation R1: the bone INFORMATION FLOOR, measured model-free,
which also directly tests the retrieval idea.

Logic: any MR embedding is a deterministic function of the MR, so NO MR-based
predictor (U-Net, retrieval, diffusion, ...) can carry more bone information than
the MR itself. We estimate that ceiling non-parametrically:

  For bone voxels, find the k nearest neighbors in LOCAL-MR-PATCH space across a
  donor library (= retrieval by MR appearance). Two readouts:
    kNN-mean prediction error  : best achievable error of a pure MR-retrieval model
    within-neighborhood CT std : the IRREDUCIBLE spread of true CT-HU among
                                 voxels whose MR looks identical (the info floor).

  Compare against the trained U-Net's sCT error at the SAME voxels. If retrieval
  and U-Net both land near the (large) neighborhood spread, bone is info-limited
  and retrieval-by-MR cannot beat the U-Net. Soft tissue is the positive control:
  there the floor should be small and both predictors accurate.

Donors = train HN subjects; queries = val HN subjects (the U-Net eval set).
"""
import os
import numpy as np
import nibabel as nib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = "/home/minsukc/MRI2CT"
DATA = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SCT = f"{REPO}/evaluation_results/full_eval_20260617/volumes/unet"
BONE_LABELS = [7, 27, 28, 29, 30]
P = 3                      # half patch -> 7x7x7 = 343-dim MR feature
K = 16                     # neighbors
NB_DON, NS_DON = 4000, 4000   # bone/soft samples per donor
NB_Q, NS_Q = 2500, 2500       # bone/soft query samples per subject
SEED = 0


def reg(s):
    return {"AB": "abdomen", "TH": "thorax", "HN": "head_neck"}.get(s[1:3].upper(), "other")


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def load_subject(s, need_sct=False):
    ct = canon(f"{DATA}/{s}/ct.nii")
    mr = canon(f"{DATA}/{s}/moved_mr.nii")
    body = canon(f"{DATA}/{s}/mask.nii") > 0
    seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    # robust per-subject MR normalization to [0,1] on body voxels (pipeline-style p0.5..p99.5)
    bv = mr[body]
    lo, hi = np.percentile(bv, 0.5), np.percentile(bv, 99.5)
    mr = np.clip((mr - lo) / (hi - lo + 1e-6), 0, 1)
    sct = canon(f"{SCT}/{s}/sample.nii.gz") if need_sct else None
    return ct, mr, body, seg, sct


def extract_patches(mr, coords):
    """coords: (M,3) int voxel indices. Returns (M, (2P+1)^3) MR patches (zero-padded)."""
    pad = np.pad(mr, P, mode="constant")
    M = coords.shape[0]
    out = np.empty((M, (2 * P + 1) ** 3), dtype=np.float32)
    for i, (x, y, z) in enumerate(coords):
        out[i] = pad[x:x + 2 * P + 1, y:y + 2 * P + 1, z:z + 2 * P + 1].ravel()
    return out


def sample_voxels(body, seg, ct, n_bone, n_soft, rng, margin=P):
    """Return bone and soft voxel coords, avoiding the volume border by `margin`."""
    bone = np.isin(seg, BONE_LABELS) & body
    soft = body & ~bone & (ct > -200) & (ct < 300)   # soft-tissue HU band
    # keep away from borders so patches are valid
    edge = np.ones_like(body)
    edge[:margin] = edge[-margin:] = False
    edge[:, :margin] = edge[:, -margin:] = False
    edge[:, :, :margin] = edge[:, :, -margin:] = False
    bone &= edge; soft &= edge

    def pick(mask, n):
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return idx
        sel = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
        return idx[sel]

    return pick(bone, n_bone), pick(soft, n_soft)


def main():
    import sys
    region = sys.argv[1] if len(sys.argv) > 1 else "head_neck"

    def split_subjects():
        tr, va = [], []
        for ln in open(f"{REPO}/splits/center_wise_split.txt"):
            p = ln.split()
            if len(p) < 2:
                continue
            if reg(p[1]) != region:
                continue
            (tr if p[0].lower() == "train" else va).append(p[1])
        return tr, va

    tr, va = split_subjects()
    tr = [s for s in tr if os.path.exists(f"{DATA}/{s}/ct.nii")]
    va = [s for s in va if os.path.exists(f"{SCT}/{s}/sample.nii.gz")]
    rng = np.random.default_rng(SEED)
    donors = list(rng.choice(tr, size=min(12, len(tr)), replace=False))
    queries = list(rng.choice(va, size=min(6, len(va)), replace=False))
    print(f"region={region}  donors={len(donors)}  queries={len(queries)}")
    print(f"donors: {donors}")
    print(f"queries: {queries}")

    # ---- build donor library ----
    libX, libY, libB = [], [], []
    for s in donors:
        ct, mr, body, seg, _ = load_subject(s)
        bc, sc = sample_voxels(body, seg, ct, NB_DON, NS_DON, rng)
        for coords, isb in [(bc, 1), (sc, 0)]:
            if len(coords) == 0:
                continue
            libX.append(extract_patches(mr, coords))
            libY.append(ct[coords[:, 0], coords[:, 1], coords[:, 2]])
            libB.append(np.full(len(coords), isb))
    libX = np.concatenate(libX); libY = np.concatenate(libY); libB = np.concatenate(libB)
    print(f"library: {libX.shape[0]} patches ({int(libB.sum())} bone)")

    scaler = StandardScaler().fit(libX)        # standardize features -> tighter MR neighborhoods
    libXs = scaler.transform(libX)
    nn = NearestNeighbors(n_neighbors=K, algorithm="auto").fit(libXs)
    prior_mean = {"bone": float(libY[libB == 1].mean()), "soft": float(libY[libB == 0].mean())}

    # ---- query ----
    agg = {"bone": dict(knn=[], unet=[], prior=[], spread=[], true=[]),
           "soft": dict(knn=[], unet=[], prior=[], spread=[], true=[])}
    for s in queries:
        ct, mr, body, seg, sct = load_subject(s, need_sct=True)
        bc, sc = sample_voxels(body, seg, ct, NB_Q, NS_Q, rng)
        for coords, name in [(bc, "bone"), (sc, "soft")]:
            if len(coords) == 0:
                continue
            X = scaler.transform(extract_patches(mr, coords))
            true = ct[coords[:, 0], coords[:, 1], coords[:, 2]]
            dist, ind = nn.kneighbors(X)
            neigh = libY[ind]                       # (M, K) neighbor CT-HU
            knn_pred = neigh.mean(axis=1)
            spread = neigh.std(axis=1)              # within-neighborhood irreducible spread
            unet = sct[coords[:, 0], coords[:, 1], coords[:, 2]]
            agg[name]["knn"].append(np.abs(knn_pred - true))
            agg[name]["unet"].append(np.abs(unet - true))
            agg[name]["prior"].append(np.abs(prior_mean[name] - true))
            agg[name]["spread"].append(spread)
            agg[name]["true"].append(true)

    print(f"\n{'tissue':>6} | {'kNN-retrieval':>13} {'U-Net sCT':>10} {'global-prior':>12} "
          f"{'neighbor-spread':>15} {'n':>7}")
    out = {}
    for name in ["bone", "soft"]:
        knn = np.concatenate(agg[name]["knn"]); unet = np.concatenate(agg[name]["unet"])
        prior = np.concatenate(agg[name]["prior"]); spread = np.concatenate(agg[name]["spread"])
        out[name] = dict(knn_mae=float(knn.mean()), unet_mae=float(unet.mean()),
                         prior_mae=float(prior.mean()), spread=float(spread.mean()), n=int(len(knn)))
        print(f"{name:>6} | {knn.mean():>13.1f} {unet.mean():>10.1f} {prior.mean():>12.1f} "
              f"{spread.mean():>15.1f} {len(knn):>7}")

    np.save(f"{REPO}/bone_study/real1_{region}.npy",
            {"out": out, "donors": donors, "queries": queries, "K": K, "P": P}, allow_pickle=True)
    print("\nVerdict logic:")
    print(" - bone: if kNN-retrieval MAE ~ U-Net MAE ~ neighbor-spread (all large), bone is")
    print("   INFORMATION-LIMITED -> retrieval-by-MR cannot beat the U-Net.")
    print(" - soft: kNN & U-Net MAE both << prior -> MR-context predicts soft well (control).")


if __name__ == "__main__":
    main()
