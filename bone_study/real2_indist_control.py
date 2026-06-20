"""Real-data control R2: IN-DISTRIBUTION bone information floor.

Addresses the OOD confound in R1 (real1_infofloor.py): there donors=center-A,
queries=center-B/C, so "U-Net ~ no-MR prior" could be OOD failure, not an
information limit. Here donors AND queries are both center A (HN train) -> no
domain shift. If bone retrieval is still ~prior with a large neighbor spread,
the information limit is intrinsic, not an OOD artifact.

No U-Net column: center-A subjects are train, so no held-out sCT exists. We
compare the model-free kNN-by-MR retrieval, the no-MR prior, and the neighbor
spread, exactly as in R1.
"""
import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import real1_infofloor as R


def run(region="head_neck", seed=0):
    tr = []
    for ln in open(f"{R.REPO}/splits/center_wise_split.txt"):
        p = ln.split()
        if len(p) >= 2 and p[0].lower() == "train" and R.reg(p[1]) == region:
            tr.append(p[1])
    tr = [s for s in tr if os.path.exists(f"{R.DATA}/{s}/ct.nii")]
    rng = np.random.default_rng(seed)
    tr = list(rng.permutation(tr))
    donors, queries = tr[:12], tr[12:18]
    assert len(set(s[3] for s in donors + queries)) == 1, "expected a single center"

    libX, libY, libB = [], [], []
    for s in donors:
        ct, mr, body, seg, _ = R.load_subject(s)
        bc, sc = R.sample_voxels(body, seg, ct, R.NB_DON, R.NS_DON, rng)
        for coords, isb in [(bc, 1), (sc, 0)]:
            if len(coords) == 0:
                continue
            libX.append(R.extract_patches(mr, coords))
            libY.append(ct[coords[:, 0], coords[:, 1], coords[:, 2]])
            libB.append(np.full(len(coords), isb))
    libX = np.concatenate(libX); libY = np.concatenate(libY); libB = np.concatenate(libB)
    scaler = StandardScaler().fit(libX)
    nn = NearestNeighbors(n_neighbors=R.K).fit(scaler.transform(libX))
    prior = {"bone": float(libY[libB == 1].mean()), "soft": float(libY[libB == 0].mean())}

    agg = {t: {"knn": [], "prior": [], "spread": []} for t in ("bone", "soft")}
    for s in queries:
        ct, mr, body, seg, _ = R.load_subject(s)
        bc, sc = R.sample_voxels(body, seg, ct, R.NB_Q, R.NS_Q, rng)
        for coords, name in [(bc, "bone"), (sc, "soft")]:
            if len(coords) == 0:
                continue
            X = scaler.transform(R.extract_patches(mr, coords))
            true = ct[coords[:, 0], coords[:, 1], coords[:, 2]]
            _, ind = nn.kneighbors(X); neigh = libY[ind]
            agg[name]["knn"].append(np.abs(neigh.mean(1) - true))
            agg[name]["prior"].append(np.abs(prior[name] - true))
            agg[name]["spread"].append(neigh.std(1))

    out = {}
    print(f"=== IN-DISTRIBUTION ({region}, center {donors[0][3]} only) ===")
    print(f"{'tissue':>6} | {'kNN-retrieval':>13} {'no-MR prior':>11} {'neighbor-spread':>15}")
    for name in ["bone", "soft"]:
        k = np.concatenate(agg[name]["knn"]).mean()
        p = np.concatenate(agg[name]["prior"]).mean()
        sp = np.concatenate(agg[name]["spread"]).mean()
        out[name] = dict(knn_mae=float(k), prior_mae=float(p), spread=float(sp))
        print(f"{name:>6} | {k:>13.1f} {p:>11.1f} {sp:>15.1f}")
    np.save(f"{R.REPO}/bone_study/real2_indist_{region}.npy",
            {"out": out, "donors": donors, "queries": queries}, allow_pickle=True)
    print("bone gap (prior-kNN)/prior = %.0f%%  -> small gap + large spread = info-limited IN-DISTRIBUTION" %
          (100 * (out["bone"]["prior_mae"] - out["bone"]["knn_mae"]) / out["bone"]["prior_mae"]))


if __name__ == "__main__":
    run("head_neck")
