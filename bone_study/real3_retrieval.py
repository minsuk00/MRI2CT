"""Decisive real-data retrieval/atlas experiment.

Builds the ACTUAL atlas-based-sCT pipeline the user proposes and measures what a
retrieved + deformably-registered donor CT delivers on bone, split into intensity
(HU MAE) and sharpness (edge gradient), against the U-Net and a no-MR prior.

Arms (give retrieval progressively its BEST case):
  realistic : retrieve donor by MR similarity, register donor MR -> query MR.
  oracle    : retrieve donor by CT similarity, register donor CT -> query CT.
              (best possible donor + best possible alignment; unreachable in
              practice because it uses the query CT we are trying to predict.)
For each arm: single best donor (sharp, misaligned) and multi-atlas fusion of
top-k (lower variance, blurred). Decisive hybrid test: does fusing [U-Net, atlas]
beat the U-Net alone on held-out bone MAE -> does the atlas add ANY information?

Edge sharpness is magnitude-matched (scale prediction to the query's bone-HU
mean/std before taking the gradient ratio) so sharpness is not credited to
amplitude. Real CT -> 1.0 by construction.
"""
import os
import sys
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

import real3_retrieval_core as C

REGION = sys.argv[1] if len(sys.argv) > 1 else "head_neck"
K = 3          # donors fused per query
N_DONORS = 12  # donor pool size
N_QUERY = 6    # query subjects
DS = 2         # registration downsample factor


def split():
    tr, va = [], []
    for ln in open(f"{C.REPO}/splits/center_wise_split.txt"):
        p = ln.split()
        if len(p) < 2 or C.reg(p[1]) != REGION:
            continue
        (tr if p[0].lower() == "train" else va).append(p[1])
    tr = [s for s in tr if os.path.exists(f"{C.DATA}/{s}/ct.nii")]
    va = [s for s in va if os.path.exists(f"{C.SCT}/{s}/sample.nii.gz")]
    return tr, va


def grad_mag(v):
    gx, gy, gz = np.gradient(v.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def edge_sharpness(pred, ct, bone):
    """Magnitude-matched gradient ratio on the bone boundary. Real CT -> 1.0."""
    band = bone ^ binary_erosion(bone, iterations=1)
    band = binary_dilation(band, iterations=1) & (bone | binary_dilation(bone, iterations=1))
    if band.sum() < 50:
        return np.nan
    cm, cs = ct[bone].mean(), ct[bone].std() + 1e-6
    pm, ps = pred[bone].mean(), pred[bone].std() + 1e-6
    pred_m = (pred - pm) / ps * cs + cm          # match bone-HU mean/std
    return float(grad_mag(pred_m)[band].mean() / (grad_mag(ct)[band].mean() + 1e-6))


def metrics(pred, Q, bone):
    body = Q["body"]
    return dict(bone_mae=float(np.abs(pred - Q["ct"])[bone].mean()),
                body_mae=float(np.abs(pred - Q["ct"])[body].mean()),
                edge=edge_sharpness(pred, Q["ct"], bone))


def main():
    tr, va = split()
    rng = np.random.default_rng(0)
    donors = list(rng.choice(tr, min(N_DONORS, len(tr)), replace=False))
    queries = list(rng.choice(va, min(N_QUERY, len(va)), replace=False))
    print(f"region={REGION} donors={len(donors)} queries={len(queries)} K={K} DS={DS}")
    print("donors", donors); print("queries", queries)

    # preload donors + descriptors
    D = {}
    for s in donors:
        d = C.load_raw(s)
        d["mrn"] = C.mr_norm(d["mr"], d["body"])
        d["mr_desc"] = C.descriptor(d["mrn"], d["body"])
        d["ct_desc"] = C.descriptor(np.clip(d["ct"], -500, 1500), d["body"])
        D[s] = d

    # no-MR prior = mean over donors of each donor's mean bone HU (the population bone level)
    prior_bone_hu = float(np.mean([D[s]["ct"][np.isin(D[s]["seg"], C.BONE_LABELS) & D[s]["body"]].mean()
                                   for s in donors]))
    rows = []
    per_query = []   # list of dicts {unet, atlas, true} arrays over bone voxels (realistic arm)

    for q in queries:
        Q = C.load_raw(q, need_sct=True)
        Qmrn = C.mr_norm(Q["mr"], Q["body"])
        q_mr_desc = C.descriptor(Qmrn, Q["body"])
        q_ct_desc = C.descriptor(np.clip(Q["ct"], -500, 1500), Q["body"])
        bone = np.isin(Q["seg"], C.BONE_LABELS) & Q["body"]

        # rankings
        idx_mr, _ = C.rank_donors(q_mr_desc, [D[s]["mr_desc"] for s in donors])
        idx_ct, _ = C.rank_donors(q_ct_desc, [D[s]["ct_desc"] for s in donors])

        for arm, idx, use_ct in [("realistic", idx_mr, False), ("oracle", idx_ct, True)]:
            warped = []
            for di in idx[:K]:
                d = D[donors[di]]
                if use_ct:   # oracle: align on CT
                    w = C.register_warp(np.clip(Q["ct"], -500, 1500), np.clip(d["ct"], -500, 1500), d["ct"], ds=DS)
                else:        # realistic: align on MR
                    w = C.register_warp(Qmrn, d["mrn"], d["ct"], ds=DS)
                warped.append(w)
            single = warped[0]
            fused = np.mean(warped, axis=0)
            m_s = metrics(single, Q, bone); m_f = metrics(fused, Q, bone)
            rows.append(dict(query=q, arm=arm, kind="single", **m_s))
            rows.append(dict(query=q, arm=arm, kind="fused", **m_f))
            print(f"  {q} {arm:9} single bone_mae={m_s['bone_mae']:.0f} edge={m_s['edge']:.2f} | "
                  f"fused bone_mae={m_f['bone_mae']:.0f} edge={m_f['edge']:.2f}")
            if arm == "realistic":
                # subsample bone voxels to keep the saved arrays/regression light
                bi = np.where(bone.ravel())[0]
                sel = np.random.default_rng(1).choice(bi, size=min(20000, len(bi)), replace=False)
                per_query.append(dict(query=q,
                                      unet=Q["sct"].ravel()[sel].astype(np.float32),
                                      atlas=fused.ravel()[sel].astype(np.float32),
                                      true=Q["ct"].ravel()[sel].astype(np.float32)))

        # reference rows once per query
        rows.append(dict(query=q, arm="unet", kind="model", **metrics(Q["sct"], Q, bone)))
        prior_pred = np.full_like(Q["ct"], prior_bone_hu)
        rows.append(dict(query=q, arm="prior", kind="const",
                         bone_mae=float(np.abs(prior_pred - Q["ct"])[bone].mean()),
                         body_mae=np.nan, edge=np.nan))
        print(f"  {q} {'unet':9} bone_mae={rows[-2]['bone_mae']:.0f} edge={rows[-2]['edge']:.2f} | "
              f"prior bone_mae={rows[-1]['bone_mae']:.0f}")

    # ---- DECISIVE hybrid add-info test: does the atlas add info beyond the U-Net? ----
    # Leave-one-query-out linear fit; compare held-out bone MAE of [unet] vs [unet, atlas].
    def loqo_mae(feats):
        from numpy.linalg import lstsq
        errs = []
        for i in range(len(per_query)):
            tr_i = [j for j in range(len(per_query)) if j != i]
            Xtr = np.concatenate([np.stack([per_query[j][f] for f in feats] + [np.ones_like(per_query[j]["true"])], 1) for j in tr_i])
            ytr = np.concatenate([per_query[j]["true"] for j in tr_i])
            w, *_ = lstsq(Xtr, ytr, rcond=None)
            Xte = np.stack([per_query[i][f] for f in feats] + [np.ones_like(per_query[i]["true"])], 1)
            errs.append(np.abs(Xte @ w - per_query[i]["true"]).mean())
        return float(np.mean(errs))

    raw_unet = float(np.mean([np.abs(p["unet"] - p["true"]).mean() for p in per_query]))
    mae_u = loqo_mae(["unet"])
    mae_ua = loqo_mae(["unet", "atlas"])
    mae_a = loqo_mae(["atlas"])
    print("\n=== DECISIVE hybrid add-info (leave-one-query-out bone MAE) ===")
    print(f"  raw U-Net sCT           : {raw_unet:.1f}")
    print(f"  fit[unet]               : {mae_u:.1f}")
    print(f"  fit[atlas]              : {mae_a:.1f}")
    print(f"  fit[unet, atlas]        : {mae_ua:.1f}")
    print(f"  atlas adds: {mae_u - mae_ua:+.1f} HU (negative-of-this <=0 => no help)")

    np.save(f"{C.REPO}/bone_study/real3_{REGION}.npy",
            {"rows": rows, "donors": donors, "queries": queries, "K": K, "prior_bone_hu": float(prior_bone_hu),
             "hybrid": {"raw_unet": raw_unet, "fit_unet": mae_u, "fit_atlas": mae_a, "fit_unet_atlas": mae_ua},
             "per_query": per_query}, allow_pickle=True)
    print("saved real3_%s.npy" % REGION)


if __name__ == "__main__":
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8")
    main()
