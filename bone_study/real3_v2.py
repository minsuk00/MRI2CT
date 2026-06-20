"""Conclusive real-data retrieval/atlas experiment (v2).

Settles whether a retrieved + deformably-registered donor CT helps bone sCT,
decomposed into INTENSITY (bone HU MAE) and SHARPNESS (edge), with the fixes from
the prove-it review so the conclusion is defensible:

  - prior = MEDIAN of donor bone HU (the L1-optimal no-MR constant), not the mean.
  - oracle arm draws donors from the SAME center as the query (no A->C domain shift)
    and retrieves+aligns on the query CT  -> the strongest possible atlas.
  - pure-registration arm: warp the QUERY's own CT through the realistic transform,
    isolating misalignment error from donor-intensity error.
  - decisive LOQO add-info test run for BOTH realistic and oracle atlases, with a
    bootstrap CI over queries (so "no help" is bounded, not asserted).

Arms per query:
  realistic : donors = center-A train; retrieve by MR; register MR->MR.   (honest)
  oracle    : donors = same-center val; retrieve by CT; register CT->CT.  (upper bound)
  reg_iso   : query CT warped by the realistic top-1 transform.           (pure misalign)
"""
import os
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

import real3_retrieval_core as C

REGION = "head_neck"
N_QUERY = 10
N_DONOR_A = 12
K = 3
DS = 2
rng0 = np.random.default_rng(0)


def grad_mag(v):
    gx, gy, gz = np.gradient(v.astype(np.float32))
    return np.sqrt(gx * gx + gy * gy + gz * gz)


def edge_ratio(pred, ct, bone, match=True):
    band = bone ^ binary_erosion(bone, iterations=1)
    band = binary_dilation(band, iterations=1) & (bone | binary_dilation(bone, iterations=1))
    if band.sum() < 50:
        return np.nan
    p = pred
    if match:
        cm, cs = ct[bone].mean(), ct[bone].std() + 1e-6
        pm, ps = pred[bone].mean(), pred[bone].std() + 1e-6
        p = (pred - pm) / ps * cs + cm
    return float(grad_mag(p)[band].mean() / (grad_mag(ct)[band].mean() + 1e-6))


def bmae(pred, ct, m):
    return float(np.abs(pred - ct)[m].mean())


def main():
    # subjects
    tr, va = [], []
    for ln in open(f"{C.REPO}/splits/center_wise_split.txt"):
        p = ln.split()
        if len(p) < 2 or C.reg(p[1]) != REGION:
            continue
        (tr if p[0].lower() == "train" else va).append(p[1])
    donorsA = [s for s in tr if os.path.exists(f"{C.DATA}/{s}/ct.nii")]
    vaC = [s for s in va if os.path.exists(f"{C.SCT}/{s}/sample.nii.gz")]
    donorsA = list(rng0.choice(donorsA, min(N_DONOR_A, len(donorsA)), replace=False))
    queries = list(rng0.choice(vaC, min(N_QUERY, len(vaC)), replace=False))
    print(f"region={REGION} donorsA={len(donorsA)} centerC_pool={len(vaC)} queries={len(queries)} K={K} DS={DS}")

    def load(s, sct=False):
        d = C.load_raw(s, need_sct=sct)
        d["mrn"] = C.mr_norm(d["mr"], d["body"])
        d["ctc"] = np.clip(d["ct"], -500, 1500)
        d["mr_desc"] = np.nan_to_num(C.descriptor(d["mrn"], d["body"]))
        d["ct_desc"] = np.nan_to_num(C.descriptor(d["ctc"], d["body"]))
        d["bone"] = np.isin(d["seg"], C.BONE_LABELS) & d["body"]
        return d

    DA = {s: load(s) for s in donorsA}
    # prior from center-A donor bone HU (pooled subsample) -> median is the L1-optimal constant
    pool = np.concatenate([(lambda v: rng0.choice(v, min(20000, len(v)), replace=False))(DA[s]["ct"][DA[s]["bone"]])
                           for s in donorsA])
    prior_median, prior_mean = float(np.median(pool)), float(pool.mean())
    print(f"prior bone HU: median={prior_median:.0f}  mean={prior_mean:.0f}")

    Ccache = {}  # lazy center-C loads (donors for oracle)

    def getC(s):
        if s not in Ccache:
            Ccache[s] = load(s)
        return Ccache[s]

    rows, per_query = [], []
    for q in queries:
        Q = load(q, sct=True)
        bone = Q["bone"]
        true_bone = Q["ct"][bone]

        # ---- realistic: center-A donors, MR retrieval, MR registration ----
        simA, _ = C.rank_donors(Q["mr_desc"], [DA[s]["mr_desc"] for s in donorsA])
        warps_r, tx1 = [], None
        for j, di in enumerate(simA[:K]):
            d = DA[donorsA[di]]
            tx = C.register(Q["mrn"], d["mrn"], ds=DS)
            if j == 0:
                tx1 = tx
            warps_r.append(C.warp(tx, Q["ct"], d["ct"]))
        single_r, fused_r = warps_r[0], np.mean(warps_r, axis=0)

        # ---- oracle: same-center (C) donors, CT retrieval, CT registration ----
        cand = [s for s in vaC if s != q]
        cand_desc = [getC(s)["ct_desc"] for s in cand]
        simC, _ = C.rank_donors(Q["ct_desc"], cand_desc)
        warps_o = []
        for di in simC[:K]:
            d = getC(cand[di])
            tx = C.register(Q["ctc"], d["ctc"], ds=DS)
            warps_o.append(C.warp(tx, Q["ct"], d["ct"]))
        single_o, fused_o = warps_o[0], np.mean(warps_o, axis=0)

        # ---- pure registration error: query CT warped by realistic top-1 transform ----
        reg_iso = C.warp(tx1, Q["ct"], Q["ct"])

        preds = {"unet": Q["sct"], "realistic_single": single_r, "realistic_fused": fused_r,
                 "oracle_single": single_o, "oracle_fused": fused_o, "reg_iso": reg_iso}
        rec = {"query": q}
        for name, pr in preds.items():
            rec[f"{name}_bmae"] = bmae(pr, Q["ct"], bone)
            rec[f"{name}_edge"] = edge_ratio(pr, Q["ct"], bone, match=True)
            rec[f"{name}_edgeraw"] = edge_ratio(pr, Q["ct"], bone, match=False)
        rec["prior_median_bmae"] = bmae(np.full_like(Q["ct"], prior_median), Q["ct"], bone)
        rec["prior_mean_bmae"] = bmae(np.full_like(Q["ct"], prior_mean), Q["ct"], bone)
        rows.append(rec)
        print(f"  {q}: unet={rec['unet_bmae']:.0f}(e{rec['unet_edge']:.2f}) "
              f"real_f={rec['realistic_fused_bmae']:.0f}(e{rec['realistic_fused_edge']:.2f}) "
              f"orac_f={rec['oracle_fused_bmae']:.0f}(e{rec['oracle_fused_edge']:.2f}) "
              f"reg_iso={rec['reg_iso_bmae']:.0f} prior_med={rec['prior_median_bmae']:.0f}")

        bi = np.where(bone.ravel())[0]
        sel = rng0.choice(bi, min(20000, len(bi)), replace=False)
        per_query.append(dict(query=q, unet=Q["sct"].ravel()[sel].astype(np.float32),
                              realistic=fused_r.ravel()[sel].astype(np.float32),
                              oracle=fused_o.ravel()[sel].astype(np.float32),
                              true=Q["ct"].ravel()[sel].astype(np.float32)))

    # ---- DECISIVE add-info: leave-one-query-out, with bootstrap CI ----
    def loqo_fold_errs(feats):
        from numpy.linalg import lstsq
        errs = []
        for i in range(len(per_query)):
            tr_i = [j for j in range(len(per_query)) if j != i]
            Xtr = np.concatenate([np.stack([per_query[j][f] for f in feats] + [np.ones_like(per_query[j]["true"])], 1) for j in tr_i])
            ytr = np.concatenate([per_query[j]["true"] for j in tr_i])
            w, *_ = lstsq(Xtr, ytr, rcond=None)
            Xte = np.stack([per_query[i][f] for f in feats] + [np.ones_like(per_query[i]["true"])], 1)
            errs.append(float(np.abs(Xte @ w - per_query[i]["true"]).mean()))
        return np.array(errs)

    e_u = loqo_fold_errs(["unet"])
    e_ur = loqo_fold_errs(["unet", "realistic"])
    e_uo = loqo_fold_errs(["unet", "oracle"])

    def boot_ci(delta):
        bs = [np.mean(rng0.choice(delta, len(delta), replace=True)) for _ in range(2000)]
        return float(np.mean(delta)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    print("\n=== DECISIVE add-info (LOQO bone MAE, mean over queries) ===")
    print(f"  fit[unet]            : {e_u.mean():.1f}")
    print(f"  fit[unet,realistic]  : {e_ur.mean():.1f}")
    print(f"  fit[unet,oracle]     : {e_uo.mean():.1f}")
    for nm, e in [("realistic", e_ur), ("oracle", e_uo)]:
        m, lo, hi = boot_ci(e_u - e)
        print(f"  {nm:10} atlas adds: {m:+.1f} HU  (95% CI [{lo:+.1f}, {hi:+.1f}])  "
              f"{'HELPS' if lo > 0 else 'no significant help'}")

    np.save(f"{C.REPO}/bone_study/real3_v2_{REGION}.npy",
            {"rows": rows, "queries": queries, "donorsA": donorsA, "K": K,
             "prior_median": prior_median, "prior_mean": prior_mean,
             "loqo": {"unet": e_u, "unet_realistic": e_ur, "unet_oracle": e_uo}}, allow_pickle=True)
    print("saved real3_v2_%s.npy" % REGION)


if __name__ == "__main__":
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "8")
    main()
