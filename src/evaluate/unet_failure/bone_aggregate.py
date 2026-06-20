"""Aggregate the bone deep-dive: comparative-oracle ranking per metric, universality,
localization-vs-magnitude, loss imbalance, and the MR->CT information limit. Plus
correctness gates reconciling the oracle baseline to the released metrics.

Writes bone_oracle.csv, bone_region.csv, bone_stats.json.
"""
import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import wilcoxon

REPO = "/home/minsukc/MRI2CT"
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
SCEN = ["air", "soft", "bone", "cortical", "skull"]


def main():
    b = pd.read_csv(os.path.join(RUN, "bone_subject.csv"))
    rel = pd.read_csv(os.path.join(EVAL, "metrics/per_subject.csv"))
    rel = rel[rel.model == "unet"]
    m = b.merge(rel[["subj_id", "body_psnr", "synthrad_mae", "body_mae_hu"]], on="subj_id")

    # ---- gates ----
    gates = {
        "g_base_psnr": {"max": float((m.base_psnr - m.body_psnr).abs().max()),
                        "pass": bool((m.base_psnr - m.body_psnr).abs().max() < 1e-2)},
        "g_base_smae": {"max": float((m.base_smae - m.synthrad_mae).abs().max()),
                        "pass": bool((m.base_smae - m.synthrad_mae).abs().max() < 1e-2)},
        "g_base_bmae": {"max": float((m.base_bmae - m.body_mae_hu).abs().max()),
                        "pass": bool((m.base_bmae - m.body_mae_hu).abs().max() < 1e-2)},
        "g_coverage": {"n": int(len(b)), "pass": bool(len(b) == 207)},
    }
    allpass = all(g["pass"] for g in gates.values())
    print("[gates]", "ALL PASS" if allpass else "FAIL")
    for k, gg in gates.items():
        print(f"  {'PASS' if gg['pass'] else 'FAIL'}  {k}  {gg}")

    # ---- comparative oracle deltas (overall + per region) ----
    def oracle_table(df):
        base_psnr, base_bmae, base_smae = df.base_psnr.mean(), df.base_bmae.mean(), df.base_smae.mean()
        rows = {}
        for sc in SCEN:
            rows[sc] = {
                "dPSNR": df[f"{sc}_psnr"].mean() - base_psnr,
                "dBodyMAE": base_bmae - df[f"{sc}_bmae"].mean(),   # reduction (positive = better)
                "dFullHU_MAE": base_smae - df[f"{sc}_smae"].mean(),
            }
        return pd.DataFrame(rows).T

    overall_oracle = oracle_table(b)
    overall_oracle.to_csv(os.path.join(RUN, "bone_oracle.csv"))
    # per-region oracle (ΔPSNR only, compact)
    reg_oracle = {}
    for r in REG:
        t = oracle_table(b[b.region == r])
        reg_oracle[r] = {sc: float(t.loc[sc, "dPSNR"]) for sc in SCEN}
    pd.DataFrame(reg_oracle).T.reindex(REG).to_csv(os.path.join(RUN, "bone_oracle_region_psnr.csv"))

    # rank tissues per metric (which fix wins)
    rank = {met: list(overall_oracle[met].sort_values(ascending=False).index)
            for met in ["dPSNR", "dBodyMAE", "dFullHU_MAE"]}

    # ---- universality ----
    uni = {
        "pct_bone_under_clip": float((b.bias_bone_clip < 0).mean() * 100),
        "pct_bone_under_raw": float((b.bias_bone_raw < 0).mean() * 100),
        "pct_cort_under_clip": float((b.bias_cortical_clip < 0).mean() * 100),
        "pct_cort_under_raw": float((b.bias_cortical_raw < 0).mean() * 100),
        "mean_frac_bone_under": float(b.frac_bone_under.mean() * 100),
        "min_frac_bone_under": float(b.frac_bone_under.min() * 100),
        "n_subjects": int(len(b)),
        "worst_bias_bone_clip": float(b.bias_bone_clip.max()),  # least-negative (closest to 0)
    }

    # ---- loss imbalance ----
    loss = {
        "bone_vox_pct": float(100 * b.n_bone.sum() / b.n_body.sum()),
        "bone_l1_share_pct": float(100 * b.aerr_sum_bone.sum() / b.total_aerr.sum()),  # raw frame
    }
    loss["leverage_ratio"] = loss["bone_l1_share_pct"] / loss["bone_vox_pct"]

    # ---- localization vs magnitude ----
    locmag = {"overall": {k: float(b[k].mean()) for k in
                          ["shape_dice", "missed_frac", "fp_frac", "mae_bone_interior", "mae_bone_boundary"]}}
    for r in REG:
        sr = b[b.region == r]
        locmag[r] = {k: float(sr[k].mean()) for k in
                     ["shape_dice", "missed_frac", "fp_frac", "mae_bone_interior", "mae_bone_boundary"]}

    # ---- MR -> CT information limit (use |rho| = predictive strength) ----
    b["arho_bone"] = b.rho_bone.abs()
    b["arho_soft"] = b.rho_soft.abs()
    b["arho_all"] = b.rho_all.abs()
    valid = b.dropna(subset=["arho_bone", "arho_soft"])
    try:
        w_stat, w_p = wilcoxon(valid.arho_bone, valid.arho_soft)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    # conditional vs marginal CT-HU spread from the pooled MR-rank x CT-HU histogram:
    # how much does knowing MR narrow the CT-HU uncertainty? (low reduction = MR uninformative)
    npz = np.load(os.path.join(RUN, "mrct_hist.npz"))
    hc = (npz["hu_edges"][:-1] + npz["hu_edges"][1:]) / 2

    def spread(key):
        H = npz[key]
        col = H.sum(0)
        mu = (hc * col).sum() / col.sum()
        marg = float(np.sqrt(((hc - mu) ** 2 * col).sum() / col.sum()))
        cs, ws = [], []
        for j in range(H.shape[0]):
            c = H[j]
            if c.sum() < 50:
                continue
            m = (hc * c).sum() / c.sum()
            cs.append(np.sqrt(((hc - m) ** 2 * c).sum() / c.sum()))
            ws.append(c.sum())
        cond = float(np.average(cs, weights=ws))
        return marg, cond, 100 * (1 - cond / marg)

    marg_b, cond_b, red_b = spread("bone")
    marg_s, cond_s, red_s = spread("soft")
    mr = {
        "arho_bone_mean": float(b.arho_bone.mean()), "arho_bone_std": float(b.arho_bone.std()),
        "arho_soft_mean": float(b.arho_soft.mean()), "arho_soft_std": float(b.arho_soft.std()),
        "arho_all_mean": float(b.arho_all.mean()),
        "rho2_bone": float((b.rho_bone ** 2).mean()),   # MR-explained variance fraction in bone
        "rho2_soft": float((b.rho_soft ** 2).mean()),
        "rho_bone_signed_mean": float(b.rho_bone.mean()), "rho_soft_signed_mean": float(b.rho_soft.mean()),
        "wilcoxon_p": float(w_p),
        "ctstd_bone_mean": float(b.ctstd_bone.mean()), "ctstd_soft_mean": float(b.ctstd_soft.mean()),
        "marg_std_bone": marg_b, "cond_std_bone": cond_b, "mr_reduction_bone": red_b,
        "marg_std_soft": marg_s, "cond_std_soft": cond_s, "mr_reduction_soft": red_s,
    }

    # ---- per-voxel severity vs aggregate leverage ----
    sev = {
        "mae_bone": float(b.mae_bone.mean()), "mae_soft": float(b.mae_soft.mean()),
        "bone_per_voxel_ratio": float(b.mae_bone.mean() / b.mae_soft.mean()),
        "pred_bone_max_mean": float(b.pred_bone_max.mean()),
        "gt_bone_mean": float(b.gt_bone_mean.mean()), "pred_bone_mean": float(b.pred_bone_mean.mean()),
        "bias_cortical_clip": float(b.bias_cortical_clip.mean()),
        "bias_cortical_raw": float(b.bias_cortical_raw.mean()),
    }

    # ---- per-tissue table (air/soft/bone): commonness, per-voxel error, bias, error share ----
    ps05 = pd.read_csv(os.path.join(RUN, "per_subject.csv"))  # 05 extraction has air/soft tissue cols
    tv, te = ps05.n_body.sum(), ps05.total_aerr_sum_clip.sum()
    tissue = {}
    for t in ["air", "soft"]:   # air/soft are HU reference classes
        tissue[t] = {
            "vox_pct": float(100 * ps05[f"n_{t}"].sum() / tv),
            "pv_mae": float(ps05[f"mae_{t}_clip"].mean()),
            "bias": float(ps05[f"bias_{t}_clip"].mean()),
            "err_share_pct": float(100 * ps05[f"aerr_sum_{t}_clip"].sum() / te),
        }
    # BONE row = CADS bone labels (clipped frame), NOT an HU threshold
    tissue["bone"] = {
        "vox_pct": float(100 * b.n_bone.sum() / b.n_body.sum()),
        "pv_mae": float(b.mae_bone.mean()),
        "bias": float(b.bias_bone_clip.mean()),
        "err_share_pct": float("nan"),   # CADS bone overlaps soft-HU, so not part of the HU tiling
    }
    region_air = {r: {"air_pct": float(100 * ps05[ps05.region == r].n_air.sum() / ps05[ps05.region == r].n_body.sum()),
                      "air_pv_mae": float(ps05[ps05.region == r].mae_air_clip.mean())} for r in REG}

    # ---- proof that "air" = intra-body gas, not external background (one thorax subject) ----
    DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
    sid = ps05[ps05.region == "thorax"].subj_id.iloc[0]
    gt = np.asarray(nib.as_closest_canonical(nib.load(os.path.join(DATA, sid, "ct.nii"))).dataobj, np.float32)
    bd = np.asarray(nib.as_closest_canonical(nib.load(os.path.join(DATA, sid, "mask.nii"))).dataobj, np.float32) > 0
    a = gt < -300
    air_proof = {
        "subj": sid, "total_vox": int(gt.size),
        "air_outside_body": int((a & ~bd).sum()), "air_inside_body": int((a & bd).sum()),
        "mean_hu_inside_air": float(gt[a & bd].mean()),
        "body_pct_of_volume": float(100 * bd.mean()),
    }

    stats = {"gates": gates, "all_pass": allpass, "rank": rank,
             "oracle_overall": overall_oracle.to_dict("index"),
             "universality": uni, "loss": loss, "locmag": locmag, "mr": mr, "severity": sev,
             "tissue": tissue, "region_air": region_air, "air_proof": air_proof}
    json.dump(stats, open(os.path.join(RUN, "bone_stats.json"), "w"), indent=2)

    print("\n[oracle] overall (ΔPSNR dB / ΔbodyMAE HU / ΔfullHU-MAE HU):")
    print(overall_oracle.round(2).to_string())
    print("\n[rank] PSNR winners:", rank["dPSNR"])
    print("[universality] bone undershoot subjects:", f"{uni['pct_bone_under_clip']:.0f}% (clip),",
          f"{uni['pct_bone_under_raw']:.0f}% (raw); mean {uni['mean_frac_bone_under']:.0f}% of bone voxels/subj")
    print("[loss] bone =", f"{loss['bone_vox_pct']:.1f}% voxels but {loss['bone_l1_share_pct']:.0f}% of L1 error")
    print("[MR] intrinsic CT std bone", f"{mr['ctstd_bone_mean']:.0f} vs soft {mr['ctstd_soft_mean']:.0f} HU;",
          f"MR conditioning reduces bone spread only {mr['mr_reduction_bone']:.0f}% (soft {mr['mr_reduction_soft']:.0f}%);",
          f"rho^2 bone {mr['rho2_bone']:.2f} soft {mr['rho2_soft']:.2f}")
    print("[locmag] dice", round(locmag["overall"]["shape_dice"], 2),
          "missed", round(locmag["overall"]["missed_frac"], 2),
          "interior", round(locmag["overall"]["mae_bone_interior"], 0),
          "boundary", round(locmag["overall"]["mae_bone_boundary"], 0))
    print("[bone_aggregate] wrote bone_oracle.csv, bone_oracle_region_psnr.csv, bone_stats.json")


if __name__ == "__main__":
    main()
