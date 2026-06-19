"""Aggregate the per-subject / per-label extraction into region rollups, the oracle
counterfactual, and 8 correctness gates. Writes agg_stats.json + table CSVs that
build_figures.py and report.py consume.

All per-region/overall means are subject-mean (each subject = 1 sample) unless a
quantity is an error-MASS share, which is voxel-weighted (sum of per-voxel abs error
across subjects). Per-label means are taken only over subjects that contain the label
and are always reported with prevalence (n subjects).
"""
import os
import json
import numpy as np
import pandas as pd

REPO = "/home/minsukc/MRI2CT"
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
REGION_N = {"abdomen": 52, "brain": 60, "head_neck": 32, "pelvis": 30, "thorax": 33}


def main():
    s = pd.read_csv(os.path.join(RUN, "per_subject.csv"))
    lab = pd.read_csv(os.path.join(RUN, "per_label.csv"))
    rel = pd.read_csv(os.path.join(EVAL, "metrics/per_subject.csv"))
    rel = rel[rel.model == "unet"].copy()
    by_region = pd.read_csv(os.path.join(EVAL, "metrics/by_region.csv"))
    bru = by_region[by_region.model == "unet"].set_index("region")

    gates = {}

    # ---- GATE 1: Frame R vs synthrad_mae ----
    m = s.merge(rel[["subj_id", "synthrad_mae", "body_mae_hu", "body_psnr", "region"]],
                on="subj_id", suffixes=("", "_rel"))
    d_raw = (m.mae_raw - m.synthrad_mae).abs()
    gates["g1_mae_raw_vs_synthrad"] = {"max": float(d_raw.max()), "pass": bool(d_raw.max() < 1e-2)}

    # ---- GATE 2: Frame C body MAE vs released body_mae_hu ----
    body_recompute = m.mae_clip * m.n_body / m.n_total
    d_body = (body_recompute - m.body_mae_hu).abs()
    gates["g2_body_mae_clip_vs_released"] = {"max": float(d_body.max()), "pass": bool(d_body.max() < 1e-2)}

    # ---- GATE 3: per-region mae_raw vs by_region synthrad_mae_mean ----
    reg_mae_raw = s.groupby("region").mae_raw.mean()
    d_reg = {r: float(abs(reg_mae_raw[r] - bru.loc[r, "synthrad_mae_mean"])) for r in REG}
    gates["g3_region_reconcile"] = {"deltas": d_reg, "max": max(d_reg.values()),
                                    "pass": bool(max(d_reg.values()) < 0.05)}

    # ---- GATE 4: region key + counts ----
    counts = s.region.value_counts().to_dict()
    region_mismatch = int((m.region != m.region_rel).sum())
    gates["g4_region_counts"] = {"counts": {r: int(counts.get(r, 0)) for r in REG},
                                 "expected": REGION_N, "mismatch_vs_released": region_mismatch,
                                 "total": int(len(s)),
                                 "pass": bool(len(s) == 207 and region_mismatch == 0 and
                                              all(counts.get(r, 0) == REGION_N[r] for r in REG))}

    # ---- GATE 5: coverage ----
    gates["g5_coverage"] = {"n_subjects": int(len(s)), "pass": bool(len(s) == 207)}

    # ---- GATE 6: mass conservation (voxel counts exact; mass relative due to float32 sums) ----
    tis_sum_c = s.aerr_sum_air_clip + s.aerr_sum_soft_clip + s.aerr_sum_bone_clip
    rel_mass = ((tis_sum_c - s.total_aerr_sum_clip).abs() / s.total_aerr_sum_clip).max()
    cons_n = (s.n_air + s.n_soft + s.n_bone - s.n_body).abs().max()
    cons_bone = (s.n_midbone + s.n_cortical - s.n_bone).abs().max()
    gates["g6_mass_conservation"] = {"max_rel_mass_resid": float(rel_mass), "max_n_resid": int(cons_n),
                                     "max_bone_n_resid": int(cons_bone),
                                     "pass": bool(rel_mass < 1e-4 and cons_n == 0 and cons_bone == 0)}

    # ---- GATE 7: HU reference proof (raw GT reaches dense cortical) ----
    gt_bone_max = float(s.gt_bone_max.max())
    gates["g7_raw_gt_reference"] = {"gt_bone_max": gt_bone_max, "pass": bool(gt_bone_max > 1500)}

    # ---- GATE 8: ceiling precondition (pred << gt in bone) ----
    pbm = float(s.pred_bone_max.mean())
    gbm = float(s.gt_bone_max.mean())
    gates["g8_ceiling"] = {"pred_bone_max_mean": pbm, "gt_bone_max_mean": gbm,
                           "pass": bool(pbm < 0.6 * gbm)}

    all_pass = all(g["pass"] for g in gates.values())
    print("[gates] " + ("ALL PASS" if all_pass else "SOME FAILED"))
    for k, g in gates.items():
        print(f"  {'PASS' if g['pass'] else 'FAIL'}  {k}")

    # ================= rollups =================
    def reg_mean(col):
        return {r: float(s[s.region == r][col].mean()) for r in REG}

    def reg_mass_share(tissue):
        out = {}
        for r in REG:
            sr = s[s.region == r]
            out[r] = float(sr[f"aerr_sum_{tissue}_clip"].sum() / sr.total_aerr_sum_clip.sum())
        return out

    def reg_vox_share(tissue):
        out = {}
        for r in REG:
            sr = s[s.region == r]
            out[r] = float(sr[f"n_{tissue}"].sum() / sr.n_body.sum())
        return out

    region_tissue = pd.DataFrame({
        "MAE air": reg_mean("mae_air_clip"), "MAE soft": reg_mean("mae_soft_clip"),
        "MAE bone": reg_mean("mae_bone_clip"), "MAE all": reg_mean("mae_clip"),
        "bias bone (C)": reg_mean("bias_bone_clip"), "bias cort (C)": reg_mean("bias_cortical_clip"),
        "bias cort (R)": reg_mean("bias_cortical_raw"),
        "body PSNR": {r: float(bru.loc[r, "body_psnr_mean"]) for r in REG},
    }).reindex(REG)
    region_tissue["bone:soft"] = region_tissue["MAE bone"] / region_tissue["MAE soft"]
    region_tissue.to_csv(os.path.join(RUN, "region_tissue.csv"))

    region_mass = pd.DataFrame({
        "bone vox %": {r: 100 * v for r, v in reg_vox_share("bone").items()},
        "bone err-mass %": {r: 100 * v for r, v in reg_mass_share("bone").items()},
        "soft err-mass %": {r: 100 * v for r, v in reg_mass_share("soft").items()},
        "air err-mass %": {r: 100 * v for r, v in reg_mass_share("air").items()},
    }).reindex(REG)
    region_mass["mass/vox ratio"] = region_mass["bone err-mass %"] / region_mass["bone vox %"]
    region_mass.to_csv(os.path.join(RUN, "region_mass.csv"))

    # bone HU diagnostics per region (both frames)
    region_bonehu = pd.DataFrame({
        "GT bone mean": reg_mean("gt_bone_mean"), "pred bone mean": reg_mean("pred_bone_mean"),
        "pred bone p95": reg_mean("pred_bone_p95"), "pred bone max": reg_mean("pred_bone_max"),
        "GT bone max": reg_mean("gt_bone_max"),
        "cort MAE (C)": reg_mean("mae_cortical_clip"), "cort MAE (R)": reg_mean("mae_cortical_raw"),
        "cort bias (C)": reg_mean("bias_cortical_clip"), "cort bias (R)": reg_mean("bias_cortical_raw"),
        "near-ceil %": {r: 100 * v for r, v in reg_mean("pred_near_ceiling_frac").items()},
        "GT %>1024": {r: 100 * v for r, v in reg_mean("gt_cortical_frac").items()},
    }).reindex(REG)
    region_bonehu.to_csv(os.path.join(RUN, "region_bonehu.csv"))

    # ---- per-label aggregation (with prevalence) ----
    g = lab.groupby("name")
    per_label = pd.DataFrame({
        "is_bone": g.is_bone.first(), "cads_region": g.cads_region.first(),
        "MAE (C)": g.mae_clip.mean(), "MAE (R)": g.mae_raw.mean(),
        "bias (C)": g.bias_clip.mean(), "bias (R)": g.bias_raw.mean(),
        "GT HU": g.gt_mean.mean(), "pred HU": g.pred_mean.mean(),
        "GT %>1024": g.gt_frac_gt1024.mean() * 100,
        "n subj": g.subj_id.nunique(), "mean vox": g.n.mean(),
        "MAE std": g.mae_clip.std(),
    }).sort_values("MAE (C)", ascending=False)
    per_label.to_csv(os.path.join(RUN, "per_label_agg.csv"))

    # ---- bone vs non-bone (label-level), voxel-weighted, per region + overall ----
    def vw_mae(df):
        return float((df.mae_clip * df.n).sum() / df.n.sum()) if len(df) else np.nan
    bvn_rows = {}
    for r in REG + ["OVERALL"]:
        sub = lab if r == "OVERALL" else lab[lab.region == r]
        bvn_rows[r] = {"bone MAE": vw_mae(sub[sub.is_bone]),
                       "organ/soft MAE": vw_mae(sub[~sub.is_bone])}
    bvn = pd.DataFrame(bvn_rows).T.reindex(REG + ["OVERALL"])
    bvn["ratio"] = bvn["bone MAE"] / bvn["organ/soft MAE"]
    bvn.to_csv(os.path.join(RUN, "bone_vs_nonbone.csv"))

    # ---- per-region worst-5 labels ----
    worst = {}
    for r in REG:
        sub = lab[lab.region == r]
        gg = sub.groupby("name").agg(MAE=("mae_clip", "mean"), bias=("bias_clip", "mean"),
                                     is_bone=("is_bone", "first"), n_subj=("subj_id", "nunique"))
        gg = gg.sort_values("MAE", ascending=False).head(5)
        worst[r] = gg.reset_index().to_dict("records")
    json.dump(worst, open(os.path.join(RUN, "region_worst.json"), "w"), indent=2)

    # ---- oracle counterfactual: body MAE if bone predicted perfectly ----
    s["body_mae_C"] = s.mae_clip
    s["body_mae_C_bonefix"] = s.bone_zeroed_aerr_sum_clip / s.n_body
    s["body_mae_R"] = s.mae_raw
    s["body_mae_R_bonefix"] = s.bone_zeroed_aerr_sum_raw / s.n_body
    oracle = pd.DataFrame({
        "body MAE (C)": {**reg_mean("body_mae_C"), "OVERALL": float(s.body_mae_C.mean())},
        "bone-fixed (C)": {**{r: float(s[s.region == r].body_mae_C_bonefix.mean()) for r in REG},
                           "OVERALL": float(s.body_mae_C_bonefix.mean())},
        "body MAE (R)": {**reg_mean("body_mae_R"), "OVERALL": float(s.body_mae_R.mean())},
        "bone-fixed (R)": {**{r: float(s[s.region == r].body_mae_R_bonefix.mean()) for r in REG},
                           "OVERALL": float(s.body_mae_R_bonefix.mean())},
    }).reindex(REG + ["OVERALL"])
    oracle["drop % (C)"] = 100 * (oracle["body MAE (C)"] - oracle["bone-fixed (C)"]) / oracle["body MAE (C)"]
    oracle["drop % (R)"] = 100 * (oracle["body MAE (R)"] - oracle["bone-fixed (R)"]) / oracle["body MAE (R)"]
    oracle.to_csv(os.path.join(RUN, "oracle.csv"))

    # ---- reconciliation table ----
    recon = pd.DataFrame({
        "recomp MAE (R)": reg_mae_raw.reindex(REG),
        "released synthrad_mae": {r: float(bru.loc[r, "synthrad_mae_mean"]) for r in REG},
        "recomp body MAE (C)": {r: float((s[s.region == r].mae_clip * s[s.region == r].n_body /
                                          s[s.region == r].n_total).mean()) for r in REG},
        "released body_mae_hu": {r: float(bru.loc[r, "body_mae_hu_mean"]) for r in REG},
    }).reindex(REG)
    recon["Δ R"] = (recon["recomp MAE (R)"] - recon["released synthrad_mae"]).abs()
    recon["Δ C"] = (recon["recomp body MAE (C)"] - recon["released body_mae_hu"]).abs()
    recon.to_csv(os.path.join(RUN, "recon.csv"))

    # ---- scalar stats for report prose ----
    stats = {
        "gates": gates, "all_pass": all_pass,
        "overall": {
            "body_mae_C": float(s.mae_clip.mean()), "body_mae_R": float(s.mae_raw.mean()),
            "mae_bone_C": float(s.mae_bone_clip.mean()), "mae_soft_C": float(s.mae_soft_clip.mean()),
            "mae_air_C": float(s.mae_air_clip.mean()),
            "bias_bone_C": float(s.bias_bone_clip.mean()), "bias_bone_R": float(s.bias_bone_raw.mean()),
            "bias_cortical_C": float(s.bias_cortical_clip.mean()),
            "bias_cortical_R": float(s.bias_cortical_raw.mean()),
            "bias_midbone_C": float(s.bias_midbone_clip.mean()),
            "mae_cortical_C": float(s.mae_cortical_clip.mean()),
            "mae_cortical_R": float(s.mae_cortical_raw.mean()),
            "gt_bone_mean": float(s.gt_bone_mean.mean()), "pred_bone_mean": float(s.pred_bone_mean.mean()),
            "pred_bone_p95": float(s.pred_bone_p95.mean()), "pred_bone_max": float(s.pred_bone_max.mean()),
            "gt_bone_max": float(s.gt_bone_max.mean()),
            "near_ceiling_pct": float(s.pred_near_ceiling_frac.mean() * 100),
            "bone_vox_pct": float(100 * s.n_bone.sum() / s.n_body.sum()),
            "bone_mass_pct": float(100 * s.aerr_sum_bone_clip.sum() / s.total_aerr_sum_clip.sum()),
            "oracle_drop_C": float(oracle.loc["OVERALL", "drop % (C)"]),
            "oracle_drop_R": float(oracle.loc["OVERALL", "drop % (R)"]),
            "bone_vs_organ_ratio": float(bvn.loc["OVERALL", "ratio"]),
            "bone_mae_label": float(bvn.loc["OVERALL", "bone MAE"]),
            "organ_mae_label": float(bvn.loc["OVERALL", "organ/soft MAE"]),
        },
        "region_tissue": region_tissue.to_dict("index"),
        "worst_label_overall": per_label.index[0],
        "worst_bone_label": per_label[per_label.is_bone].index[0],
    }
    # structural vs in-range cortical error split (overall)
    cr, cc = stats["overall"]["mae_cortical_R"], stats["overall"]["mae_cortical_C"]
    stats["overall"]["cortical_structural_frac"] = float((cr - cc) / cr) if cr else np.nan
    json.dump(stats, open(os.path.join(RUN, "agg_stats.json"), "w"), indent=2)
    print("[aggregate] wrote region_tissue/region_mass/region_bonehu/per_label_agg/"
          "bone_vs_nonbone/oracle/recon CSVs, region_worst.json, agg_stats.json")


if __name__ == "__main__":
    main()
