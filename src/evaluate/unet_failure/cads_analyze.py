"""Aggregate cads_extract outputs into the additive, micro-consistent tables for
report 10, and print a summary to read before writing claims."""
import os
import json
import numpy as np
import pandas as pd

RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"
BONE = [7, 27, 28, 29, 30]
AIRORG = [9, 13]


def main():
    pl = pd.read_csv(os.path.join(RUN, "cads_per_label.csv"))
    su = pd.read_csv(os.path.join(RUN, "cads_subject.csv"))

    # ---- micro per-label (sums -> additive) ----
    g = pl.groupby(["label", "name", "is_bone"]).agg(
        n=("n", "sum"), sabs=("sabs", "sum"), serr=("serr", "sum"),
        sgt=("sgt", "sum"), spred=("spred", "sum"), n_gt1024=("n_gt1024", "sum"),
        n_subj=("subj", "nunique")).reset_index()
    g["mae"] = g.sabs / g.n
    g["bias"] = g.serr / g.n
    g["gt_hu"] = g.sgt / g.n
    g["pred_hu"] = g.spred / g.n
    tot_abs = g.sabs.sum()
    tot_n = g.n.sum()
    g["errmass_pct"] = 100 * g.sabs / tot_abs
    g["voxshare_pct"] = 100 * g.n / tot_n
    g = g.sort_values("errmass_pct", ascending=False)
    g.to_csv(os.path.join(RUN, "cads_label_micro.csv"), index=False)

    # ---- gate: micro reconstruction of body MAE ----
    body_mae_micro = tot_abs / su.n_body.sum()           # Σ|err| over labeled body / Σ body vox
    recon = tot_abs / tot_n                               # Σ|err| / Σ labeled vox (== body since labels cover body)
    body_mae_macro = su.body_mae.mean()
    gate_ok = abs(body_mae_micro - recon) < 1e-6 and abs(tot_n - su.n_body.sum()) < 1

    # ---- group rollup ----
    def grp(mask, name):
        sub = g[mask]
        return {"group": name, "voxshare_pct": float(sub.voxshare_pct.sum()),
                "mae": float(sub.sabs.sum() / sub.n.sum()), "errmass_pct": float(sub.errmass_pct.sum()),
                "bias": float(sub.serr.sum() / sub.n.sum())}
    groups = pd.DataFrame([
        grp(g.label.isin(BONE), "bone (5 labels)"),
        grp(g.label.isin(AIRORG), "air-organs (airway+lung)"),
        grp((~g.label.isin(BONE + AIRORG)) & (g.label != 0), "soft (other CADS)"),
        grp(g.label == 0, "unlabeled (CADS=0)"),
    ])
    groups.to_csv(os.path.join(RUN, "cads_groups.csv"), index=False)

    # ---- mask audit ----
    audit = {
        "n_subj": int(len(su)),
        "body_mae_micro": float(body_mae_micro), "body_mae_macro": float(body_mae_macro),
        "pct_body_unlabeled": float(100 * su.n_lab0.sum() / su.n_body.sum()),
        "pct_body_airHU": float(100 * su.n_air.sum() / su.n_body.sum()),
        "pct_body_lab0_air": float(100 * su.n_lab0_air.sum() / su.n_body.sum()),
        "lab0_air_rim_share": float(su.n_lab0_air_rim.sum() / max(su.n_lab0_air.sum(), 1)),
        "errmass_lab0_pct": float(100 * su.sabs_lab0.sum() / tot_abs),
        "errmass_lab0_air_rim_pct": float(100 * su.sabs_lab0_air_rim.sum() / tot_abs),
        "errmass_lab0_air_int_pct": float(100 * su.sabs_lab0_air_int.sum() / tot_abs),
        "gate_micro_reconstructs_body_mae": bool(gate_ok),
    }
    json.dump(audit, open(os.path.join(RUN, "cads_audit.json"), "w"), indent=2)

    print(f"== body MAE: micro {body_mae_micro:.1f}  macro {body_mae_macro:.1f}  (recon {recon:.1f}, gate {'OK' if gate_ok else 'FAIL'}) ==")
    print("\n== group decomposition (micro, additive) ==")
    print(groups.round(1).to_string(index=False))
    print(f"   sum errmass = {groups.errmass_pct.sum():.1f}%  sum voxshare = {groups.voxshare_pct.sum():.1f}%")
    print("\n== mask audit ==")
    for k, v in audit.items():
        print(f"   {k}: {v}")
    print("\n== top-10 labels by error-mass ==")
    print(g.head(10)[["name", "voxshare_pct", "mae", "bias", "gt_hu", "pred_hu", "errmass_pct", "n_subj"]].round(1).to_string(index=False))
    print("\n== bone labels ==")
    print(g[g.is_bone][["name", "voxshare_pct", "mae", "bias", "gt_hu", "pred_hu", "errmass_pct", "n_gt1024"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
