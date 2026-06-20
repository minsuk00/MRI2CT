"""Simple CADS-group error decomposition table for the U-Net sCT.
Reads the per-(subject,label) sums in cads_per_label.csv and rolls them into the
4 groups, reporting voxel share, micro MAE, macro MAE, bias, and error share.

  micro = pool all 207 subjects' voxels, then average  (Sum|err| / Sum n)
  macro = MAE per subject, then average over subjects   (matches synthrad_mae)
"""

import os

import pandas as pd

RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"
BONE = [7, 27, 28, 29, 30]
AIRORG = [9, 13]


def group_of(label):
    if label in BONE:
        return "bone (5 labels)"
    if label in AIRORG:
        return "air-organs (airway+lung)"
    if label == 0:
        return "unlabeled (CADS=0)"
    return "soft (other CADS)"


def main():
    df = pd.read_csv(os.path.join(RUN, "cads_per_label.csv"))
    df["group"] = df.label.map(group_of)

    tot_n = df.n.sum()  # all body voxels over 207 subjects
    tot_abs = df.sabs.sum()  # all body |err| over 207 subjects

    # per-(subject,group) sums -> needed for macro (per-subject MAE then average)
    sg = df.groupby(["subj", "group"]).agg(n=("n", "sum"), sabs=("sabs", "sum")).reset_index()
    sg["mae"] = sg.sabs / sg.n  # this subject's MAE within this group

    order = ["bone (5 labels)", "air-organs (airway+lung)", "soft (other CADS)", "unlabeled (CADS=0)"]
    rows = []
    for grp in order:
        d = df[df.group == grp]
        rows.append(
            {
                "CADS group": grp,
                "% body vox": 100 * d.n.sum() / tot_n,
                "micro MAE": d.sabs.sum() / d.n.sum(),  # pooled
                "macro MAE": sg[sg.group == grp].mae.mean(),  # per-subject then averaged
                "bias": d.serr.sum() / d.n.sum(),
                "% of body error": 100 * d.sabs.sum() / tot_abs,
            }
        )
    out = pd.DataFrame(rows)

    # whole-body reference rows
    body_micro = tot_abs / tot_n
    per_subj = df.groupby("subj").agg(n=("n", "sum"), sabs=("sabs", "sum"))
    body_macro = (per_subj.sabs / per_subj.n).mean()

    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(out.to_string(index=False))
    print(f"\nwhole body:  micro MAE {body_micro:.1f}   macro MAE {body_macro:.1f} (= synthrad_mae)")
    print(f"check: sum %vox {out['% body vox'].sum():.1f}   sum %error {out['% of body error'].sum():.1f}")

    out.to_csv(os.path.join(RUN, "cads_group_table.csv"), index=False)
    print(f"\nwrote {os.path.join(RUN, 'cads_group_table.csv')}")


if __name__ == "__main__":
    main()
