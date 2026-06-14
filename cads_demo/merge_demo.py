"""Demo: priority-painting merge of CADS 9-part segs -> single labelmap, 1 subj/region.

Reads the read-only seg/ tree; writes ONLY to cads_demo/. Does not alter any data.

Two variants (--variant):
  grouped : 35-class final_id grouping from the CSV (default).
  all     : keep every source structure as its own class (full CADS granularity).
Both use the SAME priority painting (paint low priority first, high priority wins
on overlap). 'all' simply assigns a unique id per (src_model, src_index) instead
of the grouped final_id.

Outputs:
  cads_demo/merged[_all]/<subj>_merged[_all].nii.gz
  cads_demo/merge_demo[_all].png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
import pandas as pd

PROJ = "/home/minsukc/MRI2CT"
DATA = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SEG = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/cads/seg"
CSV = os.path.join(PROJ, "cads_demo", "cads_labelmap (1).csv")
TASKS = [551, 552, 553, 554, 555, 556, 557, 558, 559]

# NO region gating. We investigated denying "out-of-FOV" tasks/structures per region,
# but every alarming case (head on 1THA293, hip on 1THB165, neck-558 on thorax)
# turned out to be REAL anatomy: the SynthRAD region label does not bound the scan
# FOV, and CADS correctly segments whatever is in the field of view. Genuine errors
# are tiny isolated specks (~few k voxels on ~30/843), negligible for our use.
# So: merge ALL 9 tasks, every subject, by priority painting. See
# _html/cads_merge_report.html for the full investigation.


def region_of(subj):
    """Region code from subject id prefix -- used only for the brain CT display window."""
    s = subj[1:]
    for r in ("AB", "HN", "TH", "P", "B"):  # AB before B; HN/TH before P
        if s.startswith(r):
            return r
    return "?"


# region -> example subject
SUBJECTS = [
    ("Abdomen", "1ABA005"),
    ("Brain", "1BA001"),
    ("Head & Neck", "1HNA001"),
    ("Pelvis", "1PA001"),
    ("Thorax", "1THA001"),
]


def load_map(variant):
    """Return df with columns: src_model, src_index, priority, paint_id, paint_label."""
    m = pd.read_csv(CSV)
    m = m[m.src_index != 0].copy()                 # drop Background no-ops
    if variant == "grouped":
        m["paint_id"] = m.final_id.astype(int)
        m["paint_label"] = m.final_label
    else:  # all: unique id per distinct (src_model, src_index)
        keys = list(dict.fromkeys(zip(m.src_model, m.src_index)))   # stable, unique
        uid = {k: i + 1 for i, k in enumerate(keys)}                # 1..K, 0=background
        m["paint_id"] = [uid[(sm, si)] for sm, si in zip(m.src_model, m.src_index)]
        m["paint_label"] = m.src_name
    m = m.sort_values("priority", kind="stable")   # paint low->high; MANDATORY
    return m


def merge_case(subj, m, n_classes):
    """Priority painting of ALL 9 tasks (no gating). Load each part once.
    Returns (merged, ref_img, present)."""
    seg_dir = os.path.join(SEG, subj)
    parts, ref = {}, None
    for t in TASKS:
        f = os.path.join(seg_dir, f"{subj}_part_{t}.nii.gz")
        assert os.path.exists(f), f"MISSING part {t} for {subj}"
        img = nib.load(f)
        if ref is None:
            ref = img
        else:
            assert img.shape == ref.shape, f"shape mismatch {subj} part {t}"
            assert np.allclose(img.affine, ref.affine, atol=1e-3), f"affine mismatch {subj} part {t}"
        parts[t] = np.asarray(img.dataobj).astype(np.int16)

    dtype = np.uint8 if n_classes <= 255 else np.uint16
    out = np.zeros(ref.shape, dtype=dtype)
    for _, r in m.iterrows():
        part = parts[int(r.src_model)]
        out[part == int(r.src_index)] = int(r.paint_id)

    expected = set(int(x) for x in m.paint_id.unique()) | {0}
    present = set(int(v) for v in np.unique(out))
    assert present <= expected, f"{subj}: unexpected labels {present - expected}"
    return out, ref, present


def best_slice(merged):
    """Axial (axis=2) slice with the most labeled voxels."""
    counts = (merged > 0).sum(axis=(0, 1))
    return int(counts.argmax()) if counts.max() > 0 else merged.shape[2] // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["grouped", "all"], default="grouped")
    args = ap.parse_args()
    variant = args.variant

    out_dir = os.path.join(PROJ, "cads_demo", "merged" if variant == "grouped" else "merged_all")
    fig_path = os.path.join(PROJ, "cads_demo",
                            "merge_demo.png" if variant == "grouped" else "merge_demo_all.png")
    suffix = "_merged" if variant == "grouped" else "_merged_all"
    os.makedirs(out_dir, exist_ok=True)

    m = load_map(variant)
    id2label = m.drop_duplicates("paint_id").set_index("paint_id")["paint_label"].to_dict()
    id2label[0] = "Background"
    n_classes = int(m.paint_id.max()) + 1
    print(f"variant={variant}: {len(m)} source rows -> {m.paint_id.nunique()} classes "
          f"(ids 1..{n_classes - 1})")

    n = len(SUBJECTS)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    vmax = n_classes - 1
    cmap = plt.get_cmap("nipy_spectral", n_classes)
    present_all = set()

    for i, (region, subj) in enumerate(SUBJECTS):
        merged, ref, present = merge_case(subj, m, n_classes)
        nib.save(nib.Nifti1Image(merged, ref.affine, ref.header),
                 os.path.join(out_dir, f"{subj}{suffix}.nii.gz"))

        vals, cnts = np.unique(merged, return_counts=True)
        report = {int(v): int(c) for v, c in zip(vals, cnts) if v != 0}
        print(f"\n[{region}] {subj} ({region_of(subj)})  shape={merged.shape}  "
              f"{len(report)} labels present")

        present_all |= (present - {0})
        ct = np.asarray(nib.load(os.path.join(DATA, subj, "ct.nii")).dataobj)
        z = best_slice(merged)
        ct_sl = ct[:, :, z].T
        seg_sl = merged[:, :, z].T
        lo, hi = (-100, 100) if region == "Brain" else (-1024, 1024)
        for j in (0, 1):
            axes[i, j].imshow(ct_sl, cmap="gray", vmin=lo, vmax=hi, origin="lower")
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        masked = np.ma.masked_where(seg_sl == 0, seg_sl)
        axes[i, 1].imshow(masked, cmap=cmap, vmin=0, vmax=vmax, alpha=0.55, origin="lower",
                          interpolation="nearest")
        axes[i, 0].set_ylabel(f"{region}\n{subj}", fontsize=11)
        axes[i, 0].set_title("CT" if i == 0 else "")
        axes[i, 1].set_title(f"CT + merged seg ({variant})" if i == 0 else "")

    legend_ids = [0] + sorted(present_all)
    handles = [
        Patch(facecolor=cmap(v / vmax), edgecolor="k", linewidth=0.3,
              label=f"{v}  {id2label.get(v, '?')}")
        for v in legend_ids
    ]
    ncol = 3 if variant == "grouped" else 5
    fsize = 8 if variant == "grouped" else 6
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=ncol, fontsize=fsize, frameon=True,
               title=f"Merged label ids ({variant}, {len(legend_ids)} shown)")

    plt.tight_layout(rect=[0, 0.0, 1, 1])
    plt.savefig(fig_path, dpi=110, bbox_inches="tight")
    print(f"\nSaved figure: {fig_path}")
    print(f"Saved merged NIfTIs in: {out_dir}")


if __name__ == "__main__":
    main()
