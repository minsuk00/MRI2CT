"""Standalone colortable PNGs: every label id -> color + name (grouped 35 and all 167)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_demo import load_map  # noqa: E402


def colortable(variant, ncol, out, figw):
    m = load_map(variant)
    id2label = m.drop_duplicates("paint_id").set_index("paint_id")["paint_label"].to_dict()
    id2label[0] = "Background"
    n_classes = int(m.paint_id.max()) + 1
    vmax = n_classes - 1
    cmap = plt.get_cmap("nipy_spectral", n_classes)
    ids = list(range(n_classes))
    handles = [Patch(facecolor=cmap(v / vmax), edgecolor="k", linewidth=0.4,
                     label=f"{v}  {id2label.get(v, '?')}") for v in ids]
    nrow = (len(ids) + ncol - 1) // ncol
    fig = plt.figure(figsize=(figw, 0.34 * nrow + 0.8))
    fig.legend(handles=handles, loc="center", ncol=ncol, fontsize=8 if variant == "grouped" else 6.5,
               frameon=False, title=f"CADS merged labels — '{variant}' ({n_classes} classes incl. background)")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out, f"({n_classes} labels)")


if __name__ == "__main__":
    colortable("grouped", ncol=2, out="cads_demo/labels_grouped.png", figw=9)
    colortable("all", ncol=4, out="cads_demo/labels_all.png", figw=15)
