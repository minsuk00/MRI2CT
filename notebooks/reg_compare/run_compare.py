"""
Three-way deformable-registration comparison, all at NATIVE resolution.

For one subject, three MR<->CT alignments, one row each:
  #1 anatomix  (what we have now): native_registered/<reg>/<id>/{ct.nii.gz, mr_moved.nii.gz}
                moving=MR, fixed=CT  -> MR deformed onto CT.
  #2 elastix "apples" (our direction): elastix B-spline, fixed=CT, moving=MR.
  #3 elastix "faithful" (SynthRAD direction): elastix B-spline, fixed=MR, moving=CT.

All three start from the SAME raw rigid MR+CT pair at native spacing (no resolution confound).

Per plane (axial/coronal/sagittal) we render one figure: rows = the 3 methods, columns =
  moving | moved | fixed | overlay(fixed+moving) | overlay(fixed+moved)
Overlay is RGB with R=CT, G=MR: aligned -> yellow, CT-only -> red, MR-only -> green.
The "before" overlay (fixed+moving) is rigid-only; the "after" overlay (fixed+moved) shows
what the deformable step fixed.

Elastix volumes are cached on GPFS and reused; delete them to force re-registration.

Run in the `elastix` env:
  micromamba run -n elastix python notebooks/reg_compare/run_compare.py --id 1THA001 --region thorax
"""
import argparse
import os

import matplotlib
import numpy as np
import SimpleITK as sitk

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/home/minsukc/MRI2CT/dataset"
PARAMS = os.path.join(HERE, "params")
GPFS_VOL = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/reg_compare_vols"

# region -> (raw dataset subdir, native_registered subdir, elastix param file)
REGION = {
    "thorax":    ("SynthRAD2025/Task1/TH",     "TH",     "param_def_mr_TH.txt"),
    "abdomen":   ("SynthRAD2025/Task1/AB",     "AB",     "param_def_mr_AB.txt"),
    "head_neck": ("SynthRAD2025/Task1/HN",     "HN",     "param_def_mr_HN.txt"),
    "brain":     ("SynthRAD2023/Task1/brain",  "brain",  "param_def_mr_brain_ADAPTED_from_HN.txt"),
    "pelvis":    ("SynthRAD2023/Task1/pelvis", "pelvis", "param_def_mr_pelvis_ADAPTED_from_AB.txt"),
}
CT_WIN = (-1024, 1024)   # default CT window (grayscale + overlay red channel)
CT_WIN_BY_REGION = {"brain": (-100, 100)}  # narrow window reveals brain soft tissue
# axial slice as a fraction of the body z-extent; default (None) = mask center-of-mass.
# HN: 0.65 lands at orbits/sinuses/brain-base (anatomy-rich), not the small neck or skull cap.
AXIAL_ZFRAC = {"head_neck": 0.65}


def read(path):
    if not os.path.exists(path):
        alt = path[:-3] if path.endswith(".gz") else path + ".gz"
        path = alt if os.path.exists(alt) else path
    return sitk.ReadImage(path)


def elastix_deformable(fixed, moving, param_file, mask, default_value):
    p = sitk.ReadParameterFile(param_file)
    p["DefaultPixelValue"] = (str(default_value),)
    f = sitk.ElastixImageFilter()
    f.SetParameterMap(p)
    f.SetFixedImage(fixed)
    f.SetMovingImage(moving)
    if mask is not None:
        f.SetFixedMask(sitk.Cast(mask, sitk.sitkUInt8))
    f.SetNumberOfThreads(int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 4))
    f.LogToConsoleOff()
    f.LogToFileOff()
    f.Execute()
    return f.GetResultImage()


def arr(img):
    return sitk.GetArrayFromImage(img)  # (z, y, x)


def nrm(a, lo, hi):
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def bbox(mask_a, margin=6):
    idx = np.argwhere(mask_a > 0)
    if len(idx) == 0:
        return [(0, s) for s in mask_a.shape]
    out = []
    for d in range(3):
        lo = max(0, idx[:, d].min() - margin)
        hi = min(mask_a.shape[d], idx[:, d].max() + margin)
        out.append((int(lo), int(hi)))
    return out  # [(zlo,zhi),(ylo,yhi),(xlo,xhi)]


def plane_slice(a, plane, com, bb):
    (zlo, zhi), (ylo, yhi), (xlo, xhi) = bb
    cz, cy, cx = com
    if plane == "axial":
        return a[cz, ylo:yhi, xlo:xhi]
    if plane == "coronal":
        return a[zlo:zhi, cy, xlo:xhi]
    return a[zlo:zhi, ylo:yhi, cx]  # sagittal


def gray(ax, s2d, is_ct, aspect, ctwin):
    if is_ct:
        ax.imshow(nrm(s2d, *ctwin), cmap="gray", origin="lower", aspect=aspect, vmin=0, vmax=1)
    else:
        lo, hi = np.percentile(s2d, [1, 99])
        ax.imshow(s2d, cmap="gray", origin="lower", aspect=aspect, vmin=lo, vmax=hi)
    ax.set_xticks([]); ax.set_yticks([])


def overlay(ax, ct2d, mr2d, aspect, ctwin):
    r = nrm(ct2d, *ctwin)
    g = nrm(mr2d, *np.percentile(mr2d, [1, 99]))
    rgb = np.zeros((*ct2d.shape, 3))
    rgb[..., 0] = r
    rgb[..., 1] = g
    ax.imshow(rgb, origin="lower", aspect=aspect)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--region", required=True, choices=list(REGION))
    ap.add_argument("--zfrac", type=float, default=None,
                    help="axial slice as fraction of body z-extent (0-1); default = mask COM / region default")
    args = ap.parse_args()

    raw_sub, nat_sub, param_name = REGION[args.region]
    param_file = os.path.join(PARAMS, param_name)
    raw_dir = os.path.join(DATA, raw_sub, args.id)
    nat_dir = os.path.join(DATA, "native_registered", nat_sub, args.id)
    out_local = os.path.join(HERE, "out")
    os.makedirs(out_local, exist_ok=True)
    out_dir = os.path.join(GPFS_VOL, args.id)   # per-subject GPFS folder: niftis + plane PNGs
    os.makedirs(out_dir, exist_ok=True)
    link = os.path.join(out_local, args.id)     # repo symlink into the GPFS folder
    if not os.path.lexists(link):
        os.symlink(out_dir, link)

    print(f"[{args.id}] loading raw rigid pair + anatomix native ...")
    ct_raw = read(os.path.join(raw_dir, "ct.nii.gz"))
    mr_raw = read(os.path.join(raw_dir, "mr.nii.gz"))
    mask = read(os.path.join(raw_dir, "mask.nii.gz"))
    ct_amix = read(os.path.join(nat_dir, "ct.nii.gz"))
    mr_amix = read(os.path.join(nat_dir, "mr_moved.nii.gz"))

    mr2ct_path = os.path.join(out_dir, "mr_def_elastix.nii.gz")
    ct2mr_path = os.path.join(out_dir, "ct_def_elastix.nii.gz")
    if os.path.exists(mr2ct_path) and os.path.exists(ct2mr_path):
        print(f"[{args.id}] reusing cached elastix volumes")
        mr_def = read(mr2ct_path)
        ct_def = read(ct2mr_path)
    else:
        print(f"[{args.id}] #2 elastix MR→CT (fixed=CT, moving=MR) ...")
        mr_def = elastix_deformable(ct_raw, mr_raw, param_file, mask, default_value=0)
        sitk.WriteImage(mr_def, mr2ct_path)
        print(f"[{args.id}] #3 elastix CT→MR (fixed=MR, moving=CT) ...")
        ct_def = elastix_deformable(mr_raw, ct_raw, param_file, mask, default_value=-1000)
        sitk.WriteImage(ct_def, ct2mr_path)

    sx, sy, sz = ct_raw.GetSpacing()
    asp = {"axial": sy / sx, "coronal": sz / sx, "sagittal": sz / sy}

    mask_a = arr(mask)
    bb = bbox(mask_a)
    com = tuple(int(round(v)) for v in np.argwhere(mask_a > 0).mean(0))
    # axial z: region default / CLI override via fraction of body z-extent, else COM
    ctwin = CT_WIN_BY_REGION.get(args.region, CT_WIN)
    zfrac = args.zfrac if args.zfrac is not None else AXIAL_ZFRAC.get(args.region)
    if zfrac is not None:
        (zlo, zhi), _, _ = bb
        com = (int(round(zlo + zfrac * (zhi - zlo))), com[1], com[2])

    # per method: (label, moving, moved, fixed, moving_is_ct)
    methods = [
        ("#1 anatomix  (MR→CT)", arr(mr_raw), arr(mr_amix), arr(ct_raw), False),
        ("#2 elastix  (MR→CT)", arr(mr_raw), arr(mr_def), arr(ct_raw), False),
        ("#3 elastix  (CT→MR)", arr(ct_raw), arr(ct_def), arr(mr_raw), True),
    ]
    col_titles = ["moving", "moved", "fixed", "overlay: fixed+moving (rigid)", "overlay: fixed+moved (deformable)"]

    for plane in ["axial", "coronal", "sagittal"]:
        a = asp[plane]
        fig, axes = plt.subplots(3, 5, figsize=(16, 9))
        for r, (label, mv, md, fx, mv_is_ct) in enumerate(methods):
            mv_s = plane_slice(mv, plane, com, bb)
            md_s = plane_slice(md, plane, com, bb)
            fx_s = plane_slice(fx, plane, com, bb)
            mv_is_ct = mv_is_ct  # moving/moved modality
            gray(axes[r, 0], mv_s, mv_is_ct, a, ctwin)
            gray(axes[r, 1], md_s, mv_is_ct, a, ctwin)
            gray(axes[r, 2], fx_s, not mv_is_ct, a, ctwin)  # fixed is the other modality
            # overlay: R=CT, G=MR regardless of direction
            if mv_is_ct:           # #3: moving/moved=CT, fixed=MR
                overlay(axes[r, 3], mv_s, fx_s, a, ctwin)   # before: rawCT + MR
                overlay(axes[r, 4], md_s, fx_s, a, ctwin)   # after:  defCT + MR
            else:                  # #1/#2: moving/moved=MR, fixed=CT
                overlay(axes[r, 3], fx_s, mv_s, a, ctwin)   # before: CT + rigid MR
                overlay(axes[r, 4], fx_s, md_s, a, ctwin)   # after:  CT + moved MR
            axes[r, 0].set_ylabel(label, fontsize=9)
            if r == 0:
                for c, t in enumerate(col_titles):
                    axes[0, c].set_title(t, fontsize=9)
        fig.suptitle(f"{args.id} ({args.region}) — {plane} @ {sx:.0f}x{sy:.0f}x{sz:.0f}mm  "
                     f"| overlay R=CT G=MR (aligned=yellow)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_png = os.path.join(out_dir, f"{args.id}_{plane}.png")   # PNGs live with niftis on GPFS
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[{args.id}] wrote {out_png}")

    # per-subject flip frames: bare axial PNGs (same slice/crop/window) to toggle in a
    # viewer. All images share one voxel grid, so frames are pixel-aligned -> differences
    # "jump" when flipping. MR frames share one window so brightness is consistent.
    flip_dir = os.path.join(out_dir, "flip")
    os.makedirs(flip_dir, exist_ok=True)
    mrwin = tuple(np.percentile(plane_slice(methods[0][1], "axial", com, bb), [1, 99]))
    for r, (label, mv, md, fx, mv_is_ct) in enumerate(methods, start=1):
        fx_s = plane_slice(fx, "axial", com, bb)
        md_s = plane_slice(md, "axial", com, bb)
        fx_win = ctwin if not mv_is_ct else mrwin   # fixed modality
        md_win = ctwin if mv_is_ct else mrwin       # moved modality
        plt.imsave(os.path.join(flip_dir, f"{r}_fixed.png"), fx_s, cmap="gray",
                   vmin=fx_win[0], vmax=fx_win[1], origin="lower")
        plt.imsave(os.path.join(flip_dir, f"{r}_moved.png"), md_s, cmap="gray",
                   vmin=md_win[0], vmax=md_win[1], origin="lower")
    print(f"[{args.id}] wrote {flip_dir}/[1-3]_{{fixed,moved}}.png")


if __name__ == "__main__":
    main()
