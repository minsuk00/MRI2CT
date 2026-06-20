"""Core for the real retrieval/atlas experiment: retrieve a donor by MR, deform
its CT onto the query (ANTs SyN), and read off what it delivers on bone.

This implements the ACTUAL atlas-based-sCT pipeline the user proposes (retrieve a
CT via MR similarity, register, use it), so we can measure - not simulate - what a
registered donor CT gives on bone, split into intensity (HU MAE) and sharpness.
"""
import os
import numpy as np
import nibabel as nib
import ants

REPO = "/home/minsukc/MRI2CT"
DATA = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SCT = f"{REPO}/evaluation_results/full_eval_20260617/volumes/unet"
BONE_LABELS = [7, 27, 28, 29, 30]
SPACING = (1.5, 1.5, 1.5)


def reg(s):
    return {"AB": "abdomen", "TH": "thorax", "HN": "head_neck"}.get(s[1:3].upper(), "other")


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def load_raw(s, need_sct=False):
    """Raw CT (HU), MR (native), body, CADS seg, optional U-Net sCT. All same grid."""
    ct = canon(f"{DATA}/{s}/ct.nii")
    mr = canon(f"{DATA}/{s}/moved_mr.nii")
    body = canon(f"{DATA}/{s}/mask.nii") > 0
    seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    sct = canon(f"{SCT}/{s}/sample.nii.gz") if need_sct else None
    return dict(ct=ct, mr=mr, body=body, seg=seg, sct=sct)


def mr_norm(mr, body):
    """Robust per-subject MR scaling to [0,1] on body voxels (p0.5..p99.5)."""
    bv = mr[body]
    lo, hi = np.percentile(bv, 0.5), np.percentile(bv, 99.5)
    return np.clip((mr - lo) / (hi - lo + 1e-6), 0, 1)


def descriptor(vol, body, k=24):
    """Global retrieval descriptor: body-masked volume resampled to k^3, z-scored."""
    import scipy.ndimage as ndi
    v = vol * body
    zoom = [k / s for s in v.shape]
    d = ndi.zoom(v, zoom, order=1).ravel()
    return (d - d.mean()) / (d.std() + 1e-6)


def rank_donors(query_desc, donor_descs):
    """Return donor indices sorted by descending cosine similarity to the query."""
    sims = [float(np.dot(query_desc, dd) / (len(dd))) for dd in donor_descs]
    return list(np.argsort(sims)[::-1]), sims


def to_ants(a):
    img = ants.from_numpy(np.ascontiguousarray(a.astype(np.float32)))
    img.set_spacing(SPACING)
    return img


def _downsample(a, factor):
    import scipy.ndimage as ndi
    return ndi.zoom(a.astype(np.float32), 1.0 / factor, order=1)


def register(fix_img_np, mov_img_np, transform="SyNRA", ds=2):
    """Compute the SyN transform mov->fix on ds-downsampled volumes (fast). The
    transform lives in physical space, so it can be applied to FULL-RES carries.
    Returns the fwdtransforms list (moving->fixed resampling)."""
    fixed_d = to_ants(_downsample(fix_img_np, ds)); fixed_d.set_spacing(tuple(s * ds for s in SPACING))
    moving_d = to_ants(_downsample(mov_img_np, ds)); moving_d.set_spacing(tuple(s * ds for s in SPACING))
    tx = ants.registration(fixed=fixed_d, moving=moving_d, type_of_transform=transform)
    return tx["fwdtransforms"]


def warp(fwd, fix_img_np, carry_np):
    """Apply a transform (from register) to a full-res carry onto the full-res fix grid."""
    fixed_full, carry_full = to_ants(fix_img_np), to_ants(carry_np)
    warped = ants.apply_transforms(fixed=fixed_full, moving=carry_full,
                                   transformlist=fwd, interpolator="linear")
    return warped.numpy()


def register_warp(fix_img_np, mov_img_np, carry_np, transform="SyNRA", ds=2):
    """Register mov->fix and warp carry into the fix grid (validated convenience wrapper)."""
    return warp(register(fix_img_np, mov_img_np, transform, ds), fix_img_np, carry_np)


if __name__ == "__main__":
    # Single-pair sanity + timing before scaling.
    import time
    tr = [s for s in os.listdir(DATA) if reg(s) == "head_neck" and s[3] == "A"]
    tr = sorted(tr)[:2]
    q, d = tr[0], tr[1]
    print(f"query={q} donor={d}")
    Q = load_raw(q); D = load_raw(d)
    qmr = mr_norm(Q["mr"], Q["body"]); dmr = mr_norm(D["mr"], D["body"])
    t0 = time.time()
    # realistic arm: register donor MR -> query MR, carry donor CT
    warped_ct = register_warp(qmr, dmr, D["ct"], "SyN")
    dt = time.time() - t0
    print(f"SyN MR->MR took {dt:.1f}s; warped_ct shape {warped_ct.shape} vs query ct {Q['ct'].shape}")
    # sanity: warped donor CT should resemble query CT in the body more than the raw donor does
    qbody = Q["body"]
    mae_warped = np.abs(warped_ct - Q["ct"])[qbody].mean()
    print(f"body MAE (warped donor CT vs query CT) = {mae_warped:.1f} HU")
    bone = np.isin(Q["seg"], BONE_LABELS) & qbody
    print(f"bone MAE (warped donor CT vs query CT) = {np.abs(warped_ct - Q['ct'])[bone].mean():.1f} HU "
          f"(query bone voxels: {int(bone.sum())})")
