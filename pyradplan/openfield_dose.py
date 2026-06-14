"""Simplest possible pyRadPlan dose calc: UNIFORM fluence (no plan, no optimizer).

REQUIRES THE DEDICATED `pyradplan` ENV (see README.md):
    micromamba run -n pyradplan python pyradplan/openfield_dose.py [SUBJECT_ID]

We only need: CT + a target mask + gantry angles. The target mask defines the
isocenter (its center of mass) and the beamlet field (its projected shadow per
angle). "Uniform fluence" = every beamlet weight = 1.0, so there is NO objective,
NO prescription, NO optimization -- dose is just D @ ones, where D is the
dose-influence matrix. We run it twice:
  (1) target = whole BODY mask   -> open field over the whole patient
  (2) target = synthetic SPHERE  -> a pencil-ish field through the body center
and save a dose figure for each.

Note: pyRadPlan 0.4.0's photon SVD engine breaks on the torch-GPU backend, so we
force the numpy/CPU backend (fast for this).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "1THB008"
ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
OUT = os.path.join(os.path.dirname(__file__), "out")
# Evenly-spaced beams so the individual beam paths are visible as spokes converging
# on the target. 0=from top, 90=patient-left, 180=below, 270=patient-right.
GANTRY = [0, 72, 144, 216, 288]   # 5 beams, deg
SPACING = 1.5             # mm, matches the SynthRAD data


def to_sitk(arr_xyz, cast=np.float32):
    """numpy (x,y,z) -> SimpleITK image (z,y,x) at SPACING mm."""
    a = np.ascontiguousarray(np.transpose(arr_xyz, (2, 1, 0))).astype(cast)
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((SPACING,) * 3)
    return img


def uniform_dose(hu, target, gantry, body=None):
    """Build geometry from (CT, target mask, angles) and return dose for uniform fluence.

    The TARGET defines isocenter + beamlet field. We ALSO add an EXTERNAL body VOI so
    the dose engine computes dose over the WHOLE body (its calc region = union of VOIs),
    not just a tight shell around the target -- otherwise the entrance/exit beam
    corridors are never computed and the dose looks target-only. No objectives needed.

    Returns a numpy dose cube in (z,y,x) order (matches sitk / our transposed masks).
    """
    from pyRadPlan import PhotonPlan, generate_stf, calc_dose_influence
    from pyRadPlan.ct import create_ct
    from pyRadPlan.cst import create_voi, create_cst

    ct = create_ct(cube_hu=to_sitk(hu))
    vois = [create_voi(voi_type="TARGET", name="Target", ct_image=ct,
                       mask=to_sitk(target, np.uint8))]
    if body is not None:
        vois.append(create_voi(voi_type="EXTERNAL", name="Body", ct_image=ct,
                               mask=to_sitk(body, np.uint8)))   # -> dose calc'd over whole body
    cst = create_cst(vois, ct=ct)

    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {"gantry_angles": gantry, "couch_angles": [0] * len(gantry)}

    stf = generate_stf(ct, cst, pln)                  # rays/beamlets from target shadow
    dij = calc_dose_influence(ct, cst, stf, pln)      # D: (n_voxels, n_beamlets), sparse
    w = np.ones(dij.total_num_of_bixels, np.float32)  # UNIFORM fluence
    img = dij.compute_result_ct_grid(w)["physical_dose"]   # dose = D @ w  -> sitk.Image
    dose = sitk.GetArrayFromImage(img).astype(np.float32)
    return dose, dij.total_num_of_bixels


def plot_dose(hu, target, dose, title, outpng):
    """2-panel axial slice through the target center: CT+target, dose overlaid on CT.

    With UNIFORM weights each pencil beam deposits little (~0.3-0.5 Gy) while the
    convergence center is ~10x hotter, so a 0->dmax scale renders the corridors near
    black. We set the color ceiling to the corridor dose level (95th pct of in-beam
    dose OUTSIDE the target), so the beam paths get the full color range and the
    target just saturates red.
    """
    bb_t = np.transpose(target, (2, 1, 0)).astype(bool)
    hu_t = np.transpose(hu, (2, 1, 0))
    sl = int(np.argmax(bb_t.reshape(bb_t.shape[0], -1).sum(1)))  # slice with most target voxels
    dmax = float(dose.max()) if dose.max() > 0 else 1.0

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    ax[0].imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000)
    ax[0].contour(bb_t[sl], colors="lime", linewidths=0.9)
    ax[0].set_title("CT (HU) + target outline (green)")

    ax[1].imshow(hu_t[sl], cmap="gray", vmin=-1000, vmax=1000)
    im = ax[1].imshow(np.ma.masked_less(dose[sl], 0.05 * dmax), cmap="jet", alpha=0.6,
                      vmin=0, vmax=dmax)   # corridors computed (body VOI) -> beams show as blue
    ax[1].contour(bb_t[sl], colors="lime", linewidths=0.9)
    ax[1].set_title("dose overlaid on CT (beams enter from each gantry angle)")
    fig.colorbar(im, ax=ax[1], fraction=0.046, label="dose (Gy, arb. unit fluence)")

    for a in ax:
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpng, dpi=120)
    plt.close(fig)
    return sl, dmax


def main():
    from pyRadPlan.core import xp_utils as xp
    xp.PREFER_GPU = False     # 0.4.0 photon SVD engine is broken on GPU; CPU is fast here
    os.makedirs(OUT, exist_ok=True)

    p = f"{ROOT}/{SUBJ}/"
    hu = nib.load(p + "ct.nii").get_fdata().astype(np.float32)
    body = nib.load(p + "mask.nii").get_fdata().astype(np.uint8)
    print("SUBJ %s  shape=%s  HU=[%.0f,%.0f]  body_vox=%d  gantry=%s"
          % (SUBJ, hu.shape, hu.min(), hu.max(), int(body.sum()), GANTRY))

    # (1) target = whole body  -> open field
    dose_b, nb_b = uniform_dose(hu, body, GANTRY)
    sl_b, dmax_b = plot_dose(hu, body, dose_b,
                             f"{SUBJ}  UNIFORM fluence, target=BODY (open field)  beams={GANTRY}  nbeamlets={nb_b}",
                             os.path.join(OUT, f"{SUBJ}_openfield_body.png"))
    print("BODY   nbeamlets=%d  dose max=%.3f Gy  (slice %d)" % (nb_b, dmax_b, sl_b))

    # (2) target = synthetic sphere at body centroid
    c = [int(v.mean()) for v in np.nonzero(body > 0)]
    X, Y, Z = np.indices(body.shape)
    sphere = ((((X - c[0])**2 + (Y - c[1])**2 + (Z - c[2])**2) <= 20**2).astype(np.uint8)) & body
    dose_s, nb_s = uniform_dose(hu, sphere, GANTRY, body=body)   # body VOI -> corridors computed
    sl_s, dmax_s = plot_dose(hu, sphere, dose_s,
                             f"{SUBJ}  UNIFORM fluence, target=SPHERE(r=20vox)  beams={GANTRY}  nbeamlets={nb_s}",
                             os.path.join(OUT, f"{SUBJ}_openfield_sphere.png"))
    print("SPHERE nbeamlets=%d  dose max=%.3f Gy  (slice %d)" % (nb_s, dmax_s, sl_s))
    print("SAVED", os.path.join(OUT, f"{SUBJ}_openfield_body.png"),
          os.path.join(OUT, f"{SUBJ}_openfield_sphere.png"))
    print("PASS")


if __name__ == "__main__":
    main()
