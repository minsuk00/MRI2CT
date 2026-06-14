# pyRadPlan downstream dose evaluation

[pyRadPlan](https://github.com/e0404/pyRadPlan) (a Python port of matRad) is used here
as a **downstream task** for sCT vs gtCT: build a radiotherapy treatment plan, then
recompute the dose on the other CT with the *same* fluence and compare
(`calc_dose_forward(ct, cst, stf, pln, w=w)`), so HU-synthesis errors show up as
dosimetric differences. Verified that pyRadPlan dose **does** depend on HU (same-fluence
water-block vs bone-block phantom: MAE 0.19 Gy), so the comparison is meaningful.

## ⚠️ Requires a SEPARATE micromamba env — NOT `mrct`

```bash
micromamba activate pyradplan      # pyRadPlan 0.4.0 + torch (cu128) + nibabel + matplotlib
```

**Do not `pip install pyRadPlan` into `mrct`.** pyRadPlan requires `numpy>=2`, and even
older releases drag in `numba`/`numpydantic` that force-upgrade numpy. That broke `mrct`
(scikit-image C-extension `numpy.dtype size changed` errors; `tptbox` needs `numpy<2`).
The dedicated env keeps numpy 2.x isolated from `mrct`'s numpy 1.26.4.

Recreate the env if needed:

```bash
micromamba create -y -n pyradplan python=3.11
micromamba run -n pyradplan pip install pyRadPlan nibabel matplotlib
micromamba run -n pyradplan pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Run

```bash
micromamba run -n pyradplan python pyradplan/sct_dose_eval.py [SUBJECT_ID]   # default 1THB008
# -> writes pyradplan/out/<subj>_sct_dose_eval.png
```

`sct_dose_eval.py` does the full pipeline: HU-sensitivity sanity check, build gtCT,
plant a synthetic spherical PTV (TotalSeg has no tumor) + body EXTERNAL, optimize a
5-beam photon IMRT plan, forward-recompute dose on gtCT (optimized + open/uniform field),
build a **proxy sCT** (bone HU biased -250, since no real sCT is on disk yet), forward
dose on the proxy with the same fluence, then report PTV Dmean/D95 shift, body MAE,
3%-dose-difference pass rate, and a DVH — and save a 6-panel visualization.

Example (1THB008, ~50s on CPU): PTV Dmean 58.81→58.96 Gy (Δ0.24% Rx), body MAE
0.0054 Gy, max local Δ 0.64 Gy (1.0% Dmax), 3%/Dmax pass = 100%. Photons are fairly
forgiving to HU error; swap to a proton plan (`IonPlan`, HongPB engine) for a far more
HU-sensitive probe.

To plug in a **real** sCT, replace the proxy-sCT block: load the model's sCT NIfTI as
`sct` (HU, same grid/affine as the gtCT) instead of `hu.copy(); ... -= 250`.

## GPU

pyRadPlan auto-selects an Array-API backend `cupy > torch > jax > numpy`, using a GPU
backend if torch/cupy is installed and a GPU is visible (verified: torch 2.x+cu128,
A40, `array_api_compat.torch`).

**We run the dose calc on CPU because the GPU path needs a patch and gives no speedup.**

Two facts, both measured on this box (A40, subject 1THB008, 5-beam photon IMRT):

1. The default torch-GPU path *crashes*: `_svdpb.py:347` builds `betas`/`m_arr` with
   `xp.asarray(numpy)`, which defaults to a **CPU** tensor while `rd` is on **cuda:0** →
   `RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu`. That's a
   missing `device=` in upstream, workable around with `torch.set_default_device("cuda")`.
2. **Even with that workaround, GPU is no faster** — basically a tie:

   | backend | dose-influence | optimization | dose max |
   |---|---|---|---|
   | CPU (numpy)         | 10.6 s | 25.9 s | 62.679 |
   | GPU (device-patched)| 11.0 s | 24.5 s | 62.670 |

   The GPU *is* used (profiled: 90–95% utilization during the optimization phase, peak mem
   only 1.4 GB) — it just doesn't finish faster. This plan is small for a GPU (705 beamlets,
   sparse dose matrix), and the dominant cost (~26 s) is the nonlinear fluence optimizer:
   an inherently serial, iterative line-search of many tiny dependent ops. That's
   latency/launch-overhead bound, not throughput bound, so the GPU's parallelism has nothing
   to bite on. GPUs win on large *parallel* work — a big Monte-Carlo dose calc (TOPAS/FRED)
   or a many-thousand-beamlet optimization — not this small analytical photon plan.

So `sct_dose_eval.py` sets `xp.PREFER_GPU = False` — the **simpler, equally-fast, crash-free**
choice (no monkeypatch needed). MATLAB/Octave are not needed.

## API notes (pyRadPlan 0.4.0)

```python
from pyRadPlan import PhotonPlan, IonPlan, generate_stf, calc_dose_influence, \
    fluence_optimization
from pyRadPlan.ct import create_ct            # builders live in submodules, not top-level
from pyRadPlan.cst import create_voi, create_cst
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

ct  = create_ct(cube_hu=sitk_hu_image)               # MUST be the cube_hu= kwarg
voi = create_voi(voi_type="TARGET", name="PTV", ...) # voi_type= must be a keyword

dij = calc_dose_influence(ct, cst, stf, pln)         # influence matrix (sparse, per CT)
w   = fluence_optimization(ct, cst, stf, dij, pln)   # returns the weight VECTOR (ndarray)
dose = dij.compute_result_ct_grid(w)["physical_dose"]  # -> sitk.Image; GetArrayFromImage for np
```

**sCT-vs-gtCT recompute:** build a fresh `dij` from the sCT (`calc_dose_influence` on the
sCT's `create_ct`), then `dij_sct.compute_result_ct_grid(w)` with the gtCT-optimized `w`.
(`calc_dose_forward(ct, cst, stf, pln, weights=w)` exists too but returns a `Dij`; the
`compute_result_ct_grid` path is simpler and is what the script uses.)
