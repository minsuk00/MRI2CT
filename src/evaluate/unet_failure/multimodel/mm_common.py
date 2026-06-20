"""Shared constants + helpers for the cross-model CADS error decomposition (report 11).

This is a model-parameterized replication of the report-10 U-Net pipeline
(src/evaluate/unet_failure/cads_*.py + verify_*.py). All the numeric logic is
copied verbatim from those scripts so per-model numbers are directly comparable
and the `unet` column reproduces report 10 exactly. The only change is that the
prediction subdirectory (`volumes/<MODEL>/`) and output directory are
parameterized by model.

Models (all 207 center-wise val subjects, raw HU, full_eval_20260617):
  unet, amix   -> sigmoid output, ceiling ~1024 HU (cannot represent dense bone)
  maisi        -> clipped to [-1000, 1000] (tightest ceiling)
  cwdm         -> diffusion, clipped at 1024
  mcddpm       -> diffusion, no hard ceiling (max > 1024 observed)
  koalAI       -> tanh / z-score inverse, no hard ceiling (max > 1024 observed)
"""
import os
import glob
import numpy as np
import nibabel as nib

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
OUTROOT = os.path.join(REPO, "evaluation_results/cads_multimodel_20260620")

# display order; the cap annotation drives the report narrative
MODELS = ["unet", "amix", "maisi", "cwdm", "mcddpm", "koalAI"]
MODEL_LABEL = {
    "unet": "U-Net", "amix": "Anatomix", "maisi": "MAISI",
    "cwdm": "cWDM", "mcddpm": "MC-DDPM", "koalAI": "koalAI",
}
# empirically confirmed output ceiling behaviour (see plan / mm_caps.json)
MODEL_CAP = {
    "unet": "sigmoid ~1024", "amix": "sigmoid ~1024", "maisi": "clip 1000",
    "cwdm": "clip 1024", "mcddpm": "none (>1024)", "koalAI": "none (>1024)",
}
MODEL_COLOR = {
    "unet": "#dc2626", "amix": "#ea580c", "maisi": "#ca8a04",
    "cwdm": "#16a34a", "mcddpm": "#2563eb", "koalAI": "#7c3aed",
}

NL = 35
BONE = [7, 27, 28, 29, 30]
AIRORG = [9, 13]
CLASS_NAMES = [
    "Background", "Brain - other", "CSF", "Eyes & optic pathway", "Face & oral soft tissue",
    "Gray matter", "Head & neck glands", "Skull", "White matter", "Airway", "Breast",
    "Esophagus", "Heart", "Lungs", "Thoracic cavity", "Abdominal cavity", "Adrenals",
    "Bowel", "Gallbladder", "Kidneys", "Liver", "Pancreas", "Spleen", "Stomach", "Bladder",
    "Prostate & seminal vesicle", "Blood vessels", "Bone - other", "Limb & girdle bones",
    "Spine", "Thoracic cage", "Gland - other", "Muscle", "Spinal cord", "Subcutaneous tissue",
]
# calibration histogram axes. GT full range; pred axis widened vs report 10 so the
# uncapped models (mcddpm/koalAI, max ~1236) are not clipped at the top bin.
EDG = np.linspace(-1024, 3000, 202)
EDP = np.linspace(-1100, 1600, 136)

# within-bone GT-HU density bands (report 10 / cads_bone_hu_split.py)
HU_EDGES = [-1024, 150, 300, 600, 1024, 4000]
HU_BANDS = ["<150 (marrow/soft)", "150-300 (trabecular)", "300-600 (cortical)",
            "600-1024 (dense cortical)", ">1024 (above 1024)"]
NHU = len(HU_BANDS)


def reg(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def subjects():
    """The 207 center-wise val subjects (same discovery as report 10)."""
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))


def sct_path(model, s):
    return f"{EVAL}/volumes/{model}/{s}/sample.nii.gz"


def run_dir(model):
    d = os.path.join(OUTROOT, model)
    return d


def fig_dir(model):
    return os.path.join(run_dir(model), "figures")


def cross_dir():
    return os.path.join(OUTROOT, "_cross")


def ensure(model):
    os.makedirs(fig_dir(model), exist_ok=True)
    os.makedirs(cross_dir(), exist_ok=True)
    os.makedirs(os.path.join(cross_dir(), "figures"), exist_ok=True)
