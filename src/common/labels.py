"""Canonical CADS 35-class label grouping shared by the seg teacher and the
translator dice loss. Index == integer label value in the seg maps; this order
is baked into the trained teacher checkpoints, so do NOT reorder.
"""
import re

CADS_35_CLASS_NAMES = [
    "Background", "Brain - other", "CSF", "Eyes & optic pathway",
    "Face & oral soft tissue", "Gray matter", "Head & neck glands", "Skull",
    "White matter", "Airway", "Breast", "Esophagus", "Heart", "Lungs",
    "Thoracic cavity", "Abdominal cavity", "Adrenals", "Bowel", "Gallbladder",
    "Kidneys", "Liver", "Pancreas", "Spleen", "Stomach", "Bladder",
    "Prostate & seminal vesicle", "Blood vessels", "Bone - other",
    "Limb & girdle bones", "Spine", "Thoracic cage", "Gland - other",
    "Muscle", "Spinal cord", "Subcutaneous tissue",
]

# Bone-family classes: Skull, Bone-other, Limb & girdle bones, Spine, Thoracic cage.
BONE_CLASS_INDICES = [7, 27, 28, 29, 30]


def class_slug(idx, name=None):
    """wandb-safe per-class key like '07_skull' / '28_limb_girdle_bones'."""
    name = CADS_35_CLASS_NAMES[idx] if name is None else name
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{idx:02d}_{slug}"
