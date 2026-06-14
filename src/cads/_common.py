"""Shared config + helpers for the staged CADS segmentation pipeline.

Layout (all under OUT_ROOT, a parallel tree next to the read-only dataset):
  preprocessed/  <subj>.nii.gz          stage 1 out (intermediate, deletable after stage 3)
  metadata/      <subj>_metadata.pkl    stage 1 out (needed by stage 3)
  seg_prep/      <subj>/<subj>_part_55X.nii.gz   stage 2 out (preprocessed space, intermediate)
  seg/           <subj>/<subj>_part_55X.nii.gz   stage 3 out (original geometry, FINAL keeper)
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CADS"))

DATA_DIR = os.environ.get(
    "CADS_DATA_DIR",
    "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked",
)
OUT_ROOT = os.environ.get(
    "CADS_OUT_ROOT",
    "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/cads",
)
WEIGHTS_PATH = os.environ.get(
    "CADS_WEIGHTS_PATH",
    "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/cads_weights",
)

PREP_DIR = os.path.join(OUT_ROOT, "preprocessed")
META_DIR = os.path.join(OUT_ROOT, "metadata")
SEG_PREP_DIR = os.path.join(OUT_ROOT, "seg_prep")
SEG_DIR = os.path.join(OUT_ROOT, "seg")

ALL_TASKS = [551, 552, 553, 554, 555, 556, 557, 558, 559]
CT_NAME = "ct.nii"


def list_subjects():
    """All subject ids that have a CT, sorted (stable across shards)."""
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and os.path.exists(os.path.join(DATA_DIR, d, CT_NAME))
    )


def parse_shard(shard_str):
    """'i/N' -> (i, N). Validates 0 <= i < N."""
    i_str, n_str = shard_str.split("/")
    i, n = int(i_str), int(n_str)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"bad shard '{shard_str}': need 0 <= i < N and N >= 1")
    return i, n


def take_shard(items, shard_str):
    """Deterministic strided shard: items[i::N]. Union over i=0..N-1 == items."""
    i, n = parse_shard(shard_str)
    return items[i::n]
