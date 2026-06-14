"""
Verification for the all-train split + empty-val guard in unet_baseline.train.

Covers:
  PART A (logic, no GPU):
    - splits/all_train_split.txt invariants (843 train, no val/test, == center_wise set)
    - get_split_subjects() on the new split AND existing splits (regression)
    - BaseTrainer._stratify_val_indices / _log_val_metrics tolerate an empty val set
  PART B (E2E, single subject on GPU):
    - REGRESSION: split WITH val -> validate() runs, val/ logged (do_val path unchanged)
    - NEW: train-only split -> validate() skipped, no val/ keys, training completes, ckpt saved

Usage:
    cd /home/minsukc/MRI2CT
    micromamba run -n mrct python tests/test_all_train_split.py
"""

import copy
import os
import sys
import glob
import tempfile
import traceback
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("WANDB_MODE", "offline")  # don't create online runs / require login

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)  # split paths are repo-relative

results = {}


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results[name] = bool(condition)
    msg = f"  [{status}] {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return bool(condition)


# ─────────────────────────────────────────────────────────────────────────────
# PART A — logic
# ─────────────────────────────────────────────────────────────────────────────
def read_pairs(path):
    pairs = []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                pairs.append((p[0], p[1]))
    return pairs


def test_split_file_invariants():
    print("\n=== PART A1: all_train_split.txt invariants ===")
    all_train = read_pairs("splits/all_train_split.txt")
    center = read_pairs("splits/center_wise_split.txt")

    labels = {lbl for lbl, _ in all_train}
    subs = [s for _, s in all_train]
    center_subs = {s for _, s in center}

    check("all_train: every line labeled 'train'", labels == {"train"}, str(labels))
    check("all_train: no val/test lines", not (labels & {"val", "test"}))
    check("all_train: no duplicate subjects", len(subs) == len(set(subs)),
          f"{len(subs)} lines, {len(set(subs))} unique")
    check("all_train: subject set == center_wise set", set(subs) == center_subs,
          f"all_train={len(set(subs))} center_wise={len(center_subs)} "
          f"missing={len(center_subs - set(subs))} extra={len(set(subs) - center_subs)}")
    check("all_train: 843 subjects", len(subs) == 843, f"got {len(subs)}")


def test_get_split_subjects():
    print("\n=== PART A2: get_split_subjects (new + existing splits) ===")
    from common.data import get_split_subjects

    # New split
    tr = get_split_subjects("splits/all_train_split.txt", "train")
    va = get_split_subjects("splits/all_train_split.txt", "val")
    te = get_split_subjects("splits/all_train_split.txt", "test")
    check("all_train: train == 843", len(tr) == 843, f"got {len(tr)}")
    check("all_train: val == [] (the empty-val case)", va == [], f"got {len(va)}")
    check("all_train: test == []", te == [], f"got {len(te)}")
    check("all_train: train is sorted & unique", tr == sorted(set(tr)))

    # Existing splits must parse EXACTLY as before (regression)
    expected = {
        "splits/center_wise_split.txt": {"train": 427, "val": 207, "test": 209},
        "splits/thorax_center_wise_split.txt": {"train": 91, "val": 33, "test": 34},
        "splits/single_subject_split.txt": {"train": 1, "val": 1},
    }
    for path, counts in expected.items():
        for name, n in counts.items():
            got = len(get_split_subjects(path, name))
            check(f"{os.path.basename(path)} [{name}] == {n}", got == n, f"got {got}")


def test_empty_val_helpers():
    print("\n=== PART A3: BaseTrainer helpers tolerate empty val set ===")
    from common.trainer_base import BaseTrainer

    # Build a bare instance without running __init__ (avoid heavy setup).
    t = BaseTrainer.__new__(BaseTrainer)
    t.val_subjects = []
    t.cfg = SimpleNamespace(seed=42, viz_force_include=[], wandb=False)
    t.global_step = 0
    t.local_run_dir = None

    try:
        chosen, region_map = BaseTrainer._stratify_val_indices(t, n_per_region=4)
        check("_stratify_val_indices(empty) -> empty set", chosen == set() and len(region_map) == 0)
    except Exception as e:
        check("_stratify_val_indices(empty) no crash", False, repr(e))
        traceback.print_exc()

    try:
        avg = BaseTrainer._log_val_metrics(t, defaultdict(list), subject_ids=[])
        check("_log_val_metrics(empty) -> {} no crash", avg == {}, f"got {avg!r}")
    except Exception as e:
        check("_log_val_metrics(empty) no crash", False, repr(e))
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# PART B — single-subject end to end
# ─────────────────────────────────────────────────────────────────────────────
FAST_CFG = {
    "stage_data": False,
    "wandb": True,
    "project_name": "mri2ct-tests",
    "steps_per_epoch": 3,
    "total_epochs": 2,
    "val_interval": 1,
    "sanity_check": False,
    "model_save_interval": 1,
    "augment": False,
    "compile_mode": None,
    "analyze_shapes": False,
    "viz_limit": 1,
    "save_val_volumes": False,
    "patches_per_volume": 2,
    "data_queue_max_length": 10,
    "data_queue_num_workers": 2,  # train DataLoader hardcodes persistent_workers=True (needs >0)
    "val_sw_batch_size": 2,
    "val_body_mask": True,
    "patch_size": 64,
    "batch_size": 2,
    "accum_steps": 1,
    "validate_dice": False,
    "dice_w": 0.0,
    "ngf": 16,
    "num_downs": 3,
    "input_nc": 1,
    "output_nc": 1,
    "model_type": "unet_baseline",
    "diverge_wandb_branch": False,
    "resume_wandb_id": None,
    "weight_decay": 1e-4,
}


def _run_unet(split_file, label):
    """Build + train BaselineTrainer on `split_file`, spying on validate() and wandb keys."""
    import wandb
    from common.config import DEFAULT_CONFIG
    from common.utils import cleanup_gpu
    from unet_baseline.train import BaselineTrainer

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(FAST_CFG)
    cfg.update({
        "split_file": split_file,
        "run_name_prefix": f"test_alltrain_{label}",
        "wandb_note": f"test_alltrain_{label}",
        "log_dir": os.path.join(REPO_ROOT, "tests/test_logs"),
        "prediction_dir": os.path.join(REPO_ROOT, "tests/test_preds"),
    })

    validate_calls = {"n": 0}
    logged_keys = set()
    ok = False
    try:
        trainer = BaselineTrainer(cfg)

        # Spy on validate() to count invocations.
        orig_validate = trainer.validate

        def spy_validate(epoch):
            validate_calls["n"] += 1
            return orig_validate(epoch)

        trainer.validate = spy_validate

        original_log = wandb.log

        def capturing_log(data, *a, **k):
            logged_keys.update(data.keys())
            return original_log(data, *a, **k)

        with patch.object(wandb, "log", side_effect=capturing_log):
            trainer.train()

        if wandb.run:
            ckpts = glob.glob(os.path.join(wandb.run.dir, "*.pt"))
            check(f"{label}: checkpoint saved", len(ckpts) > 0, f"found {len(ckpts)}")
        ok = True
    except Exception as e:
        check(f"{label}: training completed without error", False, repr(e))
        traceback.print_exc()
    finally:
        if wandb.run:
            wandb.finish()
        cleanup_gpu()

    return validate_calls["n"], logged_keys, ok


def test_e2e_regression_val_present():
    print("\n=== PART B1: E2E REGRESSION — split WITH val (do_val path unchanged) ===")
    n_val, keys, ok = _run_unet("splits/single_subject_split.txt", "regression")
    if ok:
        check("regression: validate() ran (val present)", n_val > 0, f"called {n_val}x")
        val_keys = sorted(k for k in keys if k.startswith("val/"))
        check("regression: val/ metrics logged", len(val_keys) > 0, str(val_keys[:5]))


def test_e2e_empty_val():
    print("\n=== PART B2: E2E NEW — train-only split (empty val) ===")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir="splits") as f:
        f.write("train 1ABA005\n")  # NO val line
        tmp_split = f.name
    rel = os.path.relpath(tmp_split, REPO_ROOT)
    try:
        n_val, keys, ok = _run_unet(rel, "empty_val")
        if ok:
            check("empty_val: validate() SKIPPED", n_val == 0, f"called {n_val}x (expected 0)")
            val_keys = sorted(k for k in keys if k.startswith("val/"))
            check("empty_val: no val/ keys logged", len(val_keys) == 0, str(val_keys))
            train_keys = sorted(k for k in keys if k.startswith("train/"))
            check("empty_val: train/ keys still logged", len(train_keys) > 0, str(train_keys[:5]))
    finally:
        os.remove(tmp_split)


def main():
    test_split_file_invariants()
    test_get_split_subjects()
    test_empty_val_helpers()
    test_e2e_regression_val_present()
    test_e2e_empty_val()

    print("\n" + "=" * 60)
    passed = sum(results.values())
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    failed = [k for k, v in results.items() if not v]
    if failed:
        print("FAILED:")
        for k in failed:
            print(f"  - {k}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
