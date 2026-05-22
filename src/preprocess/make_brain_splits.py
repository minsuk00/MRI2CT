"""Generate brain-only split files for the two diagnostic experiments.

Experiments:
  #1 brain_random_split.txt
       Stratified random shuffle: 20 BA + 20 BB + 20 BC per role -> 60/60/60.
       Train size matches the center-wise split exactly, so any PSNR gap
       isolates "which cohorts are seen at train" from "how many train subjects".
       Tests: is brain learnable when training sees all 3 cohorts?

  #2 brain_center_wise_split.txt
       Train: all 60 BA            (cohort A)
       Val:   30 BB + 30 BC        (half of each non-A cohort)
       Test:  30 BB + 30 BC        (the other half of each non-A cohort)
       Val and test are statistically equivalent cross-cohort samples, so the
       only variable vs. the random split is which cohort the model trained on.

Subject pool is read from splits/center_wise_split.txt so the available-subject
list matches what the rest of the pipeline already uses. Seed is fixed for
reproducibility — re-running the script overwrites the split files identically.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "splits"
SOURCE_SPLIT = SPLITS_DIR / "center_wise_split.txt"

SEED = 42

# Experiment #1: per-cohort subjects assigned to each role (stratified random).
# 20 + 20 + 20 = 60 per role, matching center-wise train size.
RANDOM_PER_COHORT = {"train": 20, "val": 20, "test": 20}


def load_brain_subjects(source: Path) -> list[str]:
    """Extract all brain subject IDs from the canonical split file.

    Brain subjects start with '1B' (BA, BB, BC sub-cohorts). Returns a
    deterministically sorted list so the random shuffle is reproducible.
    """
    subjects = []
    for line in source.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        sid = parts[1]
        if sid.startswith("1B"):
            subjects.append(sid)
    return sorted(set(subjects))


def cohort(sid: str) -> str:
    """Return the 2-letter sub-cohort code ('BA', 'BB', 'BC')."""
    return sid[1:3].upper()


def make_random_split(subjects: list[str], per_cohort: dict[str, int], seed: int):
    """Stratified random split: each cohort contributes the same count to each role."""
    by_cohort = defaultdict(list)
    for s in subjects:
        by_cohort[cohort(s)].append(s)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for c in sorted(by_cohort):
        pool = sorted(by_cohort[c])
        rng.shuffle(pool)
        n_tr, n_va, n_te = per_cohort["train"], per_cohort["val"], per_cohort["test"]
        if len(pool) < n_tr + n_va + n_te:
            raise ValueError(
                f"Cohort {c} has {len(pool)} subjects, need {n_tr + n_va + n_te}"
            )
        train += pool[:n_tr]
        val += pool[n_tr : n_tr + n_va]
        test += pool[n_tr + n_va : n_tr + n_va + n_te]
    return train, val, test


def make_cohort_split(subjects: list[str], seed: int):
    """train = all BA; val/test = each cohort (BB, BC) split in half."""
    by_cohort = defaultdict(list)
    for s in subjects:
        by_cohort[cohort(s)].append(s)

    rng = random.Random(seed)
    train = sorted(by_cohort["BA"])
    val, test = [], []
    for c in ("BB", "BC"):
        pool = sorted(by_cohort[c])
        rng.shuffle(pool)
        half = len(pool) // 2
        val += pool[:half]
        test += pool[half:]
    return train, val, test


def write_split(path: Path, train: list[str], val: list[str], test: list[str]):
    lines = []
    for s in sorted(train):
        lines.append(f"train {s}")
    for s in sorted(val):
        lines.append(f"val {s}")
    for s in sorted(test):
        lines.append(f"test {s}")
    path.write_text("\n".join(lines) + "\n")


def summarize(name: str, train: list[str], val: list[str], test: list[str]):
    def counts(xs):
        c = defaultdict(int)
        for s in xs:
            c[cohort(s)] += 1
        return dict(sorted(c.items()))

    print(f"\n[{name}]")
    print(f"  train (n={len(train):3d}): {counts(train)}")
    print(f"  val   (n={len(val):3d}): {counts(val)}")
    print(f"  test  (n={len(test):3d}): {counts(test)}")


def main():
    if not SOURCE_SPLIT.is_file():
        raise FileNotFoundError(f"Source split file not found: {SOURCE_SPLIT}")

    brain = load_brain_subjects(SOURCE_SPLIT)
    by_c = defaultdict(list)
    for s in brain:
        by_c[cohort(s)].append(s)
    print(f"Loaded {len(brain)} brain subjects from {SOURCE_SPLIT.name}:")
    for c in sorted(by_c):
        print(f"  {c}: {len(by_c[c])}")

    # Experiment #1
    tr1, va1, te1 = make_random_split(brain, RANDOM_PER_COHORT, SEED)
    out1 = SPLITS_DIR / "brain_random_split.txt"
    write_split(out1, tr1, va1, te1)
    summarize("brain_random_split.txt", tr1, va1, te1)
    print(f"  -> wrote {out1}")

    # Experiment #2
    tr2, va2, te2 = make_cohort_split(brain, SEED)
    out2 = SPLITS_DIR / "brain_center_wise_split.txt"
    write_split(out2, tr2, va2, te2)
    summarize("brain_center_wise_split.txt", tr2, va2, te2)
    print(f"  -> wrote {out2}")


if __name__ == "__main__":
    main()
