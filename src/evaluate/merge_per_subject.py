"""Splice reused per-subject scores from a prior eval into this eval, then re-aggregate.

When several models' predictions are byte-identical to a previous full_eval (symlinked
raw) and the scoring math is unchanged, re-scoring them is wasted GPU. This reads the
already-scored rows for those models from a prior per_subject.csv, concatenates them
with the freshly-scored rows in THIS eval's per_subject.csv, writes the merged
per_subject.csv, and re-runs the canonical aggregation (by_region/overall.csv).

Usage:
    python src/evaluate/merge_per_subject.py --eval_root /gpfs/.../full_eval_20260624 \
        --reuse_csv /gpfs/.../full_eval_20260617/metrics/per_subject.csv \
        --reuse_models amix unet mcddpm cwdm koalAI
"""
import argparse
import csv
import os

from score_all_models import aggregate  # noqa: E402  (re-use the canonical roll-up)

# per_subject.csv non-metric columns (everything else is a float metric or empty).
ID_COLS = ("model", "subj_id", "region")


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f)), csv.DictReader(open(path)).fieldnames


def to_typed(row):
    """CSV strings -> the float/None shape aggregate() expects (nan for blanks)."""
    out = {}
    for k, v in row.items():
        if k in ID_COLS:
            out[k] = v
        else:
            out[k] = float(v) if v not in (None, "") else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--reuse_csv", required=True, help="prior per_subject.csv to lift rows from")
    ap.add_argument("--reuse_models", nargs="+", required=True)
    args = ap.parse_args()

    M = os.path.join(args.eval_root, "metrics")
    new_path = os.path.join(M, "per_subject.csv")
    with open(new_path) as f:
        new_rows = list(csv.DictReader(f))
        header = csv.DictReader(open(new_path)).fieldnames
    new_models = sorted({r["model"] for r in new_rows})
    print(f"[merge] freshly scored in this eval: {new_models} ({len(new_rows)} rows)")

    with open(args.reuse_csv) as f:
        reuse_rows = [r for r in csv.DictReader(f) if r["model"] in args.reuse_models]
    got = sorted({r["model"] for r in reuse_rows})
    print(f"[merge] reused from {args.reuse_csv}: {got} ({len(reuse_rows)} rows)")
    missing = set(args.reuse_models) - set(got)
    if missing:
        raise SystemExit(f"[merge] ERROR reuse models absent in reuse_csv: {sorted(missing)}")
    overlap = set(new_models) & set(args.reuse_models)
    if overlap:
        raise SystemExit(f"[merge] ERROR model scored fresh AND reused (ambiguous): {sorted(overlap)}")

    merged = new_rows + reuse_rows
    with open(new_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(merged)
    print(f"[merge] wrote {len(merged)} rows ({sorted({r['model'] for r in merged})}) -> {new_path}")

    aggregate([to_typed(r) for r in merged], M)
    print("[merge] re-aggregated by_region.csv + overall.csv")


if __name__ == "__main__":
    main()
