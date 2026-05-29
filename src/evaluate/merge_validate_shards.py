"""Merge sharded validate_metrics.txt files from cWDM/MC-DDPM/MAISI runs.

Each baseline's standalone validator (`baselines/cwdm/scripts/validate.py`,
`baselines/mc_ddpm/scripts/validate.py`, `src/maisi_baseline/validate.py`)
emits one `validate_metrics.txt` per SLURM array shard. This script glues
them into:

  - `<out>/<model_name>/validate_metrics_combined.txt`
        One TXT per model with all 207 per-subject rows + aggregate + timing.
        Format identical to eval_utils.write_metrics_txt.

  - `<out>/comparison_table.txt`
        Side-by-side table across all passed models, aliasing the MC-DDPM
        `amix/*` keys to the plain keys used by MAISI/cWDM so the columns
        compare apples-to-apples (amix-clipped HU range).

  - `<out>/comparison_table.tsv`
        Same data, TSV for spreadsheet paste.

Usage:
    python src/evaluate/merge_validate_shards.py \
        --model maisi:/path/to/maisi/validate_metrics.txt \
        --model cwdm:'/path/to/cwdm/shard_*/validate_metrics.txt' \
        --model mcddpm:'/path/to/mcddpm/shard_*/validate_metrics.txt' \
        --out evaluation_results/three_baselines_<TS>/

Globs are accepted; quote them so the shell doesn't expand to a single path.
"""
import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from common.eval_utils import write_metrics_txt  # noqa: E402


# All baselines now report on the amix [-1024, 1024] HU yardstick with plain
# key names, so the alias map is effectively a no-op identity — kept here
# only for the MAISI-only `mae_hu_air_excluded` column.
KEY_ALIASES = {
    "mae_hu":              ["mae_hu"],
    "psnr":                ["psnr"],
    "ssim":                ["ssim"],
    "grad_diff":           ["grad_diff"],
    "body_mae_hu":         ["body_mae_hu"],
    "body_psnr":           ["body_psnr"],
    "body_ssim":           ["body_ssim"],
    "dice_score_all":      ["dice_score_all"],
    "dice_score_bone":     ["dice_score_bone"],
    "body_dice_score_all": ["body_dice_score_all"],
    "body_dice_score_bone":["body_dice_score_bone"],
    "mae_hu_air_excluded": ["mae_hu_air_excluded"],  # MAISI-only
    "time_sec":            ["time_sec"],
}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_validate_txt(path):
    """Return (header_lines, per_subject_dict). Per-subject dict: subj_id → {key: float}."""
    with open(path) as f:
        lines = f.readlines()
    header_lines = []
    for l in lines:
        if l.startswith("# "):
            header_lines.append(l[2:].rstrip("\n"))
        else:
            break

    # Find the subj_id header
    try:
        hi = next(i for i, l in enumerate(lines) if l.startswith("subj_id"))
    except StopIteration:
        raise RuntimeError(f"No 'subj_id' header row in {path}")
    columns = lines[hi].split()[1:]

    rows = {}
    for l in lines[hi + 1:]:
        l = l.strip()
        if not l or l.startswith("="):
            break
        parts = l.split()
        if len(parts) < 2:
            continue
        subj = parts[0]
        vals = {}
        for col, raw in zip(columns, parts[1:]):
            try:
                vals[col] = float(raw)
            except ValueError:
                pass
        rows[subj] = vals
    return header_lines, rows


def expand_paths(spec):
    """Accept a path, a glob, or a comma-separated list of either."""
    out = []
    for part in spec.split(","):
        matched = sorted(glob.glob(part))
        if matched:
            out.extend(matched)
        elif os.path.exists(part):
            out.append(part)
        else:
            raise FileNotFoundError(f"No files match: {part}")
    return out


# ---------------------------------------------------------------------------
# Per-model combined output
# ---------------------------------------------------------------------------
def merge_model(name, paths, out_dir):
    """Combine multiple shard TXTs into one combined TXT for this model."""
    if not paths:
        raise ValueError(f"No paths for model {name}")
    combined = {}
    headers_first_shard = None
    for p in paths:
        hdr, rows = parse_validate_txt(p)
        if headers_first_shard is None:
            headers_first_shard = hdr
        overlap = set(combined) & set(rows)
        if overlap:
            print(f"⚠️  {name}: subjects appear in multiple shards (e.g. {sorted(overlap)[:3]}); using last occurrence")
        combined.update(rows)

    # Determine metric key order: union across all shards, in first-seen order
    metric_keys = []
    for p in paths:
        _, rows = parse_validate_txt(p)
        for r in rows.values():
            for k in r:
                if k not in metric_keys:
                    metric_keys.append(k)

    per_subject = [
        {"subj_id": subj, "metrics": vals}
        for subj, vals in sorted(combined.items(),
                                  key=lambda kv: kv[1].get("psnr", float("-inf")),
                                  reverse=True)
    ]

    model_dir = os.path.join(out_dir, name)
    os.makedirs(model_dir, exist_ok=True)

    # Carry forward header lines from the first shard, plus a note about the merge
    new_header = list(headers_first_shard or [])
    new_header.append(f"merged from: {len(paths)} shard file(s)")
    new_header.append(f"merged total subjects: {len(per_subject)}")

    out_path = os.path.join(model_dir, "validate_metrics_combined.txt")
    write_metrics_txt(out_path,
                      header_lines=new_header,
                      per_subject=per_subject,
                      metric_keys=metric_keys)
    return per_subject, metric_keys


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------
def _resolve_alias(metric, model_rows):
    """For a canonical metric, find the first matching source key present in
    any row of `model_rows`. Returns the source key, or None."""
    for src in KEY_ALIASES.get(metric, [metric]):
        if any(src in r["metrics"] for r in model_rows):
            return src
    return None


def write_comparison(per_model, out_dir):
    """Build comparison_table.txt and .tsv across all models."""
    models = list(per_model.keys())

    # Canonical metrics to include — only those present in at least one model
    canonical = [m for m in KEY_ALIASES if any(
        _resolve_alias(m, per_model[mdl]) is not None for mdl in models
    )]

    # Compute mean/std per (model, canonical_metric)
    cells = {}  # (model, canonical) → (mean, std, n)
    for mdl in models:
        rows = per_model[mdl]
        for can in canonical:
            src = _resolve_alias(can, rows)
            if src is None:
                cells[(mdl, can)] = None
                continue
            vals = [r["metrics"][src] for r in rows if src in r["metrics"]]
            if not vals:
                cells[(mdl, can)] = None
                continue
            arr = np.array(vals, dtype=np.float64)
            cells[(mdl, can)] = (float(arr.mean()), float(arr.std()), int(arr.size))

    # Pretty TXT
    txt_path = os.path.join(out_dir, "comparison_table.txt")
    col_w = 24
    with open(txt_path, "w") as f:
        f.write(f"# Cross-model comparison ({datetime.now().isoformat(timespec='seconds')})\n")
        f.write(f"# n per model:\n")
        for mdl in models:
            f.write(f"#   {mdl}: {len(per_model[mdl])}\n")
        f.write("# All models report on the amix [-1024, 1024] HU yardstick (span 2048).\n")
        f.write("\n")
        f.write(f"{'metric':<22}" + "".join(f"{m:>{col_w}}" for m in models) + "\n")
        f.write("-" * (22 + col_w * len(models)) + "\n")
        for can in canonical:
            line = f"{can:<22}"
            for mdl in models:
                c = cells[(mdl, can)]
                if c is None:
                    line += f"{'—':>{col_w}}"
                else:
                    mean, std, _ = c
                    line += f"{mean:>10.4f} ± {std:>8.4f}".rjust(col_w)
            f.write(line + "\n")
    print(f"📊 Wrote {txt_path}")

    # TSV
    tsv_path = os.path.join(out_dir, "comparison_table.tsv")
    with open(tsv_path, "w") as f:
        f.write("metric\t" + "\t".join(f"{m}_mean\t{m}_std\t{m}_n" for m in models) + "\n")
        for can in canonical:
            row = [can]
            for mdl in models:
                c = cells[(mdl, can)]
                if c is None:
                    row += ["", "", ""]
                else:
                    mean, std, n = c
                    row += [f"{mean:.6f}", f"{std:.6f}", str(n)]
            f.write("\t".join(row) + "\n")
    print(f"📊 Wrote {tsv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME:PATH[,PATH...]",
                    help="Model spec. PATH can be a single file or a quoted glob "
                         "(e.g. cwdm:'/path/to/shard_*/validate_metrics.txt'). "
                         "Repeat --model for each baseline.")
    ap.add_argument("--out", required=True, help="Output directory; created if missing.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[merge] output dir: {args.out}")

    per_model = {}
    for spec in args.model:
        if ":" not in spec:
            raise SystemExit(f"--model spec must be NAME:PATH (got {spec!r})")
        name, path_part = spec.split(":", 1)
        paths = expand_paths(path_part)
        print(f"[merge] {name}: {len(paths)} file(s)")
        for p in paths:
            print(f"   - {p}")
        rows, _ = merge_model(name, paths, args.out)
        per_model[name] = rows

    write_comparison(per_model, args.out)
    print(f"[merge] ✅ done")


if __name__ == "__main__":
    main()
