# CADS multi-task segmentation merge into a single labelmap

**Source HTML:** _html/cads_merge_report.html
**Date:** undated
**TL;DR:** Merge all 9 CADS task models per subject by priority painting with NO region gating. Region allow-lists were reverted because the SynthRAD region label does not bound the scan FOV: every "out-of-region hallucination" the gating acted on turned out to be real anatomy in an extended FOV, so label-based gating deletes legitimate structures. Only genuine errors are tiny isolated specks (~few thousand voxels on ~30/843 subjects), negligible for a teacher/eval target.

SynthRAD 1.5 mm registered/masked CT cohort, 843 subjects.

## Why we reverted (short version)

- A region allow-list (deny "impossible" tasks per region) looked validated until the cases it acted on were verified.
- 1THA293 ("thorax", task 557 fired 398k vox) actually contains a head (mandible + brain tissue): the firing is real, the allow-list was deleting it.
- 1THB165 ("thorax", hip fired 69k vox) is a 477 mm whole-torso scan: the hip is real.
- The 12 "558 on thorax" cases are lower-neck structures (thyroid / carotid / cervical esophagus) at the thoracic inlet: real.
- Conclusion: region label != FOV. Any label-based gating risks deleting real anatomy, so it was removed entirely.

## 1. Start point and the merge

CADS runs 9 nnU-Net task models (551-559) per CT, each emitting a separate label map (`<subj>_part_55X.nii.gz`) in the original CT geometry. They are combined into one labelmap by priority painting: rows of a mapping CSV are painted low->high priority (coarse fillers like task 559 first, fine structures like 557/558 last, so fine structures win on overlap). This ordering is the only logic in the merge ("generic paints first, specific wins"). Two granularities:

- grouped 35 classes (e.g. all vertebrae -> "Spine", all lobes -> "Lungs").
- all 167 classes: every CADS structure kept distinct.

## 2. What looked like a problem

Every CT is run through all 9 task models, including ones whose anatomy seemed out of region. A model run outside its expected FOV could hallucinate, and the merge gives fine head tasks (557/558) the highest priority, so a spurious head label would overwrite correct labels rather than lose to them. The audit flagged "head tissue on thorax", "hip on thorax", "face on abdomen", and region allow-lists were built to remove them.

## 3. Why the gating was wrong: region label != scan FOV

The SynthRAD "region" is an acquisition label, not a FOV bound (scans vary from single-region to whole-body). Flagged structures were overwhelmingly real anatomy in an extended FOV:

- 1THA293: labeled "thorax", but FOV extends up into head/neck: mandible = 14,838 vox and brain tissue (WM/GM/CSF) present. The 557 "head tissue" firing (398k vox) is real.
- 1THB165: labeled "thorax", but a 477 mm whole-torso scan from pelvis (hip + sacrum) up to lungs. The "hip on thorax" is real bone.

Confirmed as the rule, not exception: of 7 "hip on thorax" cases, 4 are real whole-torso scans (pelvis present); the 12 "558 on thorax" cases are real lower-neck (thyroid/carotid). Label-based deny would delete all of these.

## 4. Decision: no gating, trust the FOV

- Revert all region gating (both task-level allow-list and per-structure deny). Label-based denial deletes real anatomy because the label does not bound the FOV.
- Keep priority painting of all 9 tasks, every subject. The priority order (coarse 559 fillers first -> specific structures win) is correct and unchanged.

Genuine hallucinations are real but tiny: ~29 small "face" specks on head-less torso scans (<= a few k voxels each) and a handful of similar blobs. Negligible for a teacher / eval target. If a downstream metric proves sensitive, the safe cleanup is content-gated (e.g. drop "face" only when no head present) or a one-line small-component filter, never region-label gating.

Residual known artifact (minor, systematic): a thin 559 "bone" rim at the masked skin edge, where the 559 model misreads the hard body-mask boundary as bone. Cosmetic for most uses.

## 5. Status

- Per-task CADS segmentations: 843 x 9 = 7,587 files, verified voxel-aligned to each CT (`SynthRAD/cads/seg/`). Done.
- Merge: final no-gating logic prototyped on one subject per region (grouped & all); full-cohort merge still to run.
- Code (read-only on dataset; writes to `cads_demo/` only): `cads_demo/merge_demo.py` (priority painting, no gating), `audit_regions.py`, `report_figs.py`, `build_report.py`.
- Mapping CSV: `cads_demo/cads_labelmap (1).csv` (35-class grouping; the all variant assigns one id per source structure). The merge has no region rules: it paints every task by CSV priority.
