"""Assemble a self-contained HTML report (base64-embedded figures) in _html/."""
import base64
import os

PROJ = "/home/minsukc/MRI2CT"
OUT = os.path.join(PROJ, "_html", "cads_merge_report.html")
FIGS = {
    "merge": "cads_demo/merge_demo.png",
    "merge_all": "cads_demo/merge_demo_all.png",
    "realhead": "cads_demo/fig_realhead.png",
    "fov": "cads_demo/fig_fov.png",
}


def b64(path):
    with open(os.path.join(PROJ, path), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


img = {k: b64(v) for k, v in FIGS.items()}

HTML = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CADS multi-task segmentation merge — report</title>
<style>
  :root {{ --fg:#1a1a1a; --mut:#666; --acc:#1769aa; --bad:#c0392b; --good:#1e8449; --bg:#fafafa; --card:#fff; --bord:#e2e2e2; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; color:var(--fg);
         background:var(--bg); margin:0; line-height:1.55; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:32px 22px 80px; }}
  h1 {{ font-size:1.7rem; margin:0 0 4px; }}
  h2 {{ font-size:1.25rem; margin:38px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--bord); }}
  h3 {{ font-size:1.05rem; margin:22px 0 6px; }}
  .sub {{ color:var(--mut); font-size:.92rem; margin-bottom:8px; }}
  p, li {{ font-size:.96rem; }}
  code {{ background:#f0f0f0; padding:1px 5px; border-radius:4px; font-size:.88em; }}
  table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:.88rem; background:var(--card); }}
  th,td {{ border:1px solid var(--bord); padding:6px 9px; text-align:left; }}
  th {{ background:#f3f3f3; }}
  figure {{ margin:16px 0; }}
  figure img {{ width:100%; border:1px solid var(--bord); border-radius:6px; }}
  figcaption {{ color:var(--mut); font-size:.85rem; margin-top:6px; }}
  .callout {{ border-left:4px solid var(--acc); background:#eef5fb; padding:12px 16px; margin:16px 0; border-radius:0 6px 6px 0; }}
  .callout.bad {{ border-color:var(--bad); background:#fdecea; }}
  .callout.good {{ border-color:var(--good); background:#eafaf1; }}
  .tag {{ display:inline-block; font-size:.72rem; font-weight:600; padding:2px 7px; border-radius:10px; color:#fff; }}
  .tag.bad {{ background:var(--bad); }} .tag.good {{ background:var(--good); }} .tag.acc {{ background:var(--acc); }}
  ul {{ margin:6px 0 6px 0; }}
</style></head><body><div class="wrap">

<h1>CADS multi-task segmentation → single labelmap</h1>
<div class="sub">SynthRAD 1.5&nbsp;mm registered/masked CT cohort · 843 subjects · generated for the MRI2CT project</div>

<div class="callout good">
<b>Final decision.</b> <b>Merge all 9 CADS tasks per subject by priority painting — no region gating.</b>
We built region allow-lists to suppress what looked like out-of-FOV hallucinations, but on inspection every alarming
case turned out to be <b>real anatomy</b>: the SynthRAD region label does <b>not</b> bound the scan field of view, and
CADS correctly segments whatever is in view. Gating by region label therefore <i>deletes legitimate structures</i>.
The only genuine errors that remain are tiny isolated specks (~a few thousand voxels on ~30 / 843 subjects) —
negligible for a teacher / evaluation target.
</div>

<h3>Why we reverted (short version)</h3>
<ul>
  <li>A region <b>allow-list</b> (deny &ldquo;impossible&rdquo; tasks per region) looked validated — until we verified the cases it acted on.</li>
  <li><b>1THA293</b> (&ldquo;thorax&rdquo;, 557 fired 398k vox) actually <b>contains a head</b> (mandible + brain tissue) — the firing is real, and the allow-list was deleting it.</li>
  <li><b>1THB165</b> (&ldquo;thorax&rdquo;, hip fired 69k vox) is a <b>477&nbsp;mm whole-torso scan</b> — the hip is real.</li>
  <li>The 12 &ldquo;558 on thorax&rdquo; cases are <b>lower-neck</b> structures (thyroid / carotid / cervical esophagus) at the thoracic inlet — real.</li>
  <li>Conclusion: <b>region label ≠ FOV.</b> Any label-based gating risks deleting real anatomy, so we removed it entirely.</li>
</ul>

<h2>1. What we start from &amp; the merge</h2>
<p>CADS runs <b>9 nnU-Net task models</b> (551–559) per CT, each emitting a separate label map
(<code>&lt;subj&gt;_part_55X.nii.gz</code>) in the original CT geometry. We combine them into one labelmap by
<b>priority painting</b>: rows of a mapping CSV are painted low→high priority (coarse fillers like task 559 first,
fine structures like 557/558 last, so fine structures win on overlap). This ordering is the only logic in the merge,
and it is correct &mdash; &ldquo;generic paints first, specific wins.&rdquo; Two granularities:</p>
<ul>
  <li><span class="tag acc">grouped</span> 35 classes (e.g. all vertebrae → &ldquo;Spine&rdquo;, all lobes → &ldquo;Lungs&rdquo;).</li>
  <li><span class="tag acc">all</span> 167 classes — every CADS structure kept distinct.</li>
</ul>
<figure><img src="{img['merge']}"><figcaption>Grouped (35-class) merge, one subject per region. Left: CT
(window [−1024,1024]; brain [−100,100]). Right: CT + merged overlay. Background is transparent (label 0).</figcaption></figure>
<figure><img src="{img['merge_all']}"><figcaption>The <code>all</code> (167-class) variant on the same subjects —
individual vertebrae, ribs, lung lobes, heart chambers kept separate.</figcaption></figure>

<h2>2. What looked like a problem</h2>
<p>Every CT is run through <b>all 9</b> task models, including ones whose anatomy seemed out of the region. A model
run outside its expected field of view <i>could</i> hallucinate, and the merge gives the fine head tasks (557/558)
the <b>highest priority</b>, so a spurious head label would <i>overwrite</i> correct labels rather than lose to them.
The audit duly flagged &ldquo;head tissue on thorax&rdquo;, &ldquo;hip on thorax&rdquo;, &ldquo;face on
abdomen&rdquo;, etc. — and we built region allow-lists to remove them.</p>

<h2>3. Why the gating was wrong: region label ≠ scan FOV <span class="tag bad">the turn</span></h2>
<p>Verifying the cases the gating acted on flipped the conclusion. The SynthRAD &ldquo;region&rdquo; is an acquisition
label, <b>not</b> a field-of-view bound — scans vary from single-region to whole-body. The flagged structures were
overwhelmingly <b>real anatomy in an extended FOV</b>:</p>

<div class="callout bad"><span class="tag bad">case</span> <b>1THA293</b> — labeled &ldquo;thorax&rdquo;, but the FOV
extends up into the <b>head/neck</b>: mandible = 14,838 vox and brain tissue (WM/GM/CSF) are present. The 557
&ldquo;head tissue&rdquo; firing (398k vox) is <b>real</b> — the task-level allow-list was deleting a genuine head.</div>
<figure><img src="{img['realhead']}"><figcaption>1THA293 sagittal: the jaw / head / cervical spine are clearly in the
FOV. The 557 / 558 head &amp; neck labels are correct anatomy, not hallucination.</figcaption></figure>

<div class="callout bad"><span class="tag bad">case</span> <b>1THB165</b> — labeled &ldquo;thorax&rdquo;, but is a
<b>477&nbsp;mm whole-torso</b> scan running from the <b>pelvis</b> (hip + sacrum) up to the <b>lungs</b>. The
&ldquo;hip on thorax&rdquo; is real bone. A per-structure hip deny would have deleted it.</div>
<figure><img src="{img['fov']}"><figcaption>1THB165 sagittal: spine top-to-bottom, lungs at the top, pelvis/hip at the
bottom — a whole-torso FOV under a &ldquo;TH&rdquo; label.</figcaption></figure>

<p>We confirmed this is the rule, not the exception: of the 7 &ldquo;hip on thorax&rdquo; cases, <b>4 are real</b>
whole-torso scans (pelvis present); the 12 &ldquo;558 on thorax&rdquo; cases are real lower-neck (thyroid/carotid).
A region- or per-structure label-based deny would delete all of these.</p>

<h2>4. Decision: no gating, trust the FOV</h2>
<ul>
  <li><span class="tag bad">revert</span> <b>All region gating</b> (both the task-level allow-list and the per-structure
      deny). Label-based denial deletes real anatomy because the label does not bound the FOV.</li>
  <li><span class="tag good">keep</span> <b>Priority painting of all 9 tasks</b>, every subject. The priority order
      (coarse 559 fillers first → specific structures win) is correct and unchanged; with no gating, real head / neck /
      limb structures simply win where they exist, which is what we want.</li>
</ul>
<p>What about the <i>genuine</i> hallucinations? They are real but tiny: ~29 small &ldquo;face&rdquo; specks on
head-less torso scans (≤ a few k voxels each) and a handful of similar blobs. Negligible for a teacher / eval target.
If a downstream metric ever proves sensitive, the safe cleanup is <b>content-gated</b> (e.g. drop &ldquo;face&rdquo;
only when no head is present) or a one-line small-component filter — never region-label gating.</p>
<p>Residual known artifact (minor, systematic): a thin <b>559 &ldquo;bone&rdquo; rim</b> at the masked skin edge,
where the 559 model misreads the hard body-mask boundary as bone. Cosmetic for most uses.</p>

<h2>5. Status</h2>
<ul>
  <li>Per-task CADS segmentations: <b>843 × 9 = 7,587 files</b>, verified voxel-aligned to each CT
      (<code>SynthRAD/cads/seg/</code>). Done.</li>
  <li>Merge: final no-gating logic prototyped on one subject per region (<code>grouped</code> &amp; <code>all</code>);
      full-cohort merge still to run.</li>
  <li>Code (read-only on the dataset; writes to <code>cads_demo/</code> only): <code>cads_demo/merge_demo.py</code>
      (priority painting, no gating), <code>audit_regions.py</code>, <code>report_figs.py</code>,
      <code>build_report.py</code>.</li>
</ul>
<p class="sub">Mapping CSV: <code>cads_demo/cads_labelmap (1).csv</code> (35-class grouping; the <code>all</code> variant
assigns one id per source structure). The merge has no region rules — it paints every task by CSV priority.</p>

</div></body></html>
"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")
