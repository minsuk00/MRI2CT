"""Read-only audit: find weird cross-region firing in CADS part files across all 843.

Two views:
  A) per (region, task): how often it fires + voxel stats  -> spots whole-task anomalies
  B) sentinel structures that are anatomically IMPOSSIBLE for a region but fired
     -> spots per-structure leaks the allow-list can't catch (kept multi-region tasks)
Writes nothing.
"""
import os
from collections import defaultdict

import nibabel as nib
import numpy as np

SEG = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/cads/seg"
TASKS = [551, 552, 553, 554, 555, 556, 557, 558, 559]

# sentinel = (src_model, src_index, name): structures that CANNOT be in <region>'s FOV
SENTINELS = {
    "B": [(551, 5, "liver"), (551, 13, "lung"), (553, 17, "bladder"), (553, 16, "colon"),
          (556, 7, "prostate"), (554, 7, "femur"), (554, 9, "hip"), (552, 1, "L5-vert"),
          (555, 1, "rib")],
    "HN": [(551, 5, "liver"), (551, 2, "kidney"), (553, 17, "bladder"), (553, 16, "colon"),
           (556, 7, "prostate"), (554, 7, "femur"), (554, 9, "hip"), (552, 1, "L5-vert")],
    "TH": [(553, 9, "brain"), (553, 18, "face"), (553, 17, "bladder"),
           (556, 7, "prostate"), (554, 7, "femur"), (554, 9, "hip"), (557, 4, "scalp")],
    "AB": [(553, 9, "brain"), (553, 18, "face"), (552, 24, "C1-vert"), (552, 18, "C7-vert"),
           (557, 1, "WM"), (558, 4, "mandible")],
    "P": [(553, 9, "brain"), (551, 13, "lung"), (552, 24, "C1-vert"), (555, 1, "rib"),
          (557, 1, "WM"), (558, 4, "mandible")],
}


def region(s):
    s2 = s[1:]
    for r in ("AB", "HN", "TH", "P", "B"):
        if s2.startswith(r):
            return r
    return "?"


def main():
    subs = sorted(os.listdir(SEG))
    fire = defaultdict(list)          # (region, task) -> nonzero counts
    sent_hits = defaultdict(list)     # (region, model, idx, name) -> [(vox, subj)]
    region_n = defaultdict(int)

    for s in subs:
        r = region(s)
        region_n[r] += 1
        cache = {}
        for t in TASKS:
            f = os.path.join(SEG, s, f"{s}_part_{t}.nii.gz")
            if not os.path.exists(f):
                continue
            a = np.asarray(nib.load(f).dataobj)
            cache[t] = a
            fire[(r, t)].append(int((a > 0).sum()))
        for (sm, si, name) in SENTINELS.get(r, []):
            if sm in cache:
                v = int((cache[sm] == si).sum())
                if v > 0:
                    sent_hits[(r, sm, si, name)].append((v, s))

    print("=" * 70)
    print("A) PER (REGION, TASK) FIRING  [n=subjects in region]")
    print("=" * 70)
    for r in ("AB", "TH", "P", "HN", "B"):
        print(f"\n-- {r}  (n={region_n[r]}) --")
        for t in TASKS:
            v = np.array(fire[(r, t)]) if fire[(r, t)] else np.array([0])
            nfire = int((v > 0).sum())
            print(f"   {t}: fires {nfire:>3d}/{region_n[r]:<3d}  "
                  f"median={int(np.median(v[v>0])) if nfire else 0:>7d}  "
                  f"max={v.max():>8d}")

    print("\n" + "=" * 70)
    print("B) SENTINEL LEAKS  (anatomically impossible structures that fired)")
    print("=" * 70)
    any_leak = False
    for (r, sm, si, name), hits in sorted(sent_hits.items()):
        any_leak = True
        hits.sort(reverse=True)
        vmax, smax = hits[0]
        print(f"   [{r}] {name:9s} (task {sm} idx {si}): "
              f"{len(hits):>3d} subjects, max={vmax} ({smax})")
    if not any_leak:
        print("   none — no impossible structures fired anywhere.")


if __name__ == "__main__":
    main()
