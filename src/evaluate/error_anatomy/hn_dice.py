"""Diagnose the head&neck organ-Dice collapse (dice_score_all=0.47 vs abdomen 0.83).

Replicates the eval's EXACT teacher (Baby U-Net epoch_749, 12-class) + compute_dice_hard
convention, but returns PER-CLASS Dice (not just the macro mean) for HN vs a contrast
region (abdomen), plus per-class GT presence. Shows whether the low macro-Dice is driven
by genuinely poor classes or by rare/out-of-FOV classes penalised by the absent-class rule.
"""
import os, sys, numpy as np, nibabel as nib, pandas as pd, torch, json
sys.path.insert(0, os.path.join("/home/minsukc/MRI2CT", "src"))
from common.eval_utils import load_teacher_model, run_teacher_sw  # noqa

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
VOL = os.path.join(REPO, "evaluation_results/full_eval_20260609/volumes/unet")
RUN = os.path.join(REPO, "evaluation_results/unet_error_analysis_20260616")
TEACHER = os.path.join(REPO, "ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth")
PS = os.path.join(REPO, "evaluation_results/full_eval_20260609/metrics/per_subject.csv")
# 12-class teacher label names (Background + 11 fg), from ct_seg.nii scheme
NAMES = ["bg", "c1", "c2", "c3", "c4", "bone(c5)", "c6", "c7", "c8", "c9", "c10", "c11"]


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def per_class_dice(pred_lab, seg, n_classes):
    out = []
    for c in range(1, n_classes):
        a = pred_lab == c
        b = seg == c
        asum, bsum = int(a.sum()), int(b.sum())
        if asum == 0 and bsum == 0:
            d = 1.0
        elif asum == 0 or bsum == 0:
            d = 0.0
        else:
            d = 2.0 * int((a & b).sum()) / (asum + bsum)
        out.append((d, asum, bsum))
    return out


def main():
    dev = torch.device("cuda")
    teacher = load_teacher_model(TEACHER, device=dev, n_classes_minus_bg=11)
    df = pd.read_csv(PS)
    regions = {"head_neck": None, "abdomen": None, "brain": None}
    rows = []
    for reg in regions:
        subs = df[(df.model == "unet") & (df.region == reg)].subj_id.tolist()
        for i, sid in enumerate(subs):
            segp = os.path.join(DATA, sid, "ct_seg.nii")
            if not os.path.exists(segp):
                continue
            try:
                pred = canon(os.path.join(VOL, sid, "sample.nii.gz"))
                seg = canon(segp, np.int64)
            except Exception:
                continue
            if pred.shape != seg.shape:
                continue
            p01 = (np.clip(pred, -1024, 1024) + 1024) / 2048.0
            pt = torch.from_numpy(p01)[None, None].to(dev, torch.float32)
            logits = run_teacher_sw(teacher, pt, device=dev)
            nC = logits.shape[1]
            lab = logits.argmax(1)[0].cpu().numpy()
            pcd = per_class_dice(lab, seg, nC)
            rec = {"subj_id": sid, "region": reg, "n_classes": nC,
                   "dice_all": float(np.mean([d for d, _, _ in pcd]))}
            for c, (d, asum, bsum) in enumerate(pcd, start=1):
                rec[f"dice_c{c}"] = d
                rec[f"gtpresent_c{c}"] = int(bsum > 50)
                rec[f"gtvox_c{c}"] = bsum
            rows.append(rec)
            del logits, pt
            torch.cuda.empty_cache()
        print(f"[{reg}] done {len([r for r in rows if r['region']==reg])} subjects", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RUN, "hn_dice.csv"), index=False)
    nC = int(d.n_classes.iloc[0])
    print("teacher n_classes (incl bg):", nC, "-> foreground classes 1..", nC - 1)
    print("\nmacro dice_all by region:", d.groupby("region").dice_all.mean().round(3).to_dict())

    # per-class: mean dice (only over subjects where GT present) + presence rate
    summary = {}
    for reg in regions:
        sub = d[d.region == reg]
        print(f"\n=== {reg} (n={len(sub)}) per-class ===")
        for c in range(1, nC):
            present = sub[f"gtpresent_c{c}"] == 1
            pres_rate = present.mean()
            dice_present = sub.loc[present, f"dice_c{c}"].mean() if present.any() else float("nan")
            dice_macro = sub[f"dice_c{c}"].mean()  # incl. absent-class 1.0/0.0 contributions
            medvox = int(sub.loc[present, f"gtvox_c{c}"].median()) if present.any() else 0
            print(f"  c{c:<2} present={pres_rate:4.0%}  dice|present={dice_present:5.3f}  "
                  f"dice_macro={dice_macro:5.3f}  med_vox={medvox}")
            summary[f"{reg}_c{c}"] = dict(present=round(float(pres_rate), 2),
                                          dice_present=None if np.isnan(dice_present) else round(float(dice_present), 3),
                                          dice_macro=round(float(dice_macro), 3), med_vox=medvox)
    json.dump(summary, open(os.path.join(RUN, "hn_dice_stats.json"), "w"), indent=2)
    print("\nwrote hn_dice.csv + hn_dice_stats.json")


if __name__ == "__main__":
    main()
