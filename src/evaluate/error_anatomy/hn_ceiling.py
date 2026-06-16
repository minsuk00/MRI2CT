"""HN ceiling: run the SAME teacher on the GROUND-TRUTH CT (clipped [-1024,1024]) and
compute per-class Dice vs the GT seg. This is the best Dice achievable in HN given the
teacher itself (structure difficulty), independent of sCT quality. Comparing sCT-Dice
(hn_dice.csv) to this ceiling separates 'sCT is bad in HN' from 'these structures are
intrinsically hard / the macro-average penalises many tiny classes'.
"""
import os, sys, numpy as np, nibabel as nib, pandas as pd, torch
sys.path.insert(0, "/home/minsukc/MRI2CT/src")
from common.eval_utils import load_teacher_model, run_teacher_sw  # noqa

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
RUN = os.path.join(REPO, "evaluation_results/unet_error_analysis_20260616")
TEACHER = os.path.join(REPO, "ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth")
PS = os.path.join(REPO, "evaluation_results/full_eval_20260609/metrics/per_subject.csv")


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def main():
    dev = torch.device("cuda")
    teacher = load_teacher_model(TEACHER, device=dev, n_classes_minus_bg=11)
    df = pd.read_csv(PS)
    subs = df[(df.model == "unet") & (df.region == "head_neck")].subj_id.tolist()
    rows = []
    for sid in subs:
        segp = os.path.join(DATA, sid, "ct_seg.nii")
        try:
            gt = canon(os.path.join(DATA, sid, "ct.nii"))
            seg = canon(segp, np.int64)
        except Exception:
            continue
        if gt.shape != seg.shape:
            continue
        g01 = (np.clip(gt, -1024, 1024) + 1024) / 2048.0
        gtt = torch.from_numpy(g01)[None, None].to(dev, torch.float32)
        logits = run_teacher_sw(teacher, gtt, device=dev)
        nC = logits.shape[1]
        lab = logits.argmax(1)[0].cpu().numpy()
        rec = {"subj_id": sid}
        for c in range(1, nC):
            a = lab == c; b = seg == c
            asum, bsum = int(a.sum()), int(b.sum())
            if asum == 0 and bsum == 0:
                d = np.nan  # absent in GT: ignore for "present" ceiling
            elif asum == 0 or bsum == 0:
                d = 0.0
            else:
                d = 2.0 * int((a & b).sum()) / (asum + bsum)
            rec[f"gtceil_c{c}"] = d
        rows.append(rec)
        del logits, gtt; torch.cuda.empty_cache()
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RUN, "hn_ceiling.csv"), index=False)

    pred = pd.read_csv(os.path.join(RUN, "hn_dice.csv"))
    pred = pred[pred.region == "head_neck"]
    nC = int(pred.n_classes.iloc[0])
    print("class | sCT dice|present | GT-CT ceiling|present | gap")
    for c in range(1, nC):
        present = pred[f"gtpresent_c{c}"] == 1
        sct = pred.loc[present, f"dice_c{c}"].mean() if present.any() else np.nan
        ceil = d[f"gtceil_c{c}"].mean()  # nan-skipping
        print(f"  c{c:<2}  sCT={sct:5.3f}   GTceil={ceil:5.3f}   gap={ceil-sct:+.3f}")
    # macro over present classes
    print("\nmacro (present-class mean): sCT vs GT-ceiling tells if sCT is the problem")


if __name__ == "__main__":
    main()
