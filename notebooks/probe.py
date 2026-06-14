"""Comprehensive proxy suite: WHY does Anatomix fail to help MR->CT?
Runs Tier-0/1 probes on real data, no translator training.
E1 HU-regression | E2 anatomy informativeness | E3 bone/air void | E4 cross-modal | E5 redundancy/spectral
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("/home/minsukc/MRI2CT/src"))
sys.path.append(os.path.abspath("/home/minsukc/MRI2CT"))
import numpy as np, nibabel as nib, torch
from scipy import ndimage
from anatomix.model.network import Unet
from common.utils import clean_state_dict

ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT = "/home/minsukc/MRI2CT/splits/center_wise_split.txt"
CK = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
DEV = "cuda"
PATCH = 128
PATCHES_PER_SUBJ = 5
VOX_PER_PATCH = 12000
rng = np.random.default_rng(0)

def region_of(s):
    import re
    m = re.match(r"1([A-Z]+)", s); p = m.group(1)
    return {"AB":"abdomen","B":"brain","HN":"headneck","P":"pelvis","TH":"thorax"}.get(p[:2] if p[:2] in ("AB","HN","TH") else p[:1], p)

def pick_subjects():
    rows = [l.split() for l in open(SPLIT) if l.strip()]
    by = {"train": [], "test": []}
    for split, sid in rows:
        if split in by: by[split].append(sid)
    # balance across regions for in-dist(train) and ood(test)
    def balance(lst, k_per_region=3):
        out = {}
        for sid in lst:
            r = region_of(sid); out.setdefault(r, []).append(sid)
        sel = []
        for r, v in out.items():
            sel += list(rng.choice(v, min(k_per_region, len(v)), replace=False))
        return sel
    return balance(by["train"], 3), balance(by["test"], 3)

def load_subj(sid):
    d = os.path.join(ROOT, sid)
    ct = nib.load(d+"/ct.nii").get_fdata().astype(np.float32)
    mr = nib.load(d+"/moved_mr.nii").get_fdata().astype(np.float32)
    seg = nib.load(d+"/ct_seg.nii").get_fdata().astype(np.int16)
    mask = (nib.load(d+"/mask.nii").get_fdata() > 0).astype(np.uint8)
    mrn = (mr - mr.min()) / (mr.max() - mr.min() + 1e-8)
    ctn = (np.clip(ct, -1024, 1024) + 1024) / 2048.0  # [0,1], HU = v*2048-1024
    return mrn, ctn, ct, seg, mask

def mr_context(mrn):
    """Fair contextual MR baseline: multiscale gaussian + grad mag (5-dim)."""
    feats = [mrn,
             ndimage.gaussian_filter(mrn, 1.0),
             ndimage.gaussian_filter(mrn, 2.0),
             ndimage.gaussian_filter(mrn, 4.0)]
    gx, gy, gz = np.gradient(ndimage.gaussian_filter(mrn, 1.0))
    feats.append(np.sqrt(gx**2+gy**2+gz**2))
    return np.stack(feats, -1).astype(np.float32)  # (..,5)

def lap_var(v):
    return float(np.var(ndimage.laplace(v)))

def extract(net, subjects, split_tag):
    samples = []  # dict per voxel batch
    sharp = {"phi": [], "mr": []}
    for si, sid in enumerate(subjects):
        try:
            mrn, ctn, ct_hu, seg, mask = load_subj(sid)
        except Exception as e:
            print("  skip", sid, e); continue
        mrctx = mr_context(mrn)
        zz, yy, xx = np.where(mask > 0)
        if len(zz) < 5000: continue
        for _ in range(PATCHES_PER_SUBJ):
            i = rng.integers(len(zz)); c = [zz[i], yy[i], xx[i]]
            sl = tuple(slice(int(np.clip(ci-PATCH//2, 0, dim-PATCH)),
                             int(np.clip(ci-PATCH//2, 0, dim-PATCH))+PATCH)
                       for ci, dim in zip(c, mrn.shape))
            mp, cp = mrn[sl], ctn[sl]
            segp, mkp, ctxp, hup = seg[sl], mask[sl], mrctx[sl], ct_hu[sl]
            if mp.shape != (PATCH, PATCH, PATCH): continue
            with torch.no_grad():
                tm = torch.tensor(mp[None, None], device=DEV)
                tc = torch.tensor(cp[None, None], device=DEV)
                phi_mr = net(tm)[0].float().cpu().numpy()  # (16,P,P,P)
                phi_ct = net(tc)[0].float().cpu().numpy()
            sharp["phi"].append(np.mean([lap_var(phi_mr[k]) for k in range(16)]))
            sharp["mr"].append(lap_var(mp))
            # sample voxels in body
            bz, by_, bx = np.where(mkp > 0)
            if len(bz) < 100: continue
            pick = rng.choice(len(bz), min(VOX_PER_PATCH, len(bz)), replace=False)
            bz, by_, bx = bz[pick], by_[pick], bx[pick]
            samples.append(dict(
                phi_mr=phi_mr[:, bz, by_, bx].T.astype(np.float32),   # (n,16)
                phi_ct=phi_ct[:, bz, by_, bx].T.astype(np.float32),
                mrctx=ctxp[bz, by_, bx].astype(np.float32),           # (n,5)
                mr=mp[bz, by_, bx].astype(np.float32),
                hu=hup[bz, by_, bx].astype(np.float32),
                seg=segp[bz, by_, bx].astype(np.int16),
                split=np.array([split_tag]*len(bz)),
            ))
        print(f"  [{split_tag}] {si+1}/{len(subjects)} {sid}", flush=True)
    cat = lambda k: np.concatenate([s[k] for s in samples])
    return {k: cat(k) for k in samples[0]}, sharp

def main():
    t0 = time.time()
    net = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(DEV).eval()
    net.load_state_dict(clean_state_dict(torch.load(CK, map_location=DEV)), strict=True)
    tr, te = pick_subjects()
    print(f"In-dist(train) {len(tr)} subj | OOD(test) {len(te)} subj")
    print("Extracting in-dist..."); D_in, sh_in = extract(net, tr, "in")
    print("Extracting OOD...");     D_ood, sh_ood = extract(net, te, "ood")
    np.savez("/tmp/amix_probe/data_in.npz", **D_in)
    np.savez("/tmp/amix_probe/data_ood.npz", **D_ood)
    json.dump({"sharp_phi_in": float(np.mean(sh_in["phi"])), "sharp_mr_in": float(np.mean(sh_in["mr"])),
               "n_in": int(len(D_in["mr"])), "n_ood": int(len(D_ood["mr"]))},
              open("/tmp/amix_probe/meta.json", "w"), indent=2)
    print(f"Saved. in={len(D_in['mr'])} ood={len(D_ood['mr'])} voxels. {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
