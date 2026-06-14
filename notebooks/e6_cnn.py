"""E6 capstone: does feeding phi(MR) help a SMALL translator vs raw MR?
Same arch/steps/seed, only input channels differ: MR(1) vs phi(16) vs MR+phi(17).
Direct mini-version of the real finding. In-dist patches; held-out val subjects.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch, torch.nn as nn
from anatomix.model.network import Unet as AmixUnet
from common.utils import clean_state_dict
from monai.networks.nets import BasicUNet

ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT = "/home/minsukc/MRI2CT/splits/center_wise_split.txt"
CK = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
DEV = "cuda"; P = 96; rng = np.random.default_rng(2)
torch.manual_seed(0)

def region_of(s):
    import re; p = re.match(r"1([A-Z]+)", s).group(1)
    return p[:2] if p[:2] in ("AB","HN","TH") else p[:1]

def pick(split, k):
    rows = [l.split() for l in open(SPLIT) if l.strip()]
    ids = [sid for sp, sid in rows if sp == split]
    by = {}
    for sid in ids: by.setdefault(region_of(sid), []).append(sid)
    out = []
    for r, v in by.items(): out += list(rng.choice(v, min(k, len(v)), replace=False))
    return out

def load(sid):
    d = os.path.join(ROOT, sid)
    ct = nib.load(d+"/ct.nii").get_fdata().astype(np.float32)
    mr = nib.load(d+"/moved_mr.nii").get_fdata().astype(np.float32)
    mask = (nib.load(d+"/mask.nii").get_fdata() > 0).astype(np.uint8)
    mrn = (mr-mr.min())/(mr.max()-mr.min()+1e-8)
    ctn = (np.clip(ct,-1024,1024)+1024)/2048.0
    return mrn, ctn, mask

def build(net, subjects, n_patch):
    X_mr, X_phi, Y, M = [], [], [], []
    for sid in subjects:
        try: mrn, ctn, mask = load(sid)
        except Exception: continue
        zz, yy, xx = np.where(mask > 0)
        if len(zz) < 2000: continue
        for _ in range(n_patch):
            i = rng.integers(len(zz)); c = [zz[i], yy[i], xx[i]]
            sl = tuple(slice(int(np.clip(ci-P//2,0,dim-P)), int(np.clip(ci-P//2,0,dim-P))+P)
                       for ci, dim in zip(c, mrn.shape))
            mp, cp, mk = mrn[sl], ctn[sl], mask[sl]
            if mp.shape != (P,P,P): continue
            with torch.no_grad():
                phi = net(torch.tensor(mp[None,None],device=DEV))[0].float().cpu().numpy()
            X_mr.append(mp[None].astype(np.float16)); X_phi.append(phi.astype(np.float16))
            Y.append(cp[None].astype(np.float16)); M.append(mk[None].astype(np.uint8))
    return (np.stack(X_mr), np.stack(X_phi), np.stack(Y), np.stack(M))

def mae_hu(pred, y, m):  # tensors [B,1,...] in [0,1]
    hu_p = pred*2048-1024; hu_y = y*2048-1024; mk = m > 0
    d = (hu_p-hu_y).abs()[mk]
    huy = hu_y[mk]
    res = {"all": float(d.mean())}
    for nm, sel in [("air", huy < -300), ("soft", (huy>=-300)&(huy<=200)), ("bone", huy > 200)]:
        res[nm] = float(d[sel].mean()) if sel.any() else None
    return res

def train_regime(name, in_ch, get_x, tr, va, steps=500, bs=4):
    net = BasicUNet(spatial_dims=3, in_channels=in_ch, out_channels=1,
                    features=(16,16,32,64,32,16), act="relu").to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3); l1 = nn.L1Loss()
    Xtr, Ytr, Mtr = get_x(tr), torch.tensor(tr[2].astype(np.float32)), torch.tensor(tr[3].astype(np.float32))
    n = Xtr.shape[0]
    for s in range(steps):
        idx = rng.choice(n, bs)
        xb = Xtr[idx].to(DEV); yb = Ytr[idx].to(DEV); mb = Mtr[idx].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = torch.sigmoid(net(xb)); loss = l1(pr*mb, yb*mb)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    net.eval(); Xva = get_x(va); Yva = torch.tensor(va[2].astype(np.float32)); Mva = torch.tensor(va[3].astype(np.float32))
    maes = []
    with torch.no_grad():
        for i in range(Xva.shape[0]):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pr = torch.sigmoid(net(Xva[i:i+1].to(DEV))).float().cpu()
            maes.append(mae_hu(pr, Yva[i:i+1], Mva[i:i+1]))
    agg = {k: float(np.mean([m[k] for m in maes if m[k] is not None])) for k in maes[0]}
    print(f"E6[{name}] val MAE all={agg['all']:.1f} | air={agg['air']:.0f} soft={agg['soft']:.1f} bone={agg['bone']:.0f}", flush=True)
    return agg

def main():
    t0 = time.time()
    net = AmixUnet(3,1,16,4,32,norm="batch",interp="nearest",pooling="Max").to(DEV).eval()
    net.load_state_dict(clean_state_dict(torch.load(CK,map_location=DEV)), strict=True)
    tr_s, va_s = pick("train", 3), pick("val", 2)
    print(f"E6 train {len(tr_s)} subj, val {len(va_s)} subj. Building patches...", flush=True)
    tr = build(net, tr_s, 6); va = build(net, va_s, 6)
    del net; torch.cuda.empty_cache()
    print(f"patches: train {tr[0].shape[0]}, val {va[0].shape[0]}", flush=True)
    def x_mr(D):  return torch.tensor(D[0].astype(np.float32))
    def x_phi(D): return torch.tensor(D[1].astype(np.float32))
    def x_both(D):return torch.tensor(np.concatenate([D[1], D[0]], 1).astype(np.float32))
    res = {}
    res["mr_1ch"]   = train_regime("MR(1ch)",   1,  x_mr,   tr, va)
    res["phi_16ch"] = train_regime("phi(16ch)", 16, x_phi,  tr, va)
    res["both_17ch"]= train_regime("MR+phi(17)",17, x_both, tr, va)
    json.dump(res, open("/tmp/amix_probe/e6_results.json","w"), indent=2)
    print(f"E6 done {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
