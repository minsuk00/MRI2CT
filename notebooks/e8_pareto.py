"""E8: can we BREAK the bone-vs-soft Pareto tradeoff E7 revealed?
E7: boneW gets bone~290 but soft~136; multitask gets soft~88 but bone~437.
Target: a decomposition that gets bone~300 AND soft~90 (dominates both). Multi-seed.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
from monai.networks.nets import BasicUNet

ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT = "/home/minsukc/MRI2CT/splits/center_wise_split.txt"
DEV = "cuda"; P = 96; STEPS = 1000; BS = 4
rng_global = np.random.default_rng(3)

def region_of(s):
    import re; p = re.match(r"1([A-Z]+)", s).group(1)
    return p[:2] if p[:2] in ("AB","HN","TH") else p[:1]

def pick(split, k):
    rows = [l.split() for l in open(SPLIT) if l.strip()]
    ids = [sid for sp, sid in rows if sp == split]
    by = {}
    for sid in ids: by.setdefault(region_of(sid), []).append(sid)
    out = []
    for r, v in by.items(): out += list(rng_global.choice(v, min(k, len(v)), replace=False))
    return out

def load(sid):
    d = os.path.join(ROOT, sid)
    ct = nib.load(d+"/ct.nii").get_fdata().astype(np.float32)
    mr = nib.load(d+"/moved_mr.nii").get_fdata().astype(np.float32)
    mask = (nib.load(d+"/mask.nii").get_fdata() > 0).astype(np.uint8)
    mrn = (mr-mr.min())/(mr.max()-mr.min()+1e-8); ctn = (np.clip(ct,-1024,1024)+1024)/2048.0
    return mrn, ctn, mask

def build(subjects, n_patch, rng):
    MR, CT, MK = [], [], []
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
            MR.append(mp[None].astype(np.float16)); CT.append(cp[None].astype(np.float16)); MK.append(mk[None].astype(np.uint8))
    return torch.tensor(np.stack(MR)), torch.tensor(np.stack(CT)), torch.tensor(np.stack(MK))

def l1m(p,t,m): return (((p-t).abs())*m).sum()/(m.sum()+1e-6)
def bonew_l1(p,t,m,a=4.0):
    w = m*(1.0 + a*((t*2048-1024)>200).float()); return ((p-t).abs()*w).sum()/(w.sum()+1e-6)

def mae_breakdown(pred01, t01, m):
    hu_p = pred01*2048-1024; hu_t = t01*2048-1024; mk = m>0
    d = (hu_p-hu_t).abs()[mk]; huy = hu_t[mk]; res = {"all": float(d.mean())}
    for nm, sel in [("air",huy<-300),("soft",(huy>=-300)&(huy<=200)),("bone",huy>200)]:
        res[nm] = float(d[sel].mean()) if sel.any() else None
    return res

def aug(x,t,m,rng):
    for ax in (2,3,4):
        if rng.random()<0.5: x=x.flip(ax); t=t.flip(ax); m=m.flip(ax)
    return x,t,m

def net(out_ch): return BasicUNet(spatial_dims=3, in_channels=1, out_channels=out_ch,
                                  features=(32,32,64,128,64,32), act="relu").to(DEV)

def train_eval(method, tr, va, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(100+seed)
    MRtr,CTtr,MKtr = tr; n = MRtr.shape[0]
    oc = {"L1":1, "boneW":1, "gated2head":3, "mt+boneW":2, "gated+boneW":3}[method]
    m = net(oc); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for s in range(STEPS):
        idx = rng.choice(n, BS)
        xb=MRtr[idx].float().to(DEV); yb=CTtr[idx].float().to(DEV); mb=MKtr[idx].float().to(DEV)
        xb,yb,mb = aug(xb,yb,mb,rng); bone_gt = ((yb*2048-1024)>200).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(xb)
            if method=="L1":
                p=torch.sigmoid(o); loss=l1m(p,yb,mb)
            elif method=="boneW":
                p=torch.sigmoid(o); loss=bonew_l1(p,yb,mb)
            elif method=="mt+boneW":
                p=torch.sigmoid(o[:,0:1]); bl=o[:,1:2]
                loss=bonew_l1(p,yb,mb)+0.5*(F.binary_cross_entropy_with_logits(bl,bone_gt,reduction="none")*mb).mean()
            elif method in ("gated2head","gated+boneW"):
                soft=torch.sigmoid(o[:,0:1]); bone=torch.sigmoid(o[:,1:2]); g=torch.sigmoid(o[:,2:3])
                p=(1-g)*soft+g*bone
                gate_loss=0.5*(F.binary_cross_entropy_with_logits(o[:,2:3],bone_gt,reduction="none")*mb).mean()
                rec = bonew_l1(p,yb,mb) if method=="gated+boneW" else l1m(p,yb,mb)
                loss=rec+gate_loss
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    m.eval(); MRva,CTva,MKva = va; res=[]
    with torch.no_grad():
        for i in range(MRva.shape[0]):
            xb=MRva[i:i+1].float().to(DEV); yb=CTva[i:i+1].float(); mb=MKva[i:i+1].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o=m(xb)
                if method in ("L1","boneW"): p=torch.sigmoid(o)
                elif method=="mt+boneW": p=torch.sigmoid(o[:,0:1])
                else:
                    soft=torch.sigmoid(o[:,0:1]); bone=torch.sigmoid(o[:,1:2]); g=torch.sigmoid(o[:,2:3]); p=(1-g)*soft+g*bone
            res.append(mae_breakdown(p.float().cpu(), yb, mb))
    return {k: float(np.mean([r[k] for r in res if r[k] is not None])) for k in res[0]}

def main():
    t0=time.time(); tr_s,va_s = pick("train",4), pick("val",3)
    rng=np.random.default_rng(7); tr=build(tr_s,8,rng); va=build(va_s,8,rng)
    print(f"E8 patches train {tr[0].shape[0]} val {va[0].shape[0]}", flush=True)
    methods=["L1","boneW","mt+boneW","gated2head","gated+boneW"]; SEEDS=[0,1,2]
    out={}
    for meth in methods:
        runs=[train_eval(meth,tr,va,s) for s in SEEDS]
        agg={k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
        out[meth]=agg
        print(f"  {meth:13s} bone={agg['bone'][0]:.0f}±{agg['bone'][1]:.0f} soft={agg['soft'][0]:.1f}±{agg['soft'][1]:.1f} all={agg['all'][0]:.1f} air={agg['air'][0]:.0f}", flush=True)
    json.dump(out, open("/tmp/amix_probe/e8_results.json","w"), indent=2)
    print(f"E8 done {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
