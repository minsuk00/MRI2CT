"""E9 (decisive): does adding Anatomix phi SHIFT the bone-soft Pareto frontier inward?
E8 showed loss/arch only MOVE ALONG the frontier (info bottleneck: MR can't see bone).
Hypothesis: phi adds bone-location info in voids (E3) -> MR+phi frontier DOMINATES MR-only frontier.
Sweep bone-weight alpha to trace each frontier. If MR+phi is inside -> real method novelty.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from monai.networks.nets import BasicUNet
from anatomix.model.network import Unet as AmixUnet
from common.utils import clean_state_dict

ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT = "/home/minsukc/MRI2CT/splits/center_wise_split.txt"
CK = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
DEV = "cuda"; P = 96; STEPS = 1000; BS = 4
rng_g = np.random.default_rng(3)

def region_of(s):
    import re; p = re.match(r"1([A-Z]+)", s).group(1)
    return p[:2] if p[:2] in ("AB","HN","TH") else p[:1]
def pick(split, k):
    rows = [l.split() for l in open(SPLIT) if l.strip()]
    by = {}
    for sp, sid in rows:
        if sp == split: by.setdefault(region_of(sid), []).append(sid)
    out = []
    for r, v in by.items(): out += list(rng_g.choice(v, min(k, len(v)), replace=False))
    return out
def load(sid):
    d = os.path.join(ROOT, sid)
    ct = nib.load(d+"/ct.nii").get_fdata().astype(np.float32)
    mr = nib.load(d+"/moved_mr.nii").get_fdata().astype(np.float32)
    mask = (nib.load(d+"/mask.nii").get_fdata() > 0).astype(np.uint8)
    mrn = (mr-mr.min())/(mr.max()-mr.min()+1e-8); ctn = (np.clip(ct,-1024,1024)+1024)/2048.0
    return mrn, ctn, mask

def build(net, subjects, n_patch, rng):
    MR, PHI, CT, MK = [], [], [], []
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
            MR.append(mp[None].astype(np.float16)); PHI.append(phi.astype(np.float16))
            CT.append(cp[None].astype(np.float16)); MK.append(mk[None].astype(np.uint8))
    return (torch.tensor(np.stack(MR)), torch.tensor(np.stack(PHI)),
            torch.tensor(np.stack(CT)), torch.tensor(np.stack(MK)))

def bonew_l1(p,t,m,a):
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

def get_x(D, use_phi):
    return torch.cat([D[0], D[1]], 1) if use_phi else D[0]  # (MR) or (MR,phi)

def train_eval(tr, va, use_phi, alpha, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(200+seed)
    in_ch = 17 if use_phi else 1
    m = BasicUNet(spatial_dims=3, in_channels=in_ch, out_channels=1,
                  features=(32,32,64,128,64,32), act="relu").to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xtr = get_x(tr, use_phi); CTtr = tr[2]; MKtr = tr[3]; n = Xtr.shape[0]
    for s in range(STEPS):
        idx = rng.choice(n, BS)
        xb=Xtr[idx].float().to(DEV); yb=CTtr[idx].float().to(DEV); mb=MKtr[idx].float().to(DEV)
        xb,yb,mb = aug(xb,yb,mb,rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p=torch.sigmoid(m(xb)); loss=bonew_l1(p,yb,mb,alpha)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); Xva=get_x(va, use_phi); CTva=va[2]; MKva=va[3]; res=[]
    with torch.no_grad():
        for i in range(Xva.shape[0]):
            xb=Xva[i:i+1].float().to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16): p=torch.sigmoid(m(xb))
            res.append(mae_breakdown(p.float().cpu(), CTva[i:i+1].float(), MKva[i:i+1].float()))
    return {k: float(np.mean([r[k] for r in res if r[k] is not None])) for k in res[0]}

def main():
    t0=time.time()
    net = AmixUnet(3,1,16,4,32,norm="batch",interp="nearest",pooling="Max").to(DEV).eval()
    net.load_state_dict(clean_state_dict(torch.load(CK,map_location=DEV)), strict=True)
    tr_s, va_s = pick("train",4), pick("val",3)
    rng=np.random.default_rng(7); tr=build(net, tr_s, 8, rng); va=build(net, va_s, 8, rng)
    del net; torch.cuda.empty_cache()
    print(f"E9 patches train {tr[0].shape[0]} val {va[0].shape[0]}", flush=True)
    ALPHAS=[0.0, 3.0, 8.0]; SEEDS=[0,1,2]; out={}
    for use_phi in [False, True]:
        tag = "MR+phi" if use_phi else "MR"
        for a in ALPHAS:
            runs=[train_eval(tr,va,use_phi,a,s) for s in SEEDS]
            agg={k:(float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))) for k in runs[0]}
            out[f"{tag}_a{a:g}"]=agg
            print(f"  {tag:7s} a={a:g}  bone={agg['bone'][0]:.0f}±{agg['bone'][1]:.0f}  soft={agg['soft'][0]:.1f}±{agg['soft'][1]:.1f}  all={agg['all'][0]:.1f}", flush=True)
    json.dump(out, open("/tmp/amix_probe/e9_results.json","w"), indent=2)
    print(f"E9 done {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
