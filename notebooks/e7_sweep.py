"""E7: proxy sweep of LOSS/OUTPUT/ARCH interventions that could beat L1 on the BONE bottleneck.
Fixed arch/data/steps/seed + light flip-aug; only the intervention changes.
Goal: find a 'we changed X, results got better' novelty, orthogonal to augmentation.
Metric: held-out MAE all/air/soft/bone (HU) + Laplacian-MAE (blur proxy, lower=sharper).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch, torch.nn as nn, torch.nn.functional as F
from monai.networks.nets import BasicUNet

ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT = "/home/minsukc/MRI2CT/splits/center_wise_split.txt"
DEV = "cuda"; P = 96; STEPS = 1000; BS = 4
rng = np.random.default_rng(3); torch.manual_seed(0)

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

def build(subjects, n_patch):
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

# ---- losses operating on pred01 (sigmoid, [0,1]) vs t01, mask ----
def grad3(x):
    dz = x[:,:,1:]-x[:,:,:-1]; dy = x[:,:,:,1:]-x[:,:,:,:-1]; dx = x[:,:,:,:,1:]-x[:,:,:,:,:-1]
    return dz, dy, dx

def l_l1(p,t,m): return (((p-t).abs())*m).sum()/(m.sum()+1e-6)
def l_gdl(p,t,m):
    base = l_l1(p,t,m)
    loss = 0
    for gp, gt in zip(grad3(p), grad3(t)): loss = loss + (gp-gt).abs().mean()
    return base + 0.5*loss
def l_bonew(p,t,m):
    hu_t = t*2048-1024
    w = m*(1.0 + 4.0*(hu_t>200).float())
    return ((p-t).abs()*w).sum()/(w.sum()+1e-6)
def l_ffl(p,t,m):
    base = l_l1(p,t,m)
    Fp = torch.fft.rfftn(p*m, dim=(2,3,4)); Ft = torch.fft.rfftn(t*m, dim=(2,3,4))
    d = (Fp-Ft).abs(); w = d.detach()  # focal weight on hard freqs
    ffl = (w*d).mean()/(w.mean()+1e-6)
    return base + 0.3*ffl
def l_l1_ssim(p,t,m):
    base = l_l1(p,t,m)
    try:
        from fused_ssim import fused_ssim3d
        s = 1.0 - fused_ssim3d((p*m).float(), (t*m).float(), train=True)
    except Exception:
        s = l_l1(p,t,m)*0
    return base + 0.3*s
def l_bonew_gdl(p,t,m): return l_bonew(p,t,m) + (l_gdl(p,t,m)-l_l1(p,t,m))

L1FAMILY = {"L1": l_l1, "L1+SSIM(cur)": l_l1_ssim, "L1+GDL": l_gdl,
            "L1+FFL": l_ffl, "boneW-L1": l_bonew, "boneW+GDL": l_bonew_gdl}

def mae_breakdown(pred01, t01, m):
    hu_p = pred01*2048-1024; hu_t = t01*2048-1024; mk = m>0
    d = (hu_p-hu_t).abs()[mk]; huy = hu_t[mk]
    res = {"all": float(d.mean())}
    for nm, sel in [("air",huy<-300),("soft",(huy>=-300)&(huy<=200)),("bone",huy>200)]:
        res[nm] = float(d[sel].mean()) if sel.any() else None
    # blur proxy: MAE of laplacian (high-freq) in body
    lap = lambda v: (v[:,:,2:]-2*v[:,:,1:-1]+v[:,:,:-2])
    res["lapMAE"] = float(((lap(hu_p)-lap(hu_t)).abs()*m[:,:,1:-1]).sum()/(m[:,:,1:-1].sum()+1e-6))
    return res

def aug(x, t, m):  # random flips, shared
    for ax in (2,3,4):
        if rng.random() < 0.5:
            x = x.flip(ax); t = t.flip(ax); m = m.flip(ax)
    return x, t, m

def make_net(out_ch):
    return BasicUNet(spatial_dims=3, in_channels=1, out_channels=out_ch,
                     features=(32,32,64,128,64,32), act="relu").to(DEV)

def run_l1family(name, lossfn, tr, va, seed=0):
    torch.manual_seed(seed)
    net = make_net(1); opt = torch.optim.Adam(net.parameters(), 1e-3)
    MRtr, CTtr, MKtr = tr; n = MRtr.shape[0]
    for s in range(STEPS):
        idx = rng.choice(n, BS)
        xb = MRtr[idx].float().to(DEV); yb = CTtr[idx].float().to(DEV); mb = MKtr[idx].float().to(DEV)
        xb, yb, mb = aug(xb, yb, mb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(net(xb)); loss = lossfn(p, yb, mb)
        opt.zero_grad(); loss.backward(); opt.step()
    return eval_net(net, va, kind="reg")

def run_classification(name, tr, va, K=80, seed=0):
    """Anti-mean-regression: predict HU as soft classification over K bins, recon via soft-argmax."""
    torch.manual_seed(seed)
    net = make_net(K); opt = torch.optim.Adam(net.parameters(), 1e-3)
    centers = torch.linspace(0, 1, K, device=DEV)
    MRtr, CTtr, MKtr = tr; n = MRtr.shape[0]
    for s in range(STEPS):
        idx = rng.choice(n, BS)
        xb = MRtr[idx].float().to(DEV); yb = CTtr[idx].float().to(DEV); mb = MKtr[idx].float().to(DEV)
        xb, yb, mb = aug(xb, yb, mb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(xb)  # B,K,...
            tgt = torch.clamp((yb.squeeze(1)*(K-1)).round().long(), 0, K-1)  # B,...
            ce = F.cross_entropy(logits, tgt, reduction="none")*mb.squeeze(1)
            loss = ce.sum()/(mb.sum()+1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
    return eval_net(net, va, kind="cls", centers=centers, K=K)

def run_multitask(name, tr, va, seed=0):
    """L1 + auxiliary bone-segmentation head (out ch1)."""
    torch.manual_seed(seed)
    net = make_net(2); opt = torch.optim.Adam(net.parameters(), 1e-3)
    MRtr, CTtr, MKtr = tr; n = MRtr.shape[0]
    for s in range(STEPS):
        idx = rng.choice(n, BS)
        xb = MRtr[idx].float().to(DEV); yb = CTtr[idx].float().to(DEV); mb = MKtr[idx].float().to(DEV)
        xb, yb, mb = aug(xb, yb, mb)
        bone_gt = ((yb*2048-1024) > 200).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net(xb); p = torch.sigmoid(o[:,0:1]); bone_logit = o[:,1:2]
            loss = l_l1(p, yb, mb) + 0.5*(F.binary_cross_entropy_with_logits(bone_logit, bone_gt, reduction="none")*mb).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return eval_net(net, va, kind="mt")

def eval_net(net, va, kind, centers=None, K=None):
    net.eval(); MRva, CTva, MKva = va; res = []
    with torch.no_grad():
        for i in range(MRva.shape[0]):
            xb = MRva[i:i+1].float().to(DEV); yb = CTva[i:i+1].float(); mb = MKva[i:i+1].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net(xb)
                if kind == "cls":
                    prob = torch.softmax(o.float(), 1); pred01 = (prob*centers.view(1,K,1,1,1)).sum(1, keepdim=True)
                elif kind == "mt":
                    pred01 = torch.sigmoid(o[:,0:1])
                else:
                    pred01 = torch.sigmoid(o)
            res.append(mae_breakdown(pred01.float().cpu(), yb, mb))
    return {k: float(np.mean([r[k] for r in res if r[k] is not None])) for k in res[0]}

def main():
    t0 = time.time()
    tr_s, va_s = pick("train", 4), pick("val", 3)
    print(f"E7 train {len(tr_s)} subj, val {len(va_s)} subj. Building...", flush=True)
    tr = build(tr_s, 8); va = build(va_s, 8)
    print(f"patches train {tr[0].shape[0]} val {va[0].shape[0]}", flush=True)
    out = {}
    for name, fn in L1FAMILY.items():
        r = run_l1family(name, fn, tr, va); out[name] = r
        print(f"  {name:14s} all={r['all']:.1f} bone={r['bone']:.0f} soft={r['soft']:.1f} air={r['air']:.0f} lapMAE={r['lapMAE']:.1f}", flush=True)
    r = run_classification("classify-HU", tr, va); out["classify-HU"] = r
    print(f"  {'classify-HU':14s} all={r['all']:.1f} bone={r['bone']:.0f} soft={r['soft']:.1f} air={r['air']:.0f} lapMAE={r['lapMAE']:.1f}", flush=True)
    r = run_multitask("multitask-bone", tr, va); out["multitask-bone"] = r
    print(f"  {'multitask-bone':14s} all={r['all']:.1f} bone={r['bone']:.0f} soft={r['soft']:.1f} air={r['air']:.0f} lapMAE={r['lapMAE']:.1f}", flush=True)
    json.dump(out, open("/tmp/amix_probe/e7_results.json","w"), indent=2)
    print(f"E7 done {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
