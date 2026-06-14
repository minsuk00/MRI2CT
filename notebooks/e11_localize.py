"""E11 (confirmatory): does a REALISTIC bone localizer (not oracle) capture the oracle gain?
Method = localize-then-translate: Stage A net predicts bone-prob from MR; Stage B translator
conditions on it. Compare bone MAE vs MR-only (~308) and oracle ceiling (~145).
Also test phi-based localizer (uses E3 finding that phi separates bone voids at AUC 0.97).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch, torch.nn.functional as F
from monai.networks.nets import BasicUNet
from anatomix.model.network import Unet as AmixUnet
from common.utils import clean_state_dict

ROOT="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT="/home/minsukc/MRI2CT/splits/center_wise_split.txt"
CK="/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
DEV="cuda"; P=96; STEPS=1000; LOC_STEPS=800; BS=4; ALPHA=5.0; rng_g=np.random.default_rng(3)

def region_of(s):
    import re; p=re.match(r"1([A-Z]+)",s).group(1)
    return p[:2] if p[:2] in ("AB","HN","TH") else p[:1]
def pick(split,k):
    rows=[l.split() for l in open(SPLIT) if l.strip()]; by={}
    for sp,sid in rows:
        if sp==split: by.setdefault(region_of(sid),[]).append(sid)
    out=[]
    for r,v in by.items(): out+=list(rng_g.choice(v,min(k,len(v)),replace=False))
    return out
def load(sid):
    d=os.path.join(ROOT,sid)
    ct=nib.load(d+"/ct.nii").get_fdata().astype(np.float32)
    mr=nib.load(d+"/moved_mr.nii").get_fdata().astype(np.float32)
    mask=(nib.load(d+"/mask.nii").get_fdata()>0).astype(np.uint8)
    return (mr-mr.min())/(mr.max()-mr.min()+1e-8),(np.clip(ct,-1024,1024)+1024)/2048.0,mask

def build(net,subjects,n_patch,rng):
    MR,PHI,CT,MK=[],[],[],[]
    for sid in subjects:
        try: mrn,ctn,mask=load(sid)
        except Exception: continue
        zz,yy,xx=np.where(mask>0)
        if len(zz)<2000: continue
        for _ in range(n_patch):
            i=rng.integers(len(zz)); c=[zz[i],yy[i],xx[i]]
            sl=tuple(slice(int(np.clip(ci-P//2,0,dim-P)),int(np.clip(ci-P//2,0,dim-P))+P) for ci,dim in zip(c,mrn.shape))
            mp,cp,mk=mrn[sl],ctn[sl],mask[sl]
            if mp.shape!=(P,P,P): continue
            with torch.no_grad(): phi=net(torch.tensor(mp[None,None],device=DEV))[0].float().cpu().numpy()
            MR.append(mp[None].astype(np.float16)); PHI.append(phi.astype(np.float16))
            CT.append(cp[None].astype(np.float16)); MK.append(mk[None].astype(np.uint8))
    return (torch.tensor(np.stack(MR)),torch.tensor(np.stack(PHI)),torch.tensor(np.stack(CT)),torch.tensor(np.stack(MK)))

def bonew_l1(p,t,m,a=ALPHA):
    w=m*(1.0+a*((t*2048-1024)>200).float()); return ((p-t).abs()*w).sum()/(w.sum()+1e-6)
def mae_b(pred01,t01,m):
    hp=pred01*2048-1024; ht=t01*2048-1024; mk=m>0; d=(hp-ht).abs()[mk]; hy=ht[mk]; res={"all":float(d.mean())}
    for nm,sel in [("air",hy<-300),("soft",(hy>=-300)&(hy<=200)),("bone",hy>200)]:
        res[nm]=float(d[sel].mean()) if sel.any() else None
    return res
def aug2(xs,t,m,rng):  # flip a list of tensors jointly
    for ax in (2,3,4):
        if rng.random()<0.5:
            xs=[x.flip(ax) for x in xs]; t=t.flip(ax); m=m.flip(ax)
    return xs,t,m
def smallnet(ic,oc): return BasicUNet(spatial_dims=3,in_channels=ic,out_channels=oc,features=(32,32,64,128,64,32),act="relu").to(DEV)

def train_localizer(tr, src, seed):
    """src='MR'(1ch) or 'phi'(16ch). Predicts bone prob. Returns net + held bone-AUC proxy."""
    torch.manual_seed(seed); rng=np.random.default_rng(400+seed)
    ic = 1 if src=="MR" else 16
    loc=smallnet(ic,1); opt=torch.optim.Adam(loc.parameters(),1e-3)
    MR,PHI,CT,MK=tr; n=MR.shape[0]
    X = MR if src=="MR" else PHI
    for s in range(LOC_STEPS):
        idx=rng.choice(n,BS)
        xb=X[idx].float().to(DEV); yb=CT[idx].float().to(DEV); mb=MK[idx].float().to(DEV)
        [xb],yb,mb=aug2([xb],yb,mb,rng); bone=((yb*2048-1024)>200).float()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logit=loc(xb); loss=(F.binary_cross_entropy_with_logits(logit,bone,reduction="none")*mb).sum()/(mb.sum()+1e-6)
        opt.zero_grad(); loss.backward(); opt.step()
    loc.eval(); return loc

def predict_bone(loc, X, idx):
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        return torch.sigmoid(loc(X[idx].float().to(DEV))).float()

def train_translator(tr, va, cond, loc=None, locsrc=None, seed=0):
    """cond in {'none','predBone'}; loc = localizer net."""
    torch.manual_seed(seed); rng=np.random.default_rng(500+seed)
    ic = 1 if cond=="none" else 2
    m=smallnet(ic,1); opt=torch.optim.Adam(m.parameters(),1e-3)
    MR,PHI,CT,MK=tr; n=MR.shape[0]; Xloc = MR if locsrc=="MR" else PHI
    for s in range(STEPS):
        idx=rng.choice(n,BS)
        xb=MR[idx].float().to(DEV); yb=CT[idx].float().to(DEV); mb=MK[idx].float().to(DEV)
        if cond=="predBone":
            bp=predict_bone(loc,Xloc,idx)
            [xb,bp],yb,mb=aug2([xb,bp],yb,mb,rng); inp=torch.cat([xb,bp],1)
        else:
            [xb],yb,mb=aug2([xb],yb,mb,rng); inp=xb
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=torch.sigmoid(m(inp)); loss=bonew_l1(p,yb,mb)
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    m.eval(); MRv,PHIv,CTv,MKv=va; Xlocv=MRv if locsrc=="MR" else PHIv; res=[]
    with torch.no_grad():
        for i in range(MRv.shape[0]):
            xb=MRv[i:i+1].float().to(DEV)
            if cond=="predBone":
                bp=predict_bone(loc,Xlocv,slice(i,i+1)); inp=torch.cat([xb,bp],1)
            else: inp=xb
            with torch.autocast("cuda",dtype=torch.bfloat16): p=torch.sigmoid(m(inp))
            res.append(mae_b(p.float().cpu(),CTv[i:i+1].float(),MKv[i:i+1].float()))
    return {k:float(np.mean([r[k] for r in res if r[k] is not None])) for k in res[0]}

def main():
    t0=time.time()
    net=AmixUnet(3,1,16,4,32,norm="batch",interp="nearest",pooling="Max").to(DEV).eval()
    net.load_state_dict(clean_state_dict(torch.load(CK,map_location=DEV)),strict=True)
    tr_s,va_s=pick("train",4),pick("val",3); rng=np.random.default_rng(7)
    tr=build(net,tr_s,8,rng); va=build(net,va_s,8,rng); del net; torch.cuda.empty_cache()
    print(f"E11 patches train {tr[0].shape[0]} val {va[0].shape[0]}",flush=True)
    out={}; SEEDS=[0,1]
    # baseline
    runs=[train_translator(tr,va,"none",seed=s) for s in SEEDS]
    out["MR_baseline"]={k:(float(np.mean([r[k] for r in runs])),float(np.std([r[k] for r in runs]))) for k in runs[0]}
    print(f"  MR_baseline           bone={out['MR_baseline']['bone'][0]:.0f} soft={out['MR_baseline']['soft'][0]:.1f} all={out['MR_baseline']['all'][0]:.1f}",flush=True)
    # localize-then-translate, MR-localizer
    for src in ["MR","phi"]:
        runs=[]
        for s in SEEDS:
            loc=train_localizer(tr,src,s); runs.append(train_translator(tr,va,"predBone",loc=loc,locsrc=src,seed=s))
        out[f"localize({src})->translate"]={k:(float(np.mean([r[k] for r in runs])),float(np.std([r[k] for r in runs]))) for k in runs[0]}
        a=out[f"localize({src})->translate"]
        print(f"  localize({src})->translate bone={a['bone'][0]:.0f} soft={a['soft'][0]:.1f} all={a['all'][0]:.1f}",flush=True)
    json.dump(out,open("/tmp/amix_probe/e11_results.json","w"),indent=2)
    print(f"E11 done {time.time()-t0:.0f}s")

if __name__=="__main__": main()
