"""E12: do the TYPICAL ICLR/NeurIPS architecture novelties beat a plain UNet here?
Same data/loss(L1)/steps/seed/aug, MR input. Only the backbone changes:
plain UNet vs Attention-UNet vs bigger-UNet(capacity) vs SwinUNETR(transformer) vs SegResNet.
If none beat plain UNet -> the problem is information-limited, not architecture-limited (measured, not asserted).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch
from monai.networks.nets import BasicUNet, AttentionUnet, SegResNet
try: from monai.networks.nets import SwinUNETR; HAS_SWIN=True
except Exception: HAS_SWIN=False

ROOT="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT="/home/minsukc/MRI2CT/splits/center_wise_split.txt"
DEV="cuda"; P=96; STEPS=1000; BS=4; rng_g=np.random.default_rng(3)

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
def build(subjects,n_patch,rng):
    MR,CT,MK=[],[],[]
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
            MR.append(mp[None].astype(np.float16)); CT.append(cp[None].astype(np.float16)); MK.append(mk[None].astype(np.uint8))
    return torch.tensor(np.stack(MR)),torch.tensor(np.stack(CT)),torch.tensor(np.stack(MK))
def l1m(p,t,m): return (((p-t).abs())*m).sum()/(m.sum()+1e-6)
def mae_b(pred01,t01,m):
    hp=pred01*2048-1024; ht=t01*2048-1024; mk=m>0; d=(hp-ht).abs()[mk]; hy=ht[mk]; res={"all":float(d.mean())}
    for nm,sel in [("air",hy<-300),("soft",(hy>=-300)&(hy<=200)),("bone",hy>200)]:
        res[nm]=float(d[sel].mean()) if sel.any() else None
    return res
def aug(x,t,m,rng):
    for ax in (2,3,4):
        if rng.random()<0.5: x=x.flip(ax);t=t.flip(ax);m=m.flip(ax)
    return x,t,m

def make(arch):
    if arch=="UNet": return BasicUNet(spatial_dims=3,in_channels=1,out_channels=1,features=(32,32,64,128,64,32),act="relu").to(DEV)
    if arch=="UNet-big": return BasicUNet(spatial_dims=3,in_channels=1,out_channels=1,features=(64,64,128,256,128,64),act="relu").to(DEV)
    if arch=="AttentionUNet": return AttentionUnet(spatial_dims=3,in_channels=1,out_channels=1,channels=(32,64,128,256),strides=(2,2,2)).to(DEV)
    if arch=="SegResNet": return SegResNet(spatial_dims=3,in_channels=1,out_channels=1,init_filters=16).to(DEV)
    if arch=="SwinUNETR":
        try: return SwinUNETR(img_size=(P,P,P),in_channels=1,out_channels=1,feature_size=24).to(DEV)
        except TypeError: return SwinUNETR(in_channels=1,out_channels=1,feature_size=24).to(DEV)
    raise ValueError(arch)

def nparams(m): return sum(p.numel() for p in m.parameters())

def train_eval(arch,tr,va,seed):
    torch.manual_seed(seed); rng=np.random.default_rng(600+seed)
    m=make(arch); opt=torch.optim.Adam(m.parameters(),1e-3)
    MRtr,CTtr,MKtr=tr; n=MRtr.shape[0]
    for s in range(STEPS):
        idx=rng.choice(n,BS)
        xb=MRtr[idx].float().to(DEV); yb=CTtr[idx].float().to(DEV); mb=MKtr[idx].float().to(DEV)
        xb,yb,mb=aug(xb,yb,mb,rng)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=torch.sigmoid(m(xb)); loss=l1m(p,yb,mb)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); MRv,CTv,MKv=va; res=[]
    with torch.no_grad():
        for i in range(MRv.shape[0]):
            xb=MRv[i:i+1].float().to(DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=torch.sigmoid(m(xb))
            res.append(mae_b(p.float().cpu(),CTv[i:i+1].float(),MKv[i:i+1].float()))
    out={k:float(np.mean([r[k] for r in res if r[k] is not None])) for k in res[0]}
    out["params_M"]=round(nparams(m)/1e6,1)
    return out

def main():
    t0=time.time(); tr_s,va_s=pick("train",4),pick("val",3); rng=np.random.default_rng(7)
    tr=build(tr_s,8,rng); va=build(va_s,8,rng)
    print(f"E12 patches train {tr[0].shape[0]} val {va[0].shape[0]}",flush=True)
    archs=["UNet","UNet-big","AttentionUNet","SegResNet"]+(["SwinUNETR"] if HAS_SWIN else [])
    out={}
    for a in archs:
        try:
            runs=[train_eval(a,tr,va,s) for s in [0,1]]
            agg={k:(float(np.mean([r[k] for r in runs])),float(np.std([r[k] for r in runs]))) for k in ["all","soft","bone","air"]}
            agg["params_M"]=runs[0]["params_M"]; out[a]=agg
            print(f"  {a:14s} ({agg['params_M']}M) all={agg['all'][0]:.1f} bone={agg['bone'][0]:.0f}±{agg['bone'][1]:.0f} soft={agg['soft'][0]:.1f}",flush=True)
        except Exception as e:
            print(f"  {a:14s} FAILED: {e}",flush=True)
    json.dump(out,open("/tmp/amix_probe/e12_results.json","w"),indent=2)
    print(f"E12 done {time.time()-t0:.0f}s")

if __name__=="__main__": main()
