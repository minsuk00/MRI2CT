"""E10 (diagnostic): is the bone-MAE floor a LOCALIZATION problem or an HU-MAGNITUDE problem?
Give the model the ORACLE bone location (1[HU>200]) as input. boneW alpha=5, 3 seeds.
- Oracle bone-mask SHARPLY lowers bone floor -> localization is the lever -> atlas/seg/phi prior can help.
- Oracle barely helps -> it's HU magnitude (MR can't see bone density) -> need multi-contrast / generative prior.
Also re-tests MR+phi at matched setting.
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/minsukc/MRI2CT/src"); sys.path.append("/home/minsukc/MRI2CT")
import numpy as np, nibabel as nib, torch
from monai.networks.nets import BasicUNet
from anatomix.model.network import Unet as AmixUnet
from common.utils import clean_state_dict

ROOT="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SPLIT="/home/minsukc/MRI2CT/splits/center_wise_split.txt"
CK="/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
DEV="cuda"; P=96; STEPS=1000; BS=4; ALPHA=5.0; rng_g=np.random.default_rng(3)

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
    return (mr-mr.min())/(mr.max()-mr.min()+1e-8), (np.clip(ct,-1024,1024)+1024)/2048.0, mask

def build(net,subjects,n_patch,rng):
    MR,PHI,CT,MK,BONE=[],[],[],[],[]
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
            bone=((cp*2048-1024)>200).astype(np.float16)
            MR.append(mp[None].astype(np.float16)); PHI.append(phi.astype(np.float16))
            CT.append(cp[None].astype(np.float16)); MK.append(mk[None].astype(np.uint8)); BONE.append(bone[None])
    return (torch.tensor(np.stack(MR)),torch.tensor(np.stack(PHI)),torch.tensor(np.stack(CT)),
            torch.tensor(np.stack(MK)),torch.tensor(np.stack(BONE)))

def bonew_l1(p,t,m,a=ALPHA):
    w=m*(1.0+a*((t*2048-1024)>200).float()); return ((p-t).abs()*w).sum()/(w.sum()+1e-6)
def mae_b(pred01,t01,m):
    hp=pred01*2048-1024; ht=t01*2048-1024; mk=m>0; d=(hp-ht).abs()[mk]; hy=ht[mk]; res={"all":float(d.mean())}
    for nm,sel in [("air",hy<-300),("soft",(hy>=-300)&(hy<=200)),("bone",hy>200)]:
        res[nm]=float(d[sel].mean()) if sel.any() else None
    return res
def aug(x,t,m,rng):
    for ax in (2,3,4):
        if rng.random()<0.5: x=x.flip(ax);t=t.flip(ax);m=m.flip(ax)
    return x,t,m

def make_x(D, cfg):  # D=(MR,PHI,CT,MK,BONE)
    parts=[D[0]]
    if "phi" in cfg.lower(): parts.append(D[1])
    if "bone" in cfg.lower(): parts.append(D[4])
    return torch.cat(parts,1)
NCH={"MR":1,"MR+phi":17,"MR+oracleBone":2,"MR+oracleBone+phi":18}

def train_eval(tr,va,cfg,seed):
    torch.manual_seed(seed); rng=np.random.default_rng(300+seed)
    m=BasicUNet(spatial_dims=3,in_channels=NCH[cfg],out_channels=1,features=(32,32,64,128,64,32),act="relu").to(DEV)
    opt=torch.optim.Adam(m.parameters(),1e-3)
    Xtr=make_x(tr,cfg); CTtr=tr[2]; MKtr=tr[3]; n=Xtr.shape[0]
    for s in range(STEPS):
        idx=rng.choice(n,BS)
        xb=Xtr[idx].float().to(DEV); yb=CTtr[idx].float().to(DEV); mb=MKtr[idx].float().to(DEV)
        xb,yb,mb=aug(xb,yb,mb,rng)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=torch.sigmoid(m(xb)); loss=bonew_l1(p,yb,mb)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); Xva=make_x(va,cfg); CTva=va[2]; MKva=va[3]; res=[]
    with torch.no_grad():
        for i in range(Xva.shape[0]):
            xb=Xva[i:i+1].float().to(DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=torch.sigmoid(m(xb))
            res.append(mae_b(p.float().cpu(),CTva[i:i+1].float(),MKva[i:i+1].float()))
    return {k:float(np.mean([r[k] for r in res if r[k] is not None])) for k in res[0]}

def main():
    t0=time.time()
    net=AmixUnet(3,1,16,4,32,norm="batch",interp="nearest",pooling="Max").to(DEV).eval()
    net.load_state_dict(clean_state_dict(torch.load(CK,map_location=DEV)),strict=True)
    tr_s,va_s=pick("train",4),pick("val",3); rng=np.random.default_rng(7)
    tr=build(net,tr_s,8,rng); va=build(net,va_s,8,rng); del net; torch.cuda.empty_cache()
    print(f"E10 patches train {tr[0].shape[0]} val {va[0].shape[0]} (boneW a={ALPHA})",flush=True)
    out={}
    for cfg in ["MR","MR+oracleBone"]:
        runs=[train_eval(tr,va,cfg,s) for s in [0,1]]
        agg={k:(float(np.mean([r[k] for r in runs])),float(np.std([r[k] for r in runs]))) for k in runs[0]}
        out[cfg]=agg
        print(f"  {cfg:20s} bone={agg['bone'][0]:.0f}±{agg['bone'][1]:.0f} soft={agg['soft'][0]:.1f} all={agg['all'][0]:.1f}",flush=True)
    json.dump(out,open("/tmp/amix_probe/e10_results.json","w"),indent=2)
    print(f"E10 done {time.time()-t0:.0f}s")

if __name__=="__main__": main()
