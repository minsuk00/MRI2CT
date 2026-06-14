"""E1-E5 analyses on extracted voxel samples. All sklearn, CPU."""
import json, numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error

rng = np.random.default_rng(1)
D_in = dict(np.load("/tmp/amix_probe/data_in.npz", allow_pickle=True))
D_ood = dict(np.load("/tmp/amix_probe/data_ood.npz", allow_pickle=True))
meta = json.load(open("/tmp/amix_probe/meta.json"))
R = {}

def subsample(D, n):
    idx = rng.choice(len(D["mr"]), min(n, len(D["mr"])), replace=False)
    return {k: D[k][idx] for k in D}

def split_tt(n, frac=0.6):
    idx = rng.permutation(n); k = int(n*frac); return idx[:k], idx[k:]

def hu_from(D): return D["hu"]  # already HU
def tissue(hu): return np.where(hu < -300, 0, np.where(hu > 200, 2, 1))  # air,soft,bone

# ---------- E1: HU regression (pointwise) ----------
def e1(D, tag):
    D = subsample(D, 240000)
    yt = hu_from(D); tr, te = split_tt(len(yt))
    out = {}
    for name, X in [("phi_mr", D["phi_mr"]), ("mrctx", D["mrctx"]), ("phi+mrctx", np.concatenate([D["phi_mr"], D["mrctx"]], 1))]:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        rg = Ridge(alpha=1.0).fit(Xtr, yt[tr]); pr = rg.predict(Xte)
        tis = tissue(yt[te])
        out[name] = dict(mae=float(mean_absolute_error(yt[te], pr)),
                         mae_air=float(mean_absolute_error(yt[te][tis==0], pr[tis==0])) if (tis==0).any() else None,
                         mae_soft=float(mean_absolute_error(yt[te][tis==1], pr[tis==1])),
                         mae_bone=float(mean_absolute_error(yt[te][tis==2], pr[tis==2])) if (tis==2).any() else None)
    # small MLP on phi vs mrctx
    for name, X in [("phi_mr_mlp", D["phi_mr"]), ("mrctx_mlp", D["mrctx"])]:
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ml = MLPRegressor(hidden_layer_sizes=(64,64), max_iter=60, early_stopping=True).fit(Xtr, yt[tr])
        out[name] = dict(mae=float(mean_absolute_error(yt[te], ml.predict(Xte))))
    R[f"E1_HUreg_{tag}"] = out
    print(f"E1[{tag}] phi_mr MAE={out['phi_mr']['mae']:.1f} | mrctx MAE={out['mrctx']['mae']:.1f} | both={out['phi+mrctx']['mae']:.1f} | bone phi={out['phi_mr']['mae_bone']:.0f} mrctx={out['mrctx']['mae_bone']:.0f}")

# ---------- E2: anatomy informativeness (seg) ----------
def e2(D, tag):
    D = subsample(D, 200000)
    classes = np.unique(D["seg"]); remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[c] for c in D["seg"]])
    tr, te = split_tt(len(y)); out = {}
    for name, X in [("phi_mr", D["phi_mr"]), ("mrctx", D["mrctx"])]:
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=200, C=1.0, multi_class="multinomial").fit(sc.transform(X[tr]), y[tr])
        pr = lr.predict(sc.transform(X[te]))
        bone_cls = remap.get(5)
        out[name] = dict(macroF1=float(f1_score(y[te], pr, average="macro")),
                         boneF1=float(f1_score(y[te]==bone_cls, pr==bone_cls)) if bone_cls is not None else None)
    R[f"E2_seg_{tag}"] = out
    print(f"E2[{tag}] macroF1 phi={out['phi_mr']['macroF1']:.3f} mrctx={out['mrctx']['macroF1']:.3f} | boneF1 phi={out['phi_mr']['boneF1']:.3f} mrctx={out['mrctx']['boneF1']:.3f}")

# ---------- E3: bone vs air in MR signal-voids ----------
def e3(D, tag):
    void = D["mr"] < 0.12
    hu = D["hu"][void]; lab = tissue(hu)
    keep = lab != 1  # air(0) vs bone(2) only
    y = (lab[keep] == 2).astype(int)  # 1=bone
    if y.sum() < 200 or (1-y).sum() < 200:
        print(f"E3[{tag}] insufficient void bone/air"); return
    phi = D["phi_mr"][void][keep]; ctx = D["mrctx"][void][keep]
    tr, te = split_tt(len(y)); out = {}
    for name, X in [("phi_mr", phi), ("mrctx", ctx)]:
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=200).fit(sc.transform(X[tr]), y[tr])
        p = lr.predict_proba(sc.transform(X[te]))[:, 1]
        out[name] = float(roc_auc_score(y[te], p))
    out["n_void_bone"] = int(y.sum()); out["n_void_air"] = int((1-y).sum())
    R[f"E3_voidbone_{tag}"] = out
    print(f"E3[{tag}] bone-vs-air AUC phi={out['phi_mr']:.3f} mrctx={out['mrctx']:.3f} (nbone={out['n_void_bone']} nair={out['n_void_air']})")

# ---------- E4: cross-modal alignment ----------
def e4(D, tag):
    D = subsample(D, 120000)
    a = D["phi_mr"]; b = D["phi_ct"]
    an = a / (np.linalg.norm(a, axis=1, keepdims=True)+1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True)+1e-8)
    matched = float(np.mean(np.sum(an*bn, 1)))
    perm = rng.permutation(len(bn))
    shuffled = float(np.mean(np.sum(an*bn[perm], 1)))
    tr, te = split_tt(len(a))
    rg = Ridge(alpha=1.0).fit(a[tr], b[tr])
    r2 = float(rg.score(a[te], b[te]))
    R[f"E4_crossmodal_{tag}"] = dict(cos_matched=matched, cos_shuffled=shuffled, phi_mr_to_phi_ct_R2=r2)
    print(f"E4[{tag}] cos matched={matched:.3f} shuffled={shuffled:.3f} | phi_mr->phi_ct R2={r2:.3f}")

# ---------- E5: redundancy + spectral ----------
def e5(D, tag):
    D = subsample(D, 120000)
    tr, te = split_tt(len(D["mr"]))
    rg = Ridge(alpha=1.0).fit(D["mrctx"][tr], D["phi_mr"][tr])
    r2 = float(rg.score(D["mrctx"][te], D["phi_mr"][te]))  # how much phi is derivable from MR context
    R[f"E5_redundancy_{tag}"] = dict(mrctx_to_phi_mr_R2=r2)
    print(f"E5[{tag}] MRctx->phi_mr R2={r2:.3f} (high=redundant)")

for tag, D in [("in", D_in), ("ood", D_ood)]:
    e1(D, tag); e2(D, tag); e3(D, tag); e4(D, tag); e5(D, tag)
R["spectral_sharpness"] = dict(phi=meta["sharp_phi_in"], mr=meta["sharp_mr_in"],
                               ratio=meta["sharp_phi_in"]/(meta["sharp_mr_in"]+1e-12))
print(f"Spectral: phi lapvar={meta['sharp_phi_in']:.4g} mr lapvar={meta['sharp_mr_in']:.4g}")
json.dump(R, open("/tmp/amix_probe/results.json", "w"), indent=2)
print("Saved results.json")
