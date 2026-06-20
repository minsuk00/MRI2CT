"""Experiment 2: the RETRIEVAL idea, tested honestly.

User's idea: retrieve a similar CT, deform it to the patient, feed as an extra
input channel. We model the retrieved+registered donor as a TEMPLATE channel
whose quality has two INDEPENDENT axes that mirror reality:

  intensity fidelity  : does the donor have the RIGHT bone height? This is bounded
                        by what the retrieval KEY can see. If the MR can't see bone
                        intensity (alpha=0), retrieval-by-MR returns ~population-mean
                        height; only with alpha>0 can it find matching-height donors.
  registration fidelity (reg_jitter): residual bone-edge misalignment after deform.

Template variants:
  none        : baseline (MR only)
  oracle      : true height, true location           -> upper bound
  meanH_reg   : population-mean height, well-registered (reg_jitter small)
                -> tests: does a SHARP but mean-intensity template fix BLUR?
  trueH_misreg: true height but misregistered         -> tests: does misalignment hurt?
  retrieved   : realistic -> height = alpha*true + (1-alpha)*mean (+noise),
                moderate reg_jitter. Retrieval can only access alpha-fraction of
                intensity because the MR key is itself info-limited.
"""
import numpy as np
from toy_data import make_dataset, _bump, MEAN_H, N, W_LO, W_HI
from toy_core import train_regressor, predict, eval_metrics, fmt

NTR, NTE = 1500, 400


def soft_of(ct, bones):
    """Recover the soft-tissue field by subtracting the known true bones."""
    s = ct.copy()
    for b in bones:
        s = s - b["height"] * _bump(b["loc"], b["width"])
    return s


def build_templates(CT, BONES, kind, alpha=0.0, reg_jitter=0.0, seed=0):
    """Return [S,1,N] template channel = registered-donor CT (query soft + donor bones)."""
    rng = np.random.default_rng(seed + 12345)
    S = CT.shape[0]
    out = np.zeros((S, 1, N), dtype=np.float32)
    for s in range(S):
        soft = soft_of(CT[s], BONES[s])
        tmpl = soft.copy()
        for b in BONES[s]:
            if kind == "oracle":
                dh, dloc, dw = b["height"], b["loc"], b["width"]
            elif kind == "meanH_reg":
                dh = MEAN_H
                dloc = b["loc"] + rng.normal(0, reg_jitter)
                dw = int(rng.integers(W_LO, W_HI + 1))
            elif kind == "trueH_misreg":
                dh = b["height"]
                dloc = b["loc"] + rng.normal(0, reg_jitter)
                dw = int(rng.integers(W_LO, W_HI + 1))
            elif kind == "retrieved":
                # retrieval recovers only alpha-fraction of the height (key is info-limited)
                dh = alpha * b["height"] + (1 - alpha) * MEAN_H + rng.normal(0, 0.3)
                dloc = b["loc"] + rng.normal(0, reg_jitter)
                dw = int(rng.integers(W_LO, W_HI + 1))
            else:
                raise ValueError(kind)
            tmpl = tmpl + dh * _bump(dloc, dw)
        out[s, 0] = tmpl
    return out


def run(alpha, jitter, reg_jitter=0.8):
    mr_tr, ct_tr, m_tr, b_tr = make_dataset(NTR, alpha, jitter, seed=1)
    mr_te, ct_te, m_te, b_te = make_dataset(NTE, alpha, jitter, seed=999)

    print(f"\n=== alpha_info={alpha}, jitter={jitter}, reg_jitter={reg_jitter} ===")
    out = {}
    # baseline (no template)
    model = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0)
    met = eval_metrics(predict(model, mr_te), ct_te, m_te, b_te)
    out["none"] = met
    print(f"  {'none':>14} | {fmt(met)}")

    for kind in ["retrieved", "meanH_reg", "trueH_misreg", "oracle"]:
        rj = 2.5 if kind == "trueH_misreg" else reg_jitter
        tr_t = build_templates(ct_tr, b_tr, kind, alpha, rj, seed=1)
        te_t = build_templates(ct_te, b_te, kind, alpha, rj, seed=999)
        model = train_regressor(mr_tr, ct_tr, in_extra=tr_t, loss="l1", epochs=400, seed=0)
        met = eval_metrics(predict(model, mr_te, in_extra=te_t), ct_te, m_te, b_te)
        out[kind] = met
        print(f"  {kind:>14} | {fmt(met)}")
    return out


if __name__ == "__main__":
    results = {}
    for alpha in [0.0, 1.0]:
        results[alpha] = run(alpha, jitter=1.5)
    np.save("/home/minsukc/MRI2CT/bone_study/exp2_results.npy", results, allow_pickle=True)
    print("\nKey question: does a sharp MEAN-height template (meanH_reg) fix BLUR")
    print("(edge->1) even when it cannot fix intensity (amp bounded by alpha)?")
    print("And does misregistration (trueH_misreg) destroy the benefit?")
