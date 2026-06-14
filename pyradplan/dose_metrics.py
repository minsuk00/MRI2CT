"""Dose comparison metrics matching SynthRAD2025 functionally (pyRadPlan dose, numpy metrics).

Mirrors the challenge's DoseMetrics class:
  - mae_dose  : MAE in high-dose region (d_gt >= 0.9*Rx), normalized by Rx     [eq. 5]
  - dvh_metric: target_term (D98 + V95 rel-diff) + OAR_term (D2 + Dmean rel-diff over
                the 3 OARs ranked by (D5+Dmean)/2 on the GT dose)              [eqs. 6-9]
  - gamma_pass_rate : 2%/2mm global gamma pass rate (%)

All operate on (z,y,x) dose cubes (Gy) and boolean masks.
"""
import numpy as np


# ---------------------------------------------------------------- MAE (eq. 5)
def mae_dose(d_gt, d_sct, prescription, threshold=0.9):
    """Mean |d_gt - d_sct| / Rx over voxels with d_gt >= threshold*Rx (SynthRAD eq. 5)."""
    H = d_gt >= threshold * prescription
    if H.sum() == 0:
        return float("nan")
    return float(np.sum(np.abs(d_gt[H] - d_sct[H]) / prescription) / H.sum())


# --------------------------------------------------------- DVH point statistics
def d_x(dose, mask, vol_pct):
    """Dose received by at least vol_pct% of the masked volume (e.g. D98, D5, D2)."""
    d = dose[mask]
    if d.size == 0:
        return float("nan")
    return float(np.percentile(d, 100 - vol_pct))


def v_x(dose, mask, dose_level):
    """Percent of masked volume receiving >= dose_level Gy."""
    d = dose[mask]
    if d.size == 0:
        return float("nan")
    return float((d >= dose_level).mean() * 100.0)


def _rel(a, b, eps=1e-12):
    return abs(a - b + eps) / (a + eps)


# ------------------------------------------------------- DVH composite (eqs.6-9)
def dvh_metric(d_gt, d_sct, ptv_mask, oar_masks, prescription, n_oars=3):
    """SynthRAD composite DVH metric (lower = better agreement) + the raw components.

    oar_masks: dict {name: bool mask}. The 3 OARs used are the top-n_oars ranked by
    (D5 + Dmean)/2 on the GROUND-TRUTH dose (most-irradiated), matching their code.
    Returns (composite_value, details_dict).
    """
    # PTV target term: D98 + V95 relative differences
    D98_gt, D98_s = d_x(d_gt, ptv_mask, 98), d_x(d_sct, ptv_mask, 98)
    V95_gt = v_x(d_gt, ptv_mask, 0.95 * prescription)
    V95_s = v_x(d_sct, ptv_mask, 0.95 * prescription)
    target_term = _rel(D98_gt, D98_s) + _rel(V95_gt, V95_s)

    # rank OARs by (D5 + Dmean)/2 on the GT dose, take top n_oars
    ranked = []
    for name, m in oar_masks.items():
        if m.sum() == 0:
            continue
        score = (d_x(d_gt, m, 5) + float(d_gt[m].mean())) / 2.0
        ranked.append((score, name, m))
    ranked.sort(key=lambda t: -t[0])
    used = ranked[:n_oars]

    d2_rel, dmean_rel, per_oar = [], [], {}
    for _, name, m in used:
        D2_gt, D2_s = d_x(d_gt, m, 2), d_x(d_sct, m, 2)
        Dm_gt, Dm_s = float(d_gt[m].mean()), float(d_sct[m].mean())
        d2_rel.append(_rel(D2_gt, D2_s)); dmean_rel.append(_rel(Dm_gt, Dm_s))
        per_oar[name] = dict(D2_gt=D2_gt, D2_sct=D2_s, Dmean_gt=Dm_gt, Dmean_sct=Dm_s)

    oar_term = (np.mean(d2_rel) + np.mean(dmean_rel)) if d2_rel else 0.0
    composite = float(target_term + oar_term)
    details = dict(
        composite=composite, target_term=float(target_term), oar_term=float(oar_term),
        PTV_D98_gt=D98_gt, PTV_D98_sct=D98_s, PTV_V95_gt=V95_gt, PTV_V95_sct=V95_s,
        oars_used=[n for _, n, _ in used], per_oar=per_oar, n_oars=len(used),
    )
    return composite, details


# --------------------------------------------------------------------- gamma
def gamma_pass_rate(d_gt, d_sct, spacing_mm=1.5, dose_pct=2.0, dist_mm=2.0,
                    dose_thresh_pct=10.0):
    """Global gamma pass rate (%) for dose_pct%/dist_mm (default 2%/2mm).

    Global dose normalization to max(d_gt); voxels below dose_thresh_pct% of max are
    excluded (standard low-dose cutoff). Searches a +/-ceil(dist/spacing) voxel
    neighborhood and reports the fraction of evaluated voxels with gamma <= 1.
    """
    dmax = float(d_gt.max())
    if dmax <= 0:
        return float("nan")
    dose_crit = dose_pct / 100.0 * dmax
    roi = d_gt >= (dose_thresh_pct / 100.0 * dmax)
    r = int(np.ceil(dist_mm / spacing_mm))
    Z, Y, X = d_gt.shape
    gamma2 = np.full(d_gt.shape, np.inf, dtype=np.float32)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                phys2 = (dz * dz + dy * dy + dx * dx) * spacing_mm * spacing_mm
                if phys2 > dist_mm * dist_mm:        # outside the DTA sphere
                    continue
                zs = slice(max(0, dz), Z + min(0, dz)); zt = slice(max(0, -dz), Z + min(0, -dz))
                ys = slice(max(0, dy), Y + min(0, dy)); yt = slice(max(0, -dy), Y + min(0, -dy))
                xs = slice(max(0, dx), X + min(0, dx)); xt = slice(max(0, -dx), X + min(0, -dx))
                dd = (d_gt[zs, ys, xs] - d_sct[zt, yt, xt]) / dose_crit
                g2 = dd * dd + phys2 / (dist_mm * dist_mm)
                cur = gamma2[zs, ys, xs]
                np.minimum(cur, g2, out=cur)
                gamma2[zs, ys, xs] = cur
    g = np.sqrt(gamma2[roi])
    return float((g <= 1.0).mean() * 100.0)
