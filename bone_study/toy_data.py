"""
1D toy model of the MR->CT bone problem.

Essence we must reproduce (from reports 10/11):
  - CT has a smooth soft-tissue background plus RARE, SHARP, TALL "bone" features.
  - The input MR LOCATES bone well (AUC ~0.90) but is INFORMATION-LIMITED about
    bone INTENSITY (cortical HU): a single MR cannot distinguish a dense cortical
    voxel from a less-dense one (both are signal voids).
  - A regression (L1) model therefore UNDERSHOOTS bone amplitude and BLURS its
    edges -- one effect: convolving a sharp tall peak with the model's
    conditional/spatial uncertainty lowers AND widens it.

Two independent knobs let us separate the two root causes:
  alpha_info in [0,1] : fraction of TRUE bone height the MR encodes.
                        1 -> MR fully determines bone intensity (objective-limited regime)
                        0 -> MR carries only population-mean height (information-limited regime)
  jitter (float)      : std (in voxels) of bone-edge location uncertainty in the MR.
                        > 0 -> even a perfect-intensity model must average over edge
                        position -> blur. This is the spatial half of the failure.

The toy is deliberately simple and fully controlled so every claim is checkable.
"""
import numpy as np

N = 256          # signal length (voxels)
MARGIN = 12      # keep bones away from the borders
SOFT_LEVEL = 1.0 # typical soft-tissue amplitude (CT "muscle ~ 1")
H_LO, H_HI = 3.0, 6.0   # per-bone height range (the quantity MR cannot see when alpha=0)
# Right-skewed height: most bone is moderate, dense cortical bone is the RARE high tail.
# This makes the L1 conditional-median UNDERSHOOT the dense tail (the report's signed bias),
# not just scatter. Beta(1.6,3.2): mean 1/3 of range.
H_BETA_A, H_BETA_B = 1.6, 3.2
MEAN_H = H_LO + (H_HI - H_LO) * H_BETA_A / (H_BETA_A + H_BETA_B)  # population mean height
W_LO, W_HI = 3, 7       # per-bone half-width range (voxels); MR does NOT encode true width
VOID_MARKER = -1.0      # MR "void" level marking bone presence (like air/cortex on MR)
ENC_NOISE = 0.7         # noise on the MR height-encoding: even at alpha=1 the MR does
                        # not perfectly determine bone height (a realistic SNR ceiling)


def _smooth_field(rng, n=N, k=4, scale=0.3):
    """Low-frequency soft-tissue field: sum of a few random sinusoids, ~[ -scale, scale ] + SOFT_LEVEL."""
    x = np.linspace(0, 2 * np.pi, n)
    f = np.zeros(n)
    for _ in range(k):
        freq = rng.uniform(0.5, 3.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        f += amp * np.sin(freq * x + phase)
    f = f / (np.abs(f).max() + 1e-8) * scale
    return f + SOFT_LEVEL


def _bump(center, width, n=N):
    """A sharp, near-rectangular bone profile (super-Gaussian) of unit height."""
    x = np.arange(n)
    # super-Gaussian (flat top, sharp edges) approximates cortical-rim + marrow
    return np.exp(-((x - center) / max(width, 1e-3)) ** 8)


def gen_sample(rng, alpha_info=0.0, jitter=1.5, n_bones=None):
    """Return (mr, ct, bones) for one 1D 'subject'.

    bones: list of dicts with true loc/width/height and the MR-marked (jittered) loc.
    """
    if n_bones is None:
        n_bones = rng.integers(1, 4)  # 1..3 bones

    soft = _smooth_field(rng)
    ct = soft.copy()
    # MR soft-tissue observation: a DIFFERENT but informative transform of soft tissue
    # (monotone-ish + mild noise), so soft tissue is well-determined by MR.
    mr = 0.8 * soft + 0.2 * np.roll(soft, 3) + rng.normal(0, 0.02, N)

    bones = []
    locs = []
    for _ in range(n_bones):
        # reject overlaps so bone supports are separable for clean metrics
        for _try in range(50):
            loc = int(rng.integers(MARGIN, N - MARGIN))
            if all(abs(loc - p) > 18 for p in locs):
                break
        locs.append(loc)
        width = int(rng.integers(W_LO, W_HI + 1))
        height = float(H_LO + (H_HI - H_LO) * rng.beta(H_BETA_A, H_BETA_B))

        prof = _bump(loc, width)
        ct = ct + height * prof  # CT: sharp tall bone

        # --- MR encodes: PRESENCE + (jittered) LOCATION, but not true height/width ---
        loc_mr = loc + rng.normal(0, jitter)
        # marker amplitude carries height only through alpha_info; else population mean.
        # ENC_NOISE adds a fixed SNR floor so even alpha=1 is imperfectly recoverable.
        marked_h = alpha_info * height + (1.0 - alpha_info) * MEAN_H + alpha_info * rng.normal(0, ENC_NOISE)
        # constant small marker width (presence cue), independent of TRUE width:
        marker = _bump(loc_mr, 3.0)
        # MR shows a void dip at bone (negative), depth modulated by marked height:
        mr = mr + (VOID_MARKER - 0.6 * marked_h) * marker

        bones.append(dict(loc=loc, width=width, height=height,
                          loc_mr=float(loc_mr), marked_h=float(marked_h)))
    return mr.astype(np.float32), ct.astype(np.float32), bones


def bone_mask(bones, n=N, pad=1):
    """Binary mask over true bone support (where |profile|>0.5), padded by `pad`."""
    m = np.zeros(n, dtype=bool)
    for b in bones:
        prof = _bump(b["loc"], b["width"])
        idx = np.where(prof > 0.5)[0]
        if len(idx):
            lo, hi = max(0, idx.min() - pad), min(n, idx.max() + 1 + pad)
            m[lo:hi] = True
    return m


def make_dataset(n_samples, alpha_info=0.0, jitter=1.5, seed=0):
    rng = np.random.default_rng(seed)
    MR, CT, MASK, BONES = [], [], [], []
    for _ in range(n_samples):
        mr, ct, bones = gen_sample(rng, alpha_info, jitter)
        MR.append(mr); CT.append(ct); MASK.append(bone_mask(bones)); BONES.append(bones)
    return (np.stack(MR), np.stack(CT), np.stack(MASK), BONES)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, alpha in zip(axes.ravel(), [0.0, 0.0, 1.0, 1.0]):
        rng = np.random.default_rng(int(ax is axes.ravel()[1]) + int(ax is axes.ravel()[3]) * 2 + 7)
        mr, ct, bones = gen_sample(rng, alpha_info=alpha, jitter=1.5)
        ax.plot(ct, label="CT (target)", lw=1.8)
        ax.plot(mr, label="MR (input)", lw=1.0, alpha=0.8)
        ax.set_title(f"alpha_info={alpha}  bones={[round(b['height'],1) for b in bones]}")
        ax.legend(fontsize=8); ax.set_ylim(-2, 7)
    plt.tight_layout()
    plt.savefig("/home/minsukc/MRI2CT/bone_study/figs/00_toy_examples.png", dpi=90)
    print("saved 00_toy_examples.png")

    # Sanity: confirm the info limit. With alpha=0 the MR must be statistically
    # identical for bones of different height (corr ~0). With alpha=1 a windowed MR
    # feature (what a CNN integrates) must strongly track height. Single-voxel corr
    # under-reads the recoverable signal, so we use the windowed minimum (void depth).
    def windowed_feat(mr, loc, w=6):
        seg = mr[max(0, loc - w):min(N, loc + w + 1)]
        return seg.min()  # deepest void in the window

    for alpha in (0.0, 1.0):
        rng = np.random.default_rng(123)
        hs, feats = [], []
        for _ in range(4000):
            mr, ct, bones = gen_sample(rng, alpha_info=alpha, jitter=1.5, n_bones=1)
            b = bones[0]
            hs.append(b["height"]); feats.append(windowed_feat(mr, b["loc"]))
        r = np.corrcoef(hs, feats)[0, 1]
        print(f"corr(windowed-void-depth, true height): alpha={alpha} -> {r:+.3f}")
