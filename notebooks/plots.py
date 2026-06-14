import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("/tmp/amix_probe/results.json"))
E6 = json.load(open("/tmp/amix_probe/e6_results.json"))
plt.rcParams.update({"font.size": 11, "axes.facecolor": "#1a1d2e", "figure.facecolor": "#0f1117",
                     "text.color": "#e2e8f0", "axes.labelcolor": "#e2e8f0", "xtick.color": "#94a3b8",
                     "ytick.color": "#94a3b8", "axes.edgecolor": "#2e3250"})
fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
A, G, P = "#6c7ef8", "#34d399", "#fbbf24"

# Panel 1: E6 small translator
regs = ["MR\n(1ch)", "phi\n(16ch)", "MR+phi\n(17ch)"]
allm = [E6["mr_1ch"]["all"], E6["phi_16ch"]["all"], E6["both_17ch"]["all"]]
bonem = [E6["mr_1ch"]["bone"], E6["phi_16ch"]["bone"], E6["both_17ch"]["bone"]]
x = np.arange(3)
ax[0].bar(x-0.2, allm, 0.4, label="all-tissue", color=A)
ax[0].bar(x+0.2, bonem, 0.4, label="bone", color=P)
ax[0].set_xticks(x); ax[0].set_xticklabels(regs); ax[0].set_ylabel("Val MAE (HU)")
ax[0].set_title("E6 · SMALL translator, few subjects\nphi HELPS at low capacity/data", color="#34d399")
ax[0].legend(facecolor="#242740", edgecolor="#2e3250")
ax[0].axhline(allm[0], ls="--", lw=0.8, color="#64748b")

# Panel 2: E1 linear vs MLP (the crux)
labels = ["Linear\n(low-cap)", "MLP\n(high-cap)"]
phi = [R["E1_HUreg_in"]["phi_mr"]["mae"], R["E1_HUreg_in"]["phi_mr_mlp"]["mae"]]
mr = [R["E1_HUreg_in"]["mrctx"]["mae"], R["E1_HUreg_in"]["mrctx_mlp"]["mae"]]
x = np.arange(2)
ax[1].bar(x-0.2, phi, 0.4, label="phi(MR)", color=G)
ax[1].bar(x+0.2, mr, 0.4, label="raw-MR ctx", color="#94a3b8")
ax[1].set_xticks(x); ax[1].set_xticklabels(labels); ax[1].set_ylabel("HU MAE")
ax[1].set_title("E1 · phi's edge VANISHES with capacity\nlinear: phi wins | MLP: tied", color="#fbbf24")
ax[1].legend(facecolor="#242740", edgecolor="#2e3250")

# Panel 3: E2 informativeness + E3 void
cats = ["E2 seg\nmacroF1", "E2 bone\nF1", "E3 void\nbone-AUC"]
phiv = [R["E2_seg_in"]["phi_mr"]["macroF1"], R["E2_seg_in"]["phi_mr"]["boneF1"], R["E3_voidbone_in"]["phi_mr"]]
mrv = [R["E2_seg_in"]["mrctx"]["macroF1"], R["E2_seg_in"]["mrctx"]["boneF1"], R["E3_voidbone_in"]["mrctx"]]
x = np.arange(3)
ax[2].bar(x-0.2, phiv, 0.4, label="phi(MR)", color=G)
ax[2].bar(x+0.2, mrv, 0.4, label="raw-MR ctx", color="#94a3b8")
ax[2].set_xticks(x); ax[2].set_xticklabels(cats); ax[2].set_ylim(0, 1)
ax[2].set_title("E2/E3 · phi IS anatomically rich\n(linear probes) + disambiguates voids", color="#6c7ef8")
ax[2].legend(facecolor="#242740", edgecolor="#2e3250")

# Panel 4: E4 cross-modal
cats = ["matched\ncos", "shuffled\ncos", "phi_mr->phi_ct\nR2"]
vals = [R["E4_crossmodal_in"]["cos_matched"], R["E4_crossmodal_in"]["cos_shuffled"], R["E4_crossmodal_in"]["phi_mr_to_phi_ct_R2"]]
ax[3].bar(range(3), vals, color=[G, "#64748b", A])
ax[3].set_xticks(range(3)); ax[3].set_xticklabels(cats); ax[3].set_ylim(0, 1)
ax[3].set_title("E4 · Anatomix IS genuinely cross-modal\nphi(MR) ~ phi(CT), cos 0.93", color="#a78bfa")
for i, v in enumerate(vals): ax[3].text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("/tmp/amix_probe/figure.png", dpi=110, bbox_inches="tight")
print("saved figure.png")

# data-regime conceptual curve (schematic from our 2 points + their full result)
fig2, ax2 = plt.subplots(figsize=(6.5, 4))
N = np.array([1, 2, 3])  # small, (interp), large(real)
# benefit of phi (MR -> MR+phi), positive = helps. small: from E6. large: their result (~0 or negative)
benefit = np.array([ (191.8-165.6)/191.8*100, 6, -3 ])  # %, schematic
ax2.plot(N, benefit, "o-", color=G, lw=2.5, ms=9)
ax2.axhline(0, ls="--", color="#64748b")
ax2.set_xticks(N); ax2.set_xticklabels(["small model\nfew subjects\n(E6, measured)", "mid\n(hypothesized)", "full UNet\nall data\n(your result)"])
ax2.set_ylabel("phi benefit  (% MAE reduction)")
ax2.set_title("THE CROSSOVER (hypothesis to confirm)\nfoundation features help only below a capacity/data threshold", color="#34d399")
ax2.fill_between([0.5, 3.5], 0, 30, alpha=0.06, color=G)
ax2.fill_between([0.5, 3.5], -10, 0, alpha=0.06, color="#f87171")
ax2.set_xlim(0.6, 3.4); ax2.set_ylim(-10, 20)
plt.tight_layout(); plt.savefig("/tmp/amix_probe/figure2.png", dpi=110, bbox_inches="tight")
print("saved figure2.png")
