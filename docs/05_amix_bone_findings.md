# What 11 Experiments Measured: Anatomix & the Bone Bottleneck

**Source HTML:** _html/01a_amix_bone_findings.html
**Date:** undated
**TL;DR:** The best model is a strong-augmentation U-Net; no single-MR loss/architecture/feature/injection trick beats it because the only metric with headroom (bone) is information-limited, not optimization-limited. Anatomix features, perceptual losses, and bone-localizers are deterministic functions of the MR that a high-capacity translator re-derives, so they add no information and cannot lower the bone floor. The two levers that can move results: (1) add an information source (atlas/population-CT prior or multi-contrast), or (2) change the objective (generative bone where MAE is floored).

A standalone read of the empirical campaign run on SynthRAD data. Lit-review lives in 01_sota_mri2ct_translation.html; this is findings + plan. Frozen Anatomix v1.4, ~1.7M body voxels across all 5 regions, 11 experiments, multi-seed, proxy scale (no full training).

## 1. How these numbers were produced

Everything below is a proxy: a small 3D U-Net (MONAI BasicUNet), ~160 training patches (96^3) from 4 subjects/region, 1000 steps, light flip-augmentation, 2-3 seeds, held-out val subjects. The point is fast relative comparison, not final numbers. Metric is MAE in HU broken down by tissue (air <-300, soft -300..200, bone >200), because aggregate MAE is dominated by easy soft tissue and hides the bone story. Scripts: notebooks/e6-e11*.py.

Read magnitudes as relative: proxy-scale absolute MAEs are higher than the full pipeline; the orderings, frontier shape, and oracle-vs-realistic gap transfer. All flagged for full-scale confirmation (T0/T1 in section 5).

## 2. Part A: Why Anatomix doesn't help (E1-E6)

Anatomix features are NOT the problem. They are anatomically rich (linear seg macro-F1 0.40 vs 0.21 for raw MR) and genuinely cross-modal (phi(MR)*phi(CT) cosine 0.93). The foundation model works as designed.

The mechanism: phi's advantage vanishes with capacity. Linear readout phi wins (HU MAE 232 vs 294), but a small MLP ties it (138 vs 133). A high-capacity translator re-derives phi-equivalent features from raw MR itself. So feeding phi in is redundant, which is why concat-17ch doesn't beat plain MR at scale.

Law: phi helps only when the translator is too small/data-starved to re-derive it. Tiny model: MR+phi = 165.6 vs MR-only = 191.8 HU (-14%). This is also why "strong aug -> U-Net wins, weak aug -> Anatomix wins": augmentation and phi are substitutes for the same robustness.

## 3. Part B: The bone bottleneck (E7-E11)

Augmentation already supplies the robustness phi offered, so the target was the one place with headroom it can't touch: bone (MAE 3-5x soft; the metric that matters for dose/planning).

### The loss/architecture sweep (E7/E8): a robust tradeoff, no winner

| Intervention | bone | soft | verdict |
| --- | --- | --- | --- |
| L1 (baseline) | 457 | 107 | reference |
| L1 + GDL (edge) | 496 | 104 | bone worse |
| L1 + focal-frequency | 376 | 133 | on frontier |
| classification-HU | 408 | 87 | on frontier |
| multitask bone-head | 437 | 88 | on frontier |
| gated dual-head | 404 | 115 | on frontier |
| bone-weighted L1 | 290 | 143 | on frontier |

Every intervention only TRADES bone error for soft error; they all land on one Pareto frontier. The decompositions (gated, multitask) sit between the corners; none dominate. Bone is information-limited, not optimization-limited.

### The diagnostic (E10): it's localization, not magnitude

| Input (bone-weighted) | bone | soft | all |
| --- | --- | --- | --- |
| MR | 308 | 131 | 188 |
| MR + oracle bone-location | 145 (-53%) | 91 | 126 |

Just telling the model WHERE bone is halves the bone error (and frees capacity -> better soft). So the error is dominated by localization; the residual 145 HU is genuine density uncertainty.

### The ceiling (E11): that localization isn't in the MR

| Method | bone | captures of oracle gain |
| --- | --- | --- |
| MR baseline | 324 | - |
| localize(MR) -> translate | 317 | ~0% |
| localize(phi) -> translate | 301 | ~7% |
| oracle ceiling | 145 | 100% |

A realistic bone-localizer trained on MR/phi captures almost none of the oracle gain, because predicting bone location from MR hits the same information wall, and the translator already does it internally. The oracle's 53% came from GROUND-TRUTH location, i.e. information not in single-contrast MR.

The unified law (ties Part A + Part B together): the bone floor is set by what the MR physically contains about bone, period. Anything that is a deterministic function of the MR (Anatomix phi, perceptual losses, bone-localizers, frequency/edge losses) is re-derived by a high-capacity well-augmented translator. None add information; none lower the floor. phi-redundancy (Part A) and the bone-ceiling (Part B) are the same phenomenon.

## 4. Where the real novelty is

Two doors remain, both add what the MR lacks or change what you optimize. Plus the diagnostic itself.

1. Atlas / population-prior injection (add information). The oracle says bone-location is worth -53% and E11 says it isn't in the MR. A registered cohort/atlas CT (or a phi-retrieved nearest training CT) injects that distributional bone information, the one input that's NOT a function of the MR, so the translator can't re-derive it. Highest chance of a real bone win. Needs inter-subject registration.
2. Generative bone-residual (change the objective). L1 regresses bone to the mean -> blur. A diffusion/GAN refiner on the bone residual (conditioned on the regression sCT) matches the bone HU distribution -> sharp realistic cortices and better Dice/texture/dose even where mean MAE is floored. Reframes the metric from MAE (bounded) to realism/downstream (achievable).
3. The information-ceiling diagnostic (the general insight). The localization-vs-magnitude decomposition (oracle = -53%, recoverable as classification at AUC 0.97 but unusable because redundant with MR) is a sharp novel result explaining why a decade of auxiliary-feature/perceptual/cascade sCT methods give only marginal bone gains. Pairs as the motivation half of a paper whose method half is #1 or #2.

Don't bother (ruled out empirically): phi as input/loss; bone-weighting / GDL / FFL / classification / multitask / gated heads; localize-then-translate from MR or phi. All slide along the frontier or are redundant; none beat a strong-aug U-Net.

## 5. What to run next

| ID | Run | Proves |
| --- | --- | --- |
| T0 | Oracle confirmation: full pipeline + strong aug, baseline vs + oracle bone-mask channel (CT/TotalSegmentator); eval bone MAE, bone Dice, dose. | The -53% headroom is real at scale -> the motivation figure. |
| T1 | Realistic-localizer ceiling: strong full-scale bone-localizer (MR->bone, phi->bone) conditioning; measure fraction of T0 gap captured. | Confirms/bounds the E11 negative on real data. |
| T2 | (star) Atlas-prior injection: register cohort atlas (or phi-retrieve) per MR, add as conditioning channel, full train; ablate vs no-prior & oracle; eval bone + dose. | The main "results get better" bet. |
| T3 | (star) Generative bone-residual: diffusion/GAN refiner on bone residual over the regressor; eval Dice/texture/dose. | Distribution-matching beats mean-regression where MAE is floored. |
| T4 | Bone-priority + dose: sweep bone-weight on the full model, trace frontier, eval dose at each point. | Aggregate-MAE leaderboards mis-rank for the clinical task. |

Minimum viable paper: T0 + (T2 or T3). Run T0 & T1 first (cheap, real pipeline); if they replicate the proxy story, commit to the atlas prior (T2) or generative bone (T3). Paper = diagnostic (#3) as motivation -> method (#1/#2) as result -> dose/Dice downstream (not MAE alone) as evaluation. Venues: MICCAI / MedIA (clinical), or ICLR / NeurIPS (information-ceiling framing generalized).

## Reproducibility

Scripts & raw results: notebooks/e6_cnn.py, e7_sweep.py, e8_pareto.py, e9_frontier.py, e10_oracle.py, e11_localize.py (+ matching *_results.json). Full lit-review and SOTA context: _html/01_sota_mri2ct_translation.html.
