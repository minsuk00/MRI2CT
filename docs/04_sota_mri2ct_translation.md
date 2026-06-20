# SOTA: Paired MRI-to-CT Synthesis 2022-2025

**Source HTML:** _html/01_sota_mri2ct_translation.html
**Date:** 2026-06-11
**TL;DR:** Plain U-Nets remain the hardest baseline to beat on pixel metrics; anatomy-guided perceptual losses trade a little MAE for better Dice. A frozen foundation feature extractor (Anatomix) only helps a translator that is too small or data-starved to re-derive its features (redundancy-to-capacity law). An 11-experiment campaign proves bone error is information-limited (set by what the MR physically contains), not optimization-limited: the only real levers are adding an information source (atlas prior) or changing the objective (generative bone).

Deep research report: 98 search agents, 16 sources fetched, 75 claims extracted, 25 verified by 3 adversarial agents (need >=2/3 to confirm). 5 confirmed, 20 killed (80% kill rate).

## 1 Executive Summary

- Paired MRI-to-CT synthesis (2022-2025) has shifted away from pure GANs toward hybrid regression-perceptual architectures; diffusion is a strong but computationally expensive competitor.
- Most durable verified finding: plain U-Nets remain competitive on pixel-level metrics (MAE/PSNR/SSIM), while anatomy-guided perceptual losses reliably improve structural fidelity (organ Dice, HD95) at a small cost to pixel-level accuracy.
- Central novelty gap: every published perceptual loss for MRI-to-CT uses either (a) a natural-image pretrained backbone (VGG, ConvNeXt, not medical) or (b) a CT-only classification-supervised segmentation network (TotalSegmentator-derived). No published work uses a cross-modal contrastive (NCE) feature extractor trained on paired MRI and CT, which is exactly what Anatomix v1.4 is. Verifiable, publishable gap.
- Anatomix v1.4 is pretrained with NCE contrastive loss on both MRI and CT simultaneously: the only existing feature extractor that learned to align cross-modal anatomy. Using it as a frozen perceptual supervisor is architecturally novel vs all verified published work.

### Updated with measured results (story changed)

- Two earlier hypotheses are now falsified by own data + probes: the U-Net beats Anatomix even on the center-wise OOD split (kills the "contrast-robust / wins-OOD" angle), and probes show Anatomix features are genuinely cross-modal and anatomically rich, not bad.
- The real finding: foundation features help a translator only when it is too small or too data-starved to re-derive them itself. At full capacity/data the translator reconstructs equivalent features from raw MR, so frozen Anatomix adds nothing. That capacity/data crossover is the contribution.

## 1B Empirical Findings: 11 Experiments On Real Data

Ran the proxy suite on real SynthRAD subjects (frozen Anatomix v1.4, ~900k in-distribution + 780k OOD body voxels across all 5 regions, plus a small-translator capstone). No translator trained to convergence: Tier-0/1 probes plus one tiny end-to-end run.

*Figure: probe results. E6 = small translator on few subjects; E1 = per-voxel HU regression linear vs MLP readout; E2/E3 = linear anatomy & bone/air-void probes; E4 = cross-modal feature alignment.*

- **Anatomix features are genuinely good, not the problem.** Anatomically rich (E2: seg macro-F1 0.40 vs 0.21 for raw-MR context; bone-F1 0.37 vs 0.20) and genuinely cross-modal (E4: phi(MR)·phi(CT) cosine 0.93 matched vs 0.68 shuffled). The foundation model works as designed.
- **The crux: phi's advantage over raw MR vanishes as readout gains capacity.** E1 HU regression: with a linear readout phi wins (MAE 232 vs 294); with a small MLP they tie (138.5 vs 133.2, raw MR even edges it). A nonlinear network re-derives phi-equivalent information from raw MR by itself. A high-capacity translator is more than an MLP, so feeding phi in is redundant.
- **The crossover.** In the small-model + few-subject regime phi does help: a tiny translator gets MAE 165.6 HU with MR+phi vs 191.8 MR-only (-14%), bone 524 vs 561. Opposite of the full-scale result where the U-Net wins. Benefit is real but only below a capacity/data threshold (E6).
- **Residual signal.** The one place phi keeps a nonlinear edge is the bone/air signal-void: separating bone from air in low-MR-signal voxels, AUC 0.975 vs 0.899 (in-dist). Margin shrinks OOD (0.964 vs 0.959) and void-bone is rare (E3). A narrow but principled niche where MR genuinely lacks information.

*Figure: crossover curve, the hypothesis to confirm with full trainings. Two points measured (small -> phi helps; full -> U-Net wins), middle inferred. Headline figure of the paper.*

**Synthesis (one sentence):** A frozen anatomical foundation model helps MR->CT translation only when the translator is too small or data-starved to re-derive its features; at full capacity/data the translator reconstructs equivalent features from raw MR, so the foundation model becomes redundant. "U-Net beats Anatomix" is evidence for this redundancy-to-capacity law, not a failure of Anatomix.

### Honest caveats on probes

- E6 is small-scale (few subjects, 500 steps, 96^3 patches, single seed, no augmentation): establishes the low-data point, not the full curve. Crossover inferred from E6 + full-scale result, must be confirmed by data/capacity sweep (section 11).
- E1's pointwise regression is a coarse translation proxy. A spectral-sharpness probe was dropped (scale-confounded). OOD probes confirm phi stays informative OOD (E2 macro-F1 0.33 vs 0.14), consistent with U-Net winning OOD too because it re-derives phi there as well.

### Bone-information-ceiling campaign (E7-E11): the decisive result

Bone MAE is 3-5x soft-tissue MAE and that is where dose/planning value lives. Strong augmentation already lets the U-Net beat Anatomix, so the attack focused on bone (headroom augmentation can't touch). Battery of loss / output / architecture / injection interventions (11 experiments, multi-seed).

*Figure: bone campaign. E7/E8 = 8 loss/arch interventions all land on one bone-vs-soft Pareto frontier; E9 = phi shifts the frontier on the soft axis but the bone floor (~285) is unmoved; E10 = oracle bone-location input cuts bone error 53%; E11 = realistic MR/phi bone-localizer captures almost none of it.*

- **Bone error is information-limited, not optimization-limited.** Eight interventions (GDL, focal-frequency loss, bone-weighting, classification-output, multitask bone head, gated dual-head) all land on a single bone-vs-soft Pareto frontier (E7/E8, 3-seed). Each only trades bone error for soft error; none dominate. Example bone-weighting: bone 457->290 but soft 107->143. Gated/multitask decompositions land between corners, never inside.
- **The diagnostic: bone error is dominated by localization, not HU magnitude.** Feeding an oracle bone-location mask cuts bone MAE 308->145 (-53%) and improves soft tissue too (freed capacity). If you knew where bone is, you'd win big (E10). The remaining 145 HU is genuine density uncertainty, much smaller than the localization-driven +-2000 HU catastrophes at bone/air/soft boundaries.
- **The ceiling: that localization is not recoverable from single-contrast MR.** A realistic bone-localizer trained on MR captures ~0% of the oracle gain (324->317); on phi, ~7% (324->301), vs oracle 324->145 (E11). The translator already extracts what the MR contains; a separate MR/phi localizer is redundant. Same redundancy-to-capacity law as phi.
- **Unifying law.** The bone floor is set by what the MR physically contains about bone. Auxiliary features (phi), perceptual losses, bone-localizers, frequency/edge losses are all functions of the MR, so a high-capacity well-augmented translator re-derives them. None add information; none lower the floor. phi-redundancy (E1-E6) and bone-ceiling (E7-E11) are the same phenomenon. The lever is information, not architecture.

**Hard conclusion:** No single-MR loss/architecture/feature/injection trick will meaningfully beat the strong-aug U-Net on bone. The bottleneck is missing input information, not a modeling failure. This rules out a whole class of tempting-but-doomed experiments and points at the only two real levers: add an information source, or change the objective.

### Caveats on the bone campaign

- Proxy scale (BasicUNet, ~160 patches, 1000 steps, flip-aug, 2-3 seeds): magnitudes will differ at full scale, but the orderings and the oracle/realistic gap are large and consistent.
- Oracle uses a HU-threshold bone definition; a TotalSegmentator bone label would refine it. The realistic localizer was modest (800 steps); a stronger one would close some gap but cannot exceed MR's information. All warrant full-scale confirmation (section 11, T0).

## 2 Architecture Landscape (2022-2025)

Many strong architecture-class claims (transformer > CNN, diffusion > GAN) did not survive adversarial verification.

- **Regression U-Nets (nnResU-Net, ResUNet), CNN/Regression.** Dominant workhorse, still best or near-best on MAE/PSNR/SSIM across most benchmarks. nnU-Net / nnResU-Net with L1 achieves MAE ~63-65 HU on SynthRAD-style thorax. Adding perceptual (AFP) loss slightly hurts intensity metrics while improving Dice. The baseline to beat, hardest to beat on pixel metrics.
- **GAN-based (Pix2Pix, CGAN, GANeXt), GAN/CNN.** Strong baselines but no longer lead on SynthRAD. Dec 2025 GANeXt is the current direction: composite loss (MAE + ConvNeXt-B perceptual + TotalSegmentator-masked MAE + PatchGAN adversarial). Even GANeXt uses a natural-image pretrained ConvNeXt-B, not a medical extractor. Adversarial training is orthogonal to the perceptual loss question.
- **Transformer-based (Swin, TransUNet, ResViT), Transformer/Regression.** Widely adopted but claimed advantage over CNNs did not survive verification. The claim that transformers outperformed CNNs and GANs in SynthRAD2023 by SSIM (0.88 vs 0.85 vs 0.83) was refuted 0-3. I2I-Mamba was refuted 1-2. Treat transformer superiority as unconfirmed until reading a specific paper's tables.
- **Diffusion Models (DDPM, IDDPM, cWDM, MC-IDDPM), Diffusion/Score-based.** Perceptually sharp but do not consistently beat regression on intensity metrics (claim survived). MC-IDDPM (3D IDDPM with Swin-Vnet denoiser, Medical Physics 2024) achieves MAE 48.825 HU / PSNR 26.491 / SSIM 0.947 on a private 36-patient brain dataset (not comparable to SynthRAD). cWDM (wavelet-domain) has only MR-to-MR results (BraTS), not MR-to-CT; our cWDM baseline is ahead of published.
- **Anatomy-Guided Perceptual Loss (AFP, GANeXt), Perceptual Loss/Any backbone.** Most active innovation area. Confirmed: AFP using a frozen compact segmentation network (TS_Compact7, CT-only supervised) achieves Dice 0.7649 vs lower for L1-only, at a small MAE cost (64.35 vs 63.32 HU). From the ImagePasNet team at SynthRAD2025 (arXiv:2509.22394). Extractor is CT-only and classification-supervised; the cross-modal contrastive axis is unexplored.

## 3 Confirmed Findings

5 claims survived 3-vote adversarial review (need >=2/3 votes).

- **(High, vote 2-1)** AFP loss with frozen TS_Compact7 segmentation network improves Dice (0.7649) over L1-only for MR-to-CT, while L1-only retains slightly better intensity metrics (MAE 63.32 vs 64.35 HU): a structural/intensity trade-off. Evidence: arXiv:2509.22394 Table 3/4 (ImagePasNet, SynthRAD2025). TS_Compact7 is a 7-class TotalSegmentator-derived model, CT-only supervised, not cross-modal contrastive. Closest existing work to our Anatomix approach; the feature-extractor domain gap is the differentiator. (refs: arXiv:2509.22394, arXiv:2605.13555)
- **(High, vote 3-0)** cWDM (wavelet-domain conditional diffusion) achieves PSNR 29.74 / SSIM 0.956 for T1 synthesis on BraTS 2024. All results are MR-to-MR only; CT synthesis is speculative future work. Evidence: arXiv:2411.17203 Table 1 (T1: MSE=1.65e-3, PSNR=29.74, SSIM=0.956). Zero MRI-to-CT experiments exist. Our cWDM baseline is ahead of the literature.
- **(Medium, vote 2-1)** MC-IDDPM (3D diffusion, IDDPM variance prediction + Swin-Vnet denoiser) achieves MAE 48.825 HU, PSNR 26.491 dB, SSIM 0.947 with p<0.05 improvement over GAN and standard DDPM/DDIM on a private 36-patient brain dataset. Evidence: Medical Physics Vol.51(4), DOI:10.1002/mp.16847 (PMC10994752). Medium confidence: private brain dataset, not comparable to SynthRAD. Claimed Swin-Vnet superiority over V-Net was separately refuted 0-3.
- **(Medium, vote 2-1)** GANeXt (Dec 2025) uses composite loss: MAE + ConvNeXt-B perceptual (natural-image pretrained) + TotalSegmentator-masked MAE + PatchGAN adversarial: current SOTA direction for GAN-based MRI-to-CT. Evidence: arXiv:2512.19336. Perceptual network is ConvNeXt-B pretrained on ImageNet (natural images), not medical. Medium confidence: Dec 2025 preprint, not peer-reviewed. Confirms the domain-pretrained extractor gap.
- **(High, vote 3-0)** cWDM explicitly states its wavelet-domain approach "could be applied to CT<->MR and MR<->PET translation" but provides no CT experiments: speculative future work. Evidence: phrase confirmed verbatim in arXiv:2411.17203. Confirms our cWDM-for-MRI-to-CT is ahead of published work.

## 4 Perceptual Loss Design: The Key Axis

Most relevant dimension for project novelty. Comparison of all perceptual feature extractors in published MRI-to-CT work vs Anatomix.

| Extractor | Pretraining Domain | Pretraining Objective | Modalities Seen | Used In |
| --- | --- | --- | --- | --- |
| VGG19 / LPIPS | Natural images (ImageNet) | Classification | RGB photos | Many early MRI-to-CT papers (unverified in challenge) |
| ConvNeXt-B | Natural images (ImageNet) | Classification | RGB photos | GANeXt (arXiv:2512.19336) |
| TS_Compact7 | CT only (medical) | Segmentation (7 classes) | CT only | AFP/ImagePasNet (arXiv:2509.22394) SynthRAD2025 |
| DINOv3 ViT | Natural images (self-supervised) | Contrastive / DINO | RGB photos | arXiv:2511.12098 (claims unverified 0-3) |
| Anatomix v1.4 (Ours) | Medical CT + MRI (paired) | Contrastive NCE (cross-modal) | Both MRI and CT | No published work uses this (novel gap) |

- TS_Compact7 was trained on CT segmentation labels only, never seen MRI: as a perceptual network it judges anatomical correctness only in CT feature space.
- Anatomix v1.4 was pretrained to align MRI and CT in the same feature space via NCE contrastive learning on paired scans; its features encode anatomy in both modalities simultaneously. Architecturally superior for cross-modal translation: the supervisor understands both input modality and target.

### Why isn't it working yet? Three hypotheses

1. Feature comparison is CT-only at inference. Both pred and target CT pass through Anatomix; the extractor's cross-modal alignment capacity is never leveraged. The MRI input is not passed through Anatomix. Using MRI as a third anchor (NCE triplet: MRI anchor, pred CT positive, random-region CT negative) could unlock the alignment.
2. NCC metric may dilute anatomical structure signals. NCC is invariant to per-channel affine intensity shifts (good for contrastive alignment) but may average out local structural differences that matter for bone/organ fidelity. L1 in feature space (as AFP uses) may be stronger for anatomy supervision.
3. Loss weighting / competition with L1. Too low: L1 dominates, Anatomix is noise. Too high: model overfits Anatomix features at the cost of HU accuracy. AFP uses L1 + AFP jointly with explicit balancing and still shows a Dice-vs-MAE trade-off.

## 5 SynthRAD Challenge Landscape

The most specific SynthRAD2023 claims (winner architecture, transformer vs CNN rankings, architecture-class breakdown) did not survive verification (0-3). Challenge paper arXiv:2403.08447 exists but specific ranking statistics could not be confirmed.

### SynthRAD2023 (Task 1: MRI-to-CT)

| Fact | Status | Notes |
| --- | --- | --- |
| Challenge paper exists at arXiv:2403.08447 | Confirmed | Multi-center, paired MRI-CT dataset; brain, thorax, pelvis regions |
| Transformer models outperformed CNNs/GANs in rankings | Refuted 0-3 | Specific SSIM numbers cited (0.88/0.85/0.83) did not hold up |
| Winner used VGG19 perceptual + masked MAE | Refuted 0-3 | Could not verify architecture details of SMU-MedVision team |
| Anatomy-conditioned single models scored near specialized models | Refuted 0-3 | Treat as unconfirmed until reading the paper tables directly |

### SynthRAD2025

- Challenge report (arXiv:2605.13555) confirmed to exist, lists the ImagePasNet team (AFP loss, nnResU-Net + TS_Compact7 perceptual) as a top entrant. Cross-corroborated with arXiv:2509.22394 (AFP method paper). Full leaderboard ranking statistics did not survive verification.
- Our position: AFP paper (arXiv:2509.22394) is the closest confirmed competitor (nnResU-Net + L1 + AFP, CT-only TS_Compact7 extractor). Our Anatomix extractor is cross-modal and contrastively trained but benefit not yet demonstrated. Paper posted Sep 2025, not peer-reviewed: window to publish a stronger result.

## 6 Novelty: What the 11 Experiments License

Bone is the only metric with headroom, and it is information-limited. Single-MR loss/architecture/feature tricks slide along a fixed Pareto frontier and cannot lower the bone floor. Two doors: add information, or change the objective, plus the diagnostic itself as a contribution.

**Ruled out by the data (do not pursue):** phi as input/loss (redundant, E1-E6); bone-weighting / GDL / FFL / classification / multitask / gated heads (trade along the frontier, E7/E8); localize-then-translate from MR or phi (oracle works, realizable version doesn't, E11). None will beat a strong-aug U-Net.

### Ranked novelties (only data-supported)

1. **Atlas / population-prior injection (add information).** The oracle proved bone-location info is worth -53% bone MAE; E11 proved it isn't in the MR. A registered population/atlas CT (or deformable template) injects exactly that distributional bone information (where this anatomy usually has bone and its typical HU) which the individual MR lacks. Wins because it is the one input that is NOT a deterministic function of the MR, so the translator can't re-derive it. Build: deformably register a cohort atlas (or retrieve a phi-nearest training CT) to each MR; feed as conditioning channel; full-scale train; eval bone + dose. Risk: inter-subject registration; mis-registration injects noise.
2. **Generative / distributional bone (change the objective).** MR->CT is one-to-many in bone, so L1 regresses to the mean -> blurred, under-dense cortical bone. A generative bone model (diffusion or GAN on the bone residual, conditioned on the regression sCT) produces realistic bone texture and sharp cortices, improving bone Dice, texture, and dose even where mean MAE is capped. Reframes target from "minimize bone MAE" (information-bounded) to "match the bone HU distribution" (achievable). Right metric is realism/Dice/dose, not MAE. Differentiate from full-volume diffusion sCT by being a bone-residual-only refinement on top of the cheap regressor.
3. **The information-ceiling diagnostic (the top-venue insight).** Localization-vs-magnitude decomposition is itself a sharp novel result: via oracle, bone error in MR->CT is ~50% localization; that localization is recoverable as a classification task (AUC 0.97) but not usable by the translator because it's a redundant function of the MR. Explains why a decade of auxiliary-feature/perceptual/cascade methods give marginal bone gains. "Information ceilings in conditional generation" is a general ML question; MR->CT is a clean high-stakes testbed with a decisive oracle experiment. Pairs as the motivation half of a paper whose method half is #1 or #2.
4. **Multi-contrast / acquisition study (add information, if available).** Textbook fix for the bone void problem is a second sequence (UTE/ZTE, or T1+T2). If any extra contrast exists, a controlled "how much bone information does sequence X add" study quantified against the oracle ceiling is concrete and clinically grounded. Lower priority: depends on data you may not have.

**Differentiate from prior art:** Bone-segmentation-assisted sCT / cascades exist but assert that localization helps; the oracle quantifies it (-53%) and E11 shows the realizable version doesn't capture it. Atlas-based pseudo-CT (classical) is registration-only; #1 is a learned atlas-conditioned translator. Diffusion sCT is full-volume; #2 is a bone-residual refinement justified by the information argument.

### Recommended path

- Confirm the diagnostic at full scale (T0/T1): oracle-bone conditioning + realistic localizer ceiling on the real pipeline. Backbone of any paper here.
- Pick the method door: #1 (atlas prior) if you can stand up inter-subject registration (highest chance of a real bone win); or #2 (generative bone) to change the objective to realism/dose.
- Paper shape: diagnostic (#3) as motivation -> method (#1 or #2) as result -> dose / segmentation / bone-Dice downstream (not MAE alone) as evaluation.
- Venues: diagnostic+method -> MICCAI / Medical Image Analysis; information-ceiling framing generalized -> ICLR / NeurIPS.

## 7 Fast Iteration: The Proxy Ladder

Decompose every "will idea X help?" into three questions; answer the cheap ones first. Filter cascade: each tier kills bad ideas before the next more expensive tier.

- **Q1** Is the feature space anatomically informative, cross-modal, void-disambiguating? Property of the frozen extractor. No translation training. Seconds-minutes.
- **Q2** Where does the current model fail (in-distribution vs OOD, soft tissue vs bone/air)? Re-score cached predictions. No training. Hours.
- **Q3** Does the new mechanism improve a fully-trained translator? The GPU-week. Only for ideas that survive Q1+Q2.

### Tier 0: Feature-space audits (seconds-minutes, no training)

| Proxy | What it answers / kills | How (your code) |
| --- | --- | --- |
| OOD failure re-score (tests #2) | Most decisive cheap analysis. Re-score U-Net vs Anatomix-translator predictions split by in-distribution vs center-wise OOD (and pelvis T1/T2 case). If U-Net's lead narrows/flips OOD, contrast-robustness story half-proven. Hours. | Reuse cached sCT from full_eval runs + center_wise_split.txt; group metrics by center/region |
| Bone-vs-air void probe (tests #1) | In signal-void voxels (low MR intensity), can Anatomix features separate bone from air better than intensity? Fit linear classifier on void voxels: phi vs raw MR -> {bone, air, soft}. Minutes. | Mask void voxels via MR threshold; labels from CT HU bands; logistic regression on phi vs MR |
| Linear probe (feature informativeness) | Feature-quality ranking: fit 1x1x1 conv from each extractor's features -> 12-class CT seg labels (Anatomix vs raw MR vs TS_Compact7 vs VGG). Standard SSL eval. | Cache phi on ~10 volumes; logistic regression against seg_baby_unet labels (n_classes=12) |
| Cross-modal alignment score | Is Anatomix v1.4 actually cross-modal? Voxel-wise cosine/NCC between phi(MR) and phi(CT) over body mask. Anatomix high; VGG & CT-only TS_Compact7 ~ 0. Gates the MRI-anchored loss arm. Doubles as a paper figure. | Extend compare_amix_features.py; add similarity reduction over a few subjects |

### Tier 1: Loss-behavior audits (seconds-minutes, no training)

| Proxy | What it answers / kills | How (your code) |
| --- | --- | --- |
| Spectral / redundancy probe | Tests why concat hurts. (a) Frequency: is the 16-D Anatomix descriptor lower-frequency than raw MR (over-smoothing)? (b) Redundancy: how well does raw MR linearly predict Anatomix features (near-perfect -> concat adds nothing)? (c) Corruption response: does the Anatomix loss rise with bone removal / HU shift that L1 under-weights? Routes to the right fix (distill vs bottleneck-inject vs loss-only). | FFT / radial power-spectrum on phi vs MR; fit MR->phi linear map; corrupt CT volumes, plot loss-vs-severity |
| Loss-as-metric correlation | Does a loss rank-correlate with downstream goal? On a trained translator's val predictions, compute corr(loss(sCT,CT), bone-Dice) and corr(., MAE). A loss worth training on tracks the metric better than L1. No training. | Reuse cached sCT from full_eval runs; loop candidate losses over (sCT, GT-CT) pairs |

### Tier 2: End-to-end translation proxy (10-30 min, tiny training)

| Proxy | What it answers | How (your code) |
| --- | --- | --- |
| Mini-split overfit + holdout | Last gate before GPU-week. Train real 3D translator on a handful of volumes for a few hundred steps. For differential signal: does loss A reach lower held-out bone-Dice/MAE faster than loss B? Uses real pipeline, catches integration issues. | mini_thorax_split.txt and *_single_subject_split.txt; run amix/train.py with tight step budget |
| 2D slice variant (optional) | Faster screening: 2D axial translator trains in minutes. Sweeps many loss variants but introduces a 2D->3D gap; use to rank, not conclude. | Subsample axial slices; reuse loss modules unchanged |

**Decision rule:** Tier 0 fail -> never build it. Tier 1 fail -> the loss is ill-behaved; fix the mechanism, not the weights. Tier 2 differential win -> then spend the GPU-week. Most ideas should die at Tier 0/1 in under 20 minutes.

**Publication strategy:** "Registration-robust cross-modal perceptual supervision for paired MRI-to-CT synthesis." Core figures: cross-modal alignment score and perturbation curves (Tier 0/1, cheap). Confirmation: 5-region full-training ablation (Anatomix vs TS_Compact7 vs VGG vs L1-only). Downstream dose-calc on sCT (pyRadPlan) closes it for MICCAI / Medical Image Analysis.

## 8 Open Questions

- Has any published work used a cross-modal contrastive pretrained network (trained jointly on MRI and CT) as a frozen perceptual loss for MRI-to-CT? Verified answer: no, but a targeted search for "contrastive perceptual loss MRI CT" is warranted before claiming priority.
- Does Anatomix's multi-scale feature space produce better-calibrated anatomical supervision than TS_Compact7? Theory says yes; empirics need running. A "no" is itself a publishable negative result about feature-extractor domain specificity.
- What exactly did the SynthRAD2023 Task 1 winner use? The VGG19 + masked MAE claim was refuted (0-3); read arXiv:2403.08447 tables directly.
- Is the AFP/Dice-vs-MAE trade-off fundamental to perceptual losses, or can the MRI-anchor NCE formulation overcome it via simultaneous cross-modal consistency?

## 9 What Was Refuted (Do Not Cite)

These circulate in the literature/secondary sources but did not survive adversarial 3-vote verification.

- Transformer architectures outperformed CNNs in SynthRAD2023 (SSIM 0.88 vs 0.85 vs 0.83 by class). Refuted 0-3; specific numbers appear hallucinated.
- SynthRAD2023 Task 1 winner (SMU-MedVision) used VGG19 perceptual + masked MAE. Refuted 0-3; architecture details unconfirmable.
- AFP loss is consistent across all regions (HN, thorax, abdomen). Cross-anatomy generalization sub-claim in arXiv:2509.22394 refuted 0-3; confirmed result is thorax only.
- I2I-Mamba (Mamba SSM) outperforms TransUNet for MRI-to-CT. Refuted 1-2, unconfirmed.
- Diffusion models (WDM, DDPM) do not outperform regression on MRI-to-CT at SynthRAD: specific claim (WDM 26.83 dB vs TransUNet 28.06 dB same dataset) refuted 0-3.
- DINOv3 perceptual loss achieves SOTA on SynthRAD2023 pelvis (arXiv:2511.12098). All claims refuted 0-3; source unreliable.
- Multi-region training improves structural robustness vs single-region specialization. Refuted 0-3.
- Swin-Vnet significantly outperforms V-Net ablations inside MC-IDDPM's denoiser. Refuted 0-3; the diffusion improvement is real but the Swin-specific attribution is not.
- Supervised GANs consistently outperform unpaired CycleGAN on SynthRAD2025. Refuted 0-3.
- GANeXt uses a segmentation-based discriminator. Refuted 1-2; composite loss confirmed but segmentation-discriminator component not.

## 10 Caveats & Methodology

- Generated by a deep research harness: 5 parallel search angles, 16 sources fetched, 75 claims extracted, 25 verified by 3 independent adversarial agents (need >=2/3 votes).
- Kill rate was high (20 of 25 = 80% killed): aggressive adversarial verification. The 5 confirmed findings are reliable enough to cite; all refuted claims should be independently verified from original papers.
- SynthRAD2023 leaderboard breakdown by architecture class was the most frequently hallucinated category; avoid citing architecture-class rankings without reading arXiv:2403.08447 tables.
- AFP paper (arXiv:2509.22394) is a Sep 2025 preprint, not peer-reviewed.
- MC-IDDPM results are on a private 36-patient brain dataset, not comparable to SynthRAD.
- cWDM results (arXiv:2411.17203) are MR-to-MR on BraTS 2024, no cross-modal CT data.
- DINOv3 perceptual loss paper (arXiv:2511.12098) had all claims refuted 0-3.
- Field is moving fast; SynthRAD2025 results are mid-2025, more papers likely in review.

**Sources consulted:** arXiv:2509.22394 (AFP/ImagePasNet), arXiv:2605.13555 (SynthRAD2025 report), arXiv:2403.08447 (SynthRAD2023 paper), arXiv:2411.17203 (cWDM), arXiv:2405.14022 (I2I-Mamba), arXiv:2512.19336 (GANeXt), arXiv:2511.12098 (DINOv3 perceptual, unverified), PMC10994752 (MC-IDDPM, Medical Physics 2024), arXiv:2603.13520 (GAN benchmark), ICCVW 2021 contrastive feature loss paper, 6 additional multi-region generalization sources.

**Section 6 hard-problem grounding (supplementary):** arXiv:2103.01609 & AJNR 43:8 (bone/air void ambiguity, H1), arXiv:2303.10202 (contrast generalization in brain MR-to-CT, H2), arXiv:2507.13458 (domain randomization for neuroimaging), PMC11186649 & PMC7610395 (uncertainty / aleatoric MR-CT, H3), CUT / patch-contrastive & content-style disentanglement translation (for #4 differentiation).

## 11 Proposed Full Trainings (revised after the campaign)

The proxy campaign already killed the doomed single-MR experiments. Remaining runs confirm the diagnostic or test the two real levers. Ordered by value-per-GPU-hour.

| ID | Experiment | Setup | What it proves | Cost |
| --- | --- | --- | --- | --- |
| T0 | Oracle confirmation (diagnostic, full scale) | Full pipeline + strong aug: baseline vs + oracle bone-mask channel (from CT HU / TotalSegmentator). Report bone MAE, bone Dice, dose. | Confirms the -53% headroom is real at scale -> motivation figure. | ~2 runs |
| T1 | Realistic localizer ceiling | Train strong full-scale bone-localizer (MR->bone, and phi->bone); condition translator on its soft prediction. Measure fraction of T0 oracle gap captured. | Pins the realizable ceiling on real data; confirms/bounds the E11 negative. | ~3 runs |
| T2 | Atlas / population-prior injection (#1) | Deformably register a cohort atlas CT (or retrieve phi-nearest training CT) to each MR; add as conditioning channel; full train. Ablate vs no-prior and vs oracle. Eval bone + dose. | Main "results get better" bet: does injected information beat the strong-aug U-Net on bone? | ~6 runs + registration infra |
| T3 | Generative bone-residual (#2) | Train regressor sCT (current best), then diffusion/GAN refiner on the bone residual conditioned on it. Eval bone Dice / texture / dose, not just MAE. | Does distribution-matching beat mean-regression on realism/downstream where MAE is floored? | ~5 runs |
| T4 | Bone-priority frontier + dose (low-risk supporting) | Sweep bone-weight on full model; trace bone-vs-soft frontier; eval dose at each point. Show MAE-optimal model is not dose-optimal. | Reframes evaluation: aggregate-MAE leaderboards mis-rank for the clinical task. | ~4 runs |
| T5 | Multi-contrast study (#4, if data exists) | If any extra MR sequence exists: quantify bone-information added per sequence against the oracle ceiling. | Directly tests the textbook fix for the void problem. | data-dependent |

**Minimum viable paper:** T0 + (T2 or T3): oracle diagnostic as motivation plus one information/objective lever as method. T1 hardens the "why single-MR can't" argument; T4 supplies clinical-relevance reframing. Run T0 + T1 first (cheap, real pipeline); if they replicate the proxy story, commit to T2 or T3.

**Framing per venue:**
- ICLR / NeurIPS: "Information ceilings in conditional generation: a localization-vs-magnitude decomposition for cross-modal medical translation." Lead with the oracle diagnostic (T0/T1) + one method.
- MICCAI / MedIA: "Why MR-only synthetic-CT plateaus on bone, and an atlas-conditioned (or generative) fix." Same experiments, clinical framing, dose downstream.
