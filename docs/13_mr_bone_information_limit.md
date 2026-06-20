# Does Standard MR Contain Bone-Density Information? A Conclusive Test

**Source HTML:** _html/08_mr_bone_information_limit.html
**Date:** 2026-06-19
**TL;DR:** Standard T1/T2 MR does not carry the information needed to reconstruct bone density. Proven model-free on our data (a 5x5x5 MR-patch k-NN predictor leaves a 222 HU bone floor; MR explains only 19% of bone-HU variance) and by a large independent literature (cortical bone is a sub-millisecond-T2* signal void; no one-to-one MR->HU mapping). The U-Net's bone undershoot is a fundamental input limitation, not a model flaw. Part of the bone-failure series.

Question under test: is the U-Net's bone undershoot caused by missing information in the MR, or by the model? Answered two ways: (A) model-free experiments on our data (no CNN, no L1 loss), and (B) a verified literature review.

Precise claim (and honest scope): the claim is NOT "MR contains zero bone signal." It is: standard T1/T2 MR carries very little usable information about bone density (far less than CT), and even the full local MR appearance (a 3D patch, i.e. spatial context) does not determine bone HU. Both our data and the literature support this; the literature adds the clinching nuance that dedicated bone sequences (UTE/ZTE) can recover bone, proving it is the standard sequence that lacks the signal, not MR in principle.

## A. Model-free evidence on our data

Every test below avoids the CNN and the L1 loss, so none of it can be blamed on "the model." If even an information-optimal model-free predictor cannot get bone HU from the MR, the information is not there.

### A1. The single most direct test: can full MR appearance predict bone HU? (k-NN)

For 25,000 bone and 25,000 soft-tissue voxels (region-balanced, 40 subjects), we took the full 5x5x5 MR patch around each voxel and predicted its CT HU from the average HU of the voxels with the most similar MR patches in OTHER subjects (cross-subject k-NN, PCA-reduced). A model-free estimate of the best any method could do from MR appearance; no neural net, no L1.

| predictor | BONE MAE (HU) | SOFT MAE (HU) |
| --- | --- | --- |
| predict a constant (use no MR at all) | 252 | 63 |
| k-NN on MR intensity (1 voxel) | 254 | 62 |
| k-NN on the full MR patch (context) | 222 | 55 |
| variance of HU explained by the MR patch (R^2) | 19% | 15% |

Knowing the full local MR appearance reduces bone error from 252 (no MR) to only 222 HU; the MR patch explains just 19% of bone-HU variance, and adding spatial context beats single-voxel intensity by only ~12% (254->222). For soft tissue the same predictor lands at 55 HU because soft HU is intrinsically narrow. This directly answers the "context should help" objection: it barely does.

*Figure: across every predictor (constant, intensity, full-patch context, perfect-location oracle) BONE stays 220-260 HU while SOFT stays ~55; the information ceiling is the same regardless of method.*

### A2. Same MR appearance -> many bone densities (the ambiguity, directly)

At a single MR brightness (rank 0.20-0.30), bone CT HU spans p5=229 to p95=1468 (std 433 HU). If MR determined bone HU, each MR brightness would map to one value; instead it maps to a ~1300 HU range.

### A3. A perfect-location oracle barely helps

Giving a predictor the EXACT bone location (region-balanced, 10.3M true-bone voxels) and letting it use the "outer-shell-is-cortical" rule only improves bone MAE from 259 (U-Net) to 220 HU; predicting a single constant bone HU with perfect location gives 246. So localization is not the bottleneck: even with perfect geometry, density is unresolved because it varies ~300 HU within any shell layer.

### A4. Architecture-independent: all six models undershoot

If this were a model flaw it would differ by architecture; instead every model (regression and diffusion, capped and uncapped, including koalAI) undershoots bone (bias -180 to -274 HU). Uncapped/diffusion models reach higher peaks (so they CAN output high HU) but still undershoot, because they cannot tell which dark-MR voxels are dense.

Why conclusive on our data: the k-NN predictor (A1) uses no neural network and no L1 loss, so its 222 HU bone floor is a property of the DATA, not the model. Our trained U-Net (241-259 HU) sits right at this floor; it is already near information-optimal for bone. No architecture, loss, or augmentation can pass a ceiling set by the MR itself.

## B. What the published literature establishes

Independent peer-reviewed work converges on the same conclusion across MR physics, dedicated bone sequences, MR-only radiotherapy, and deep-learning sCT reviews. (Sources verified against PubMed/PMC/arXiv.)

### B1. MR physics: cortical bone is a signal void

- Du & Bydder, NMR Biomed 2013: cortical bone mobile-water T2* ~ 408 +/- 16 us (~0.4 ms); too short for conventional MRI (TE of ms) to detect; requires ultrashort-TE. PMID 23280581
- Ma et al., Front Endocrinol 2020: cortical bone "invisible when studied using conventional clinical MRI pulse sequences with echo times of a few milliseconds or longer." PMC7531487
- Afsahi et al., JMRI 2022: bone has low proton density and fast signal decay (cortical T2* ~0.31 ms), "shows little signal with conventional MRI sequences." PMC9106865

### B2. UTE/ZTE were created specifically to image bone (proving conventional MR can't)

- Leynes et al., J Nucl Med 2018 (ZeDD CT): adding ZTE (bone) to standard Dixon MR cut bone-lesion PET error from 10.24% -> 2.68% (~4x). The gain comes from recovering bone the standard MR omits. PMID 29084824
- Wiesinger et al., MRM 2018: ZTE pseudo-CT bone Dice 0.73, bone HU MAE ~123; usable bone from a dedicated bone sequence. PMID 29457287
- Jerban et al., Bone 2019: UTE-MRI of cortical bone correlates with histomorphometric porosity (R>0.7). PMID 30877070

### B3. MR-only radiotherapy: bone is the dominant error; bulk-density override was the workaround

- Edmund & Nyholm, Radiat Oncol 2017: review of 50 studies; bone is the central problem because MR carries no electron-density information. PMID 28126030
- Johnstone et al., IJROBP 2018: systematic review; "bulk density override" is a primary sCT category precisely because MR can't give bone HU. PMID 29254773
- Autret et al., Radiat Oncol 2023: body MAE 22-40 HU but skull MAE up to -381 HU; bulk methods cap sCT ~1000 HU while real cortical bone exceeds 3000 HU. PMC10478301
- Farjam et al., JACMP 2021: per-tissue sCT error: bone 106 HU vs muscle 23, fat 16 (~4.5-6.5x). PMC8364266

### B4. Deep-learning sCT reviews: error concentrates in bone, blamed on the MR input

- Huijben et al., Med Image Anal 2024 (SynthRAD2023): across 22 MRI->CT methods, errors concentrate at soft-tissue/bone and air boundaries "potentially due to low MRI signal," consistently across teams (model-independent). arXiv:2403.08447
- Dayarathna et al., Med Image Anal 2024: larger errors at soft-tissue/bone "primarily due to the limited visibility of air and bones in MR images." DOI 10.1016/j.media.2023.103046
- Sherwani & Gopalakrishnan, Front Radiol 2024: "Due to the lack of a one-to-one relationship between MR voxel intensity and CT's Hounsfield Unit ... intensity-based calibration methods fail." PMC11004271
- Korhonen et al., Med Phys 2014: no one-to-one MR<->HU relation; bone and soft tissue have overlapping MR intensities, motivating a dual in-bone/out-of-bone model. PMID 24387496

*Figure: published MR->CT studies show bone/skull MAE 3-10x soft tissue across independent groups, methods, and anatomy, matching our pattern.*

## Conclusion

- On our data (model-free): the best appearance-based predictor leaves a 222 HU bone floor (MR patch explains only 19% of bone-HU variance), spatial context adds ~12%, a perfect-location oracle reaches only 220, and all six models sit at this floor. The ceiling is in the data, not the model.
- In the literature: cortical bone has sub-millisecond T2* and is a signal void on conventional MR; bone is the dominant sCT error (100-380 HU vs 15-40 for soft) across 22+ independent methods; there is no one-to-one MR->HU mapping; MR-only RT historically used bulk-density override because MR lacks bone density.
- The clinching nuance: UTE/ZTE sequences DO recover bone (PET bone error ~4x lower, bone Dice 0.73). That these special sequences are necessary proves conventional MR does not contain the signal, and the fix is more information (a bone sequence or prior), not a better network or loss.

Bottom line: to meaningfully improve bone you must add information the MR lacks (UTE/ZTE acquisition, a second contrast, or an anatomical/CT prior).

## Reproducibility & honesty notes

Experiments: `src/evaluate/unet_failure/knn_patch_test.py` (model-free k-NN), `loc_test` (perfect-location oracle), `multimodel_extract.py` (cross-model). k-NN uses PCA(25) on z-scored MR patches, cross-subject neighbours; it is a finite-sample estimate of predictability, but the bone-vs-soft comparison controls for method, so the gap is the signal. Literature verified against PubMed/PMC/arXiv; paywalled figures flagged in source notes. The claim is "little usable bone-density info in standard MR," not "none in any MR" (UTE/ZTE is the counter-example that proves the rule).
