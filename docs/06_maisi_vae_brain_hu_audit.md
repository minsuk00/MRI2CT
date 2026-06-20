# MAISI VAE brain HU-inflation audit

**Source HTML:** _html/02_maisi_vae_brain_hu_audit.html
**Date:** 2026-06-11
**TL;DR:** The original conclusion (the NV-Generate-CT VAE inflates brain soft tissue ~+65 HU via skull bleed; r=0.88 with bone proximity; 8x compression) is RETRACTED. It was a sliding-window inference artifact in the audit's own harness, not VAE behavior. Run whole-volume, the VAE mildly DEFLATES brain (~-16 HU sampled, -58 HU at latent mean), similar across regions, with no brightening. The brain white-out is a bug in our encoding harness (oversized SlidingWindowInferer ROI zero-pads small brain volumes, MaisiGroupNorm3D global stats flooded by air -> uniform whole-brain brightening). Compression is 4x per axis, not 8x.

## RETRACTION NOTICE (supersedes sections 1-9)

The central claim of the original report ("the VAE inflates brain soft tissue by ~+65 HU via skull bleed; r=0.88 with bone proximity; 8x compression") is a sliding-window inference artifact in this audit's own harness, NOT VAE behavior. Controls (`~/vae_audit/control.py`, `control2.py`) show:

- VAE run whole-volume (no sliding window), deterministic mean: brain -58 HU, thorax -46, pelvis -28: mild deflation, similar across regions. The VAE does not inflate brain.
- The +127 HU brain "inflation" appears only with the SlidingWindowInferer (enc ROI 384x352x256 >> the 160^3 brain volume -> heavy zero-padding). A harness/padding artifact, worst for the smallest (brain) volumes.
- Compression is 4x per axis, not 8x. The encoder is stochastic (samples noise; two runs differ up to ~1388 HU).

Figures/numbers in sections 1-9 below were computed on harness-corrupted reconstructions. The real-world brain-prediction white-out is real but its cause is reopened (likely diffusion-side, unverified). Sections 1-9 are kept for the record only.

## CORRECTED FINDING

The VAE is fine. The brain white-out is a bug in OUR encoding harness.

- The VAE reconstructs brain faithfully when run whole-volume (no sliding window): recon ~= GT, small diff. Soft-tissue error: brain -50...-67 HU (mild deflation), thorax -46. No brightening anywhere.
- The +65...+127 HU "inflation" came from a SlidingWindowInferer with ROI 384x352x256 wrapped around ~160^3 brain volumes, which zero-pads the brain to ~91% air. The VAE has 26 MaisiGroupNorm3D layers that normalize over the whole spatial extent; the air floods their statistics, drags the global mean down, and re-centers every brain voxel upward -> uniform whole-brain brightening. Demonstrated with NO sliding window via a manual air-padding sweep. The shift is global (whole brain, not near-bone) because GroupNorm is global, which is exactly why the bone-bleed story below is wrong.
- This corrupts OUR pipeline via the training targets. Training builds the diffusion target latent with the same oversized-ROI encode (`trainer.py:451` `_encode_sliding_window`), so the model is trained to predict bright-brain latents. Verified: clean whole-vol VAE -50...-67 HU vs SW-encode target +39...+82 vs real MAISI prediction +72...+90: the prediction matches the corrupted target, not the clean VAE. (Decode is innocent: brain latents are small enough that inference decodes whole-volume.)
- **Fix:** use the official codebase's `dynamic_infer` (`utils.py:647`): run whole-volume when the volume fits the ROI, else clamp the ROI to the volume, never pad beyond the volume. Apply in `_encode_sliding_window` and `encode_all_volumes.py`, then re-encode + retrain (the bias is baked into current weights). Residual: the VAE still mildly deflates brain (~-58 HU), a small separate limitation, not the white-out.

*Figure: whole-volume VAE reconstruction (recon ~= GT), showing the VAE has no brightening bug.*
*Figure: clean VAE vs the corrupted training target vs the real prediction.*

### Mean vs samples (the VAE is variational)

The encoder outputs a Gaussian over the latent code, z ~ N(z_mu, z_sigma^2), not over HU. The clean figure decoded the latent mean z_mu, giving brain ~= -58 HU, but that is unrepresentative: the decoder is nonlinear, so decode(mean of z) != mean of decode(z). The pipeline samples (`encode_stage_2_inputs` = z_mu + z_sigma*eps), and sampled reconstructions cluster tightly at ~-16 HU (GT 35); the -58 point sits below the whole sample cluster. The VAE's true brain reconstruction is a mild, stable ~-16 HU deflation, not -58.

| reconstruction | brain soft err |
| --- | --- |
| decode(z_mu), latent mean | -58 HU (outlier) |
| sample 1 / 2 / 3 / 4 | -14 / -17 / -18 / -18 |
| E[decode], avg of 12 samples | -16 HU (stable) |

*Figure: Brain 1BB006: GT, deterministic mean (dark outlier), four independent samples, 12-sample average. All samples ~= -16, decode(z_mu) is the dark outlier.*

---

## RETRACTED ORIGINAL REPORT (sections 1-9, harness-corrupted numbers, kept for record only)

Decisive test of why MAISI brain predictions read too bright in a [-100,100] HU window. All numbers measured from the NV-Generate-CT VAE applied to ground-truth CT (encode->decode, no diffusion, no ControlNet). 21 round-trips across 7 region/center groups.

Claim under test: "Just passing a CT through the VAE encoder/decoder inflates brain HU, and it does not do that for other regions." Original verdict (now retracted): confirmed. The frozen official VAE on perfect GT input lifts brain soft tissue by +93...+105 HU while lowering thorax/pelvis soft tissue by -40...-60 HU; error sign and size track surrounding bone (r=0.88).

### 1. The weights are the genuine, unmodified NVIDIA release

The local VAE (`ckpt/nv-generate-ct/models/autoencoder_v1.pt`) is byte-identical to a fresh download from huggingface.co/nvidia/NV-Generate-CT: SHA-256 local = fresh = 1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3; state_dict 130/130 tensors identical, max |delta-weight| = 0.0, param-hash 51552b589fd51586 (both). No alteration.

### 2. Method (no inference, everything measured)

- Load GT CT through the exact MAISI cached pipeline: ct_range=(-1000,1000) -> [0,1], RAS, pad to mult-of-32. Identical for every region.
- recon = VAE.decode(VAE.encode(ct)), sliding-window (enc ROI 384x352x256, dec ROI 96x88x64, gaussian, overlap 0.4), same as the trainer. bf16 autocast, clamp [0,1]. (NOTE: this oversized enc ROI is the source of the artifact per the corrected finding.)
- Convert both back to HU with x*2000-1000. Input and output share one mapping, so any nonzero recon-GT is purely the VAE.
- Soft tissue = body voxels with -100 < GT < 100 HU and not bone (GT > 300). Distance-from-bone via Euclidean transform on not(GT > 300). HU+body-mask proxy, not anatomical brain segmentation; in a head scan it also includes scalp/muscle.

Fidelity claims: model class & weights from official config_network_rflow.json autoencoder_def (byte-verified); encode/decode via the model's own encode_stage_2_inputs / decode_stage_2_outputs (same as sample.py); normalization identical to official scripts/transforms.py:65 ScaleIntensityRanged(a_min=-1000,a_max=1000,b_min=0,b_max=1,clip=True); HU reverse x*2000-1000 matches sample.py. Only the sliding-window tiling wrapper is ours. Did not run their end-to-end inference script (mask-conditioned generation, not CT reconstruction).

### 3. Result: region-dependent HU bias (retracted)

| region (split,center) | GT soft median | recon soft median | mean error [HU] | soft frac <6vox of bone |
| --- | --- | --- | --- | --- |
| brain (train,A) | 30 | 94 | +77 ± 7 | 0.50 |
| brain (val,B) | 33 | 113 | +93 ± 27 | 0.50 |
| brain (val,C) | 30 | 121 | +105 ± 10 | 0.49 |
| thorax (val,B) | 25 | -33 | -53 ± 11 | 0.31 |
| pelvis (val,C) | -21 | -68 | -49 ± 9 | 0.13 |
| abdomen (val,B) | 0 | -61 | -61 ± 2 | 0.17 |
| headneck (val,C) | 37 | 122 | +94 ± 22 | 0.42 |

Original interpretation: brain inflates on every group including training center A (in-distribution), so not OOD/center effect; thorax/pelvis/abdomen move opposite. (Now attributed to the harness artifact.)

### 4. Mechanism: 8x compression bleeds neighboring HU (retracted)

Original claim: VAE compresses 8x per axis (CORRECTED: actually 4x), cannot preserve a sharp boundary between soft tissue and adjacent extreme-HU structure, bleeds the neighbour's value inward; error depends on what surrounds the soft tissue.

*Figure: brain soft tissue inside thin dense skull (+1000 HU), error largest next to bone (>+100 HU at 0-3 vox) decaying inward; thorax/pelvis surrounded by lung/air/fat (-1000...-100 HU), bleed pulls down. (Retracted: the shift is actually global from GroupNorm, not a near-bone bleed.)*

#### 4a. Bone proximity predicts the error (retracted correlation)

*Figure: across all 21 subjects, soft-tissue HU error correlates with fraction of soft tissue near bone: r=0.88, slope ~432 HU. Brain (~50% near-bone) at top; pelvis (~12%) at bottom.*

#### 4b. VAE transfer function

*Figure: for brain the soft-tissue band is mapped above the identity line; for thorax/pelvis below. Extremes (air, dense bone) near identity; only the narrow mid-band is displaced.*

### 5. Qualitative results (retracted)

All panels: VAE round-trip of GT (no diffusion), soft-tissue window [-100,150] HU; difference maps ±250 HU.

*Figure 5a: gallery across regions. Brain & head-neck reconstructions brighten (positive halo hugging bone); thorax & pelvis darken. Same frozen VAE.*
*Figure 5b: one brain in three planes.*
*Figure 5c: one brain across axial levels.*
*Figure 5d: GT vs VAE-round-trip (no diffusion) vs full MAISI prediction: brightening present after the VAE alone, diffusion not required. (Retracted: brightening is the harness artifact.)*
*Figure 5e: aggregate error maps.*

### 6. Is brain in the VAE's training set? (no)

From MAISI-1 (autoencoder_v1 paper), 4.1, verbatim: "The Volume Compression Network (MAISI VAE) is trained on a dataset comprising 37,243 CT volumes for training and 1,963 CT volumes for validation, covering the chest, abdomen, and head and neck regions. Additionally, we include 17,887 MRI volumes ... spanning the brain, skull-stripped brain, chest, and below-abdomen regions to potentially support MRI modality in future work."

On the CT side (the only modality we use the VAE for), training set is chest / abdomen / head-and-neck. Brain CT is not listed (brain appears only on the MRI side, future work). Pelvis CT ("below-abdomen") also only on the MRI side. So brain CT is at best OOD.

Original argument: OOD does not explain the inflation, geometry does. Residual = measured error minus line err = 432*nearbone - 126 (r=0.88):

| region | near-bone frac | residual vs line [HU] |
| --- | --- | --- |
| brain (train,A) | 0.50 | -11 |
| brain (val,B) | 0.50 | +4 |
| brain (val,C) | 0.49 | +20 |
| thorax (val,B) | 0.31 | -60 |
| pelvis (val,C) | 0.13 | +20 |
| abdomen (val,B) | 0.17 | -11 |
| headneck (val,C) | 0.42 | +38 |

Original interpretation: brain sits on the line (avg residual ~+4 HU), in-distribution head-neck deviates most (+38); brain inflation explained by skull proximity, not VAE-training absence. (Whole geometric explanation retracted.)

### 7. A note on MAISI's own HU quality check

MAISI filters generations whose organ median HU falls outside a per-organ band. Brain band (image_median_statistics_ct.json): min_median=-1000, max_median=238, p99.5=126, 6-sigma-high=370. Original round-trip brain medians (80-159 HU) fall below 238, would pass the filter. The official brain check is too loose to catch the bias; we do not run that filter regardless.

### 8. Per-subject data (retracted numbers)

| region | subj | GT soft med | recon soft med | soft err | air err | bone err | near-bone frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| brain (train,A) | 1BA001 | 33 | 103 | +82 | +23 | +81 | 0.50 |
| brain (train,A) | 1BA005 | 30 | 84 | +68 | +22 | +69 | 0.50 |
| brain (train,A) | 1BA012 | 28 | 95 | +81 | +30 | +76 | 0.50 |
| brain (val,B) | 1BB005 | 34 | 80 | +62 | +40 | +67 | 0.49 |
| brain (val,B) | 1BB006 | 35 | 147 | +127 | +70 | +90 | 0.54 |
| brain (val,B) | 1BB007 | 31 | 111 | +91 | +53 | +77 | 0.47 |
| brain (val,C) | 1BC001 | 31 | 127 | +108 | +14 | +99 | 0.49 |
| brain (val,C) | 1BC006 | 29 | 105 | +91 | +5 | +91 | 0.46 |
| brain (val,C) | 1BC008 | 30 | 132 | +115 | +14 | +71 | 0.50 |
| thorax (val,B) | 1THB006 | 31 | -40 | -64 | +11 | -51 | 0.35 |
| thorax (val,B) | 1THB008 | 17 | -41 | -56 | -11 | -24 | 0.25 |
| thorax (val,B) | 1THB011 | 26 | -18 | -39 | -16 | -62 | 0.32 |
| pelvis (val,C) | 1PC001 | -57 | -116 | -61 | +31 | +33 | 0.10 |
| pelvis (val,C) | 1PC004 | 3 | -37 | -38 | +15 | +39 | 0.16 |
| pelvis (val,C) | 1PC007 | -9 | -50 | -46 | +21 | +34 | 0.13 |
| abdomen (val,B) | 1ABB002 | -26 | -79 | -59 | +6 | +39 | 0.10 |
| abdomen (val,B) | 1ABB010 | 19 | -46 | -63 | +9 | -40 | 0.27 |
| abdomen (val,B) | 1ABB020 | 8 | -57 | -63 | +11 | +6 | 0.15 |
| headneck (val,C) | 1HNC007 | 33 | 93 | +74 | +59 | +53 | 0.45 |
| headneck (val,C) | 1HNC008 | 37 | 116 | +82 | +9 | +91 | 0.32 |
| headneck (val,C) | 1HNC012 | 42 | 159 | +124 | +29 | +120 | 0.49 |

### 9. Conclusion (retracted)

Original conclusion: the official unmodified NV-Generate-CT VAE, by itself on GT CT before any diffusion, lifts brain soft tissue by ~+65 HU (median 30->~100) because its 8x compression bleeds the surrounding skull's +1000 HU into the cortex; effect is geometric (soft-tissue error tracks bone proximity r=0.88), pulling brain/head-neck up and air/fat-surrounded thorax/pelvis/abdomen down; brain worst because ~half its soft tissue is within 6 voxels of skull. Claimed a representational limit of the foundation VAE, not a bug in training/conversion/visualization, not a center/MR-sequence effect. (Superseded by the corrected finding: this is a harness sliding-window/padding artifact, the VAE mildly deflates brain.)

## Reproducibility / Scripts

- `~/vae_audit/run.py` + `build_report.py`: original (corrupted) audit.
- `~/vae_audit/control.py`, `~/vae_audit/control2.py`: controls that diagnosed the artifact.
- Corrupting encode in our pipeline: `trainer.py:451` `_encode_sliding_window`.
- Fix reference: official `dynamic_infer` at `utils.py:647`; apply in `_encode_sliding_window` and `encode_all_volumes.py`, then re-encode + retrain.
- VAE weights: `ckpt/nv-generate-ct/models/autoencoder_v1.pt`, verified against nvidia/NV-Generate-CT (SHA-256 1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3).
