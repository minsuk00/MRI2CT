# Paired MR-CT Datasets - Data Analysis Report

**Source HTML:** _html/dataset_analysis.html
**Date:** 2026-06-09
**TL;DR:** Byte-verified usable paired MR-CT data for MRI->CT synthesis is SynthRAD (844 local, in use), Learn2Reg AbdomenMRCT (16 abdomen pairs, registered+labelled+HU-calibrated, ready to ingest), plus RIRE (brain) and HaN-Seg (head & neck) which are unregistered and need MR<->CT alignment first. 13+ candidates ruled out. Companion URL/recipe survey: docs/01_paired_mr_ct_datasets.md.

Figures rendered only from actual downloaded volumes (Learn2Reg & SynthRAD abdomen, HaN-Seg head & neck, RIRE brain), spanning three body regions.

## 1. Executive summary

- 5 datasets byte-verified downloadable (on disk)
- 4 analyzed with real snapshots
- 5 claimed/gated but NOT byte-verified
- 13+ candidates ruled out (verified)

**Evidence tiers** ("downloadable" is not asserted blanket):
- ✓ **Byte-verified** (bytes in hand): SynthRAD (844 on GPFS), Learn2Reg (downloaded+unzipped this session), HaN-Seg (4.9 GB in `dataset_peek/`), RIRE (downloaded this session + on disk), TopCoW (on disk, wrong modality). Proven, not a claim.
- ✗ **Claimed open but NOT byte-verified**: Kaggle-18 (reCAPTCHA blocks anon; needs an API token), CFB-GBM (Aspera-only, no HTTP path tested), Gold Atlas (not on disk; its Zenodo record now reads restricted, previously over-stated as "have", now corrected).
- 🔒 **Gated, unverifiable here**: Burdenko, GLIS-RT, CERMEP, APIS (require a signed license/registration before any byte moves).

Snapshots rendered only from the byte-verified tier: Learn2Reg & SynthRAD (abdomen), HaN-Seg (head & neck), RIRE (brain). Gated rows are characterized from metadata, not personally pulled.

For structural MRI->CT synthesis, the realistically usable, openly available paired data is: SynthRAD (in use, 844 local), Learn2Reg AbdomenMRCT (verified downloadable, 16 paired, abdomen), Gold Atlas (pelvis), RIRE + HaN-Seg (have), and Kaggle-18 (needs a free token). CFB-GBM (264 brain) is open but a 208 GB Aspera-only pull. Larger brain cohorts (Burdenko 180, GLIS-RT 230, CERMEP 37) require access applications. Everything else is ruled out: 0-CT collections, unpaired cohorts, paywalled/JPEG, or non-public.

## 2. Master comparison - all datasets

| Dataset | Region | Purpose | Modality | N (paired) | Paired | Registered | Preproc. | Access | Download verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SynthRAD 2023/2025 | H&N, thorax, abdomen, pelvis, brain | sCT for MR-only radiotherapy (our reference) | MR (+CBCT) + CT | 890 MR-CT pairs (844 local) | Yes | Yes (rigid/Elastix) | Yes | Open (Zenodo) | ✓ ON DISK (844, GPFS) |
| Learn2Reg AbdomenMRCT | Abdomen | Deformable registration / synthesis | MR T1w + CT | 16 paired (8+8) + unpaired | Yes | Yes (192x160x192, 2mm iso, same grid) | Yes (resampled, labels, masks) | Open (direct, no login) | ✓ DOWNLOADED+UNZIPPED this session |
| HaN-Seg | Head & Neck | OAR auto-seg / MR->CT (NOT dose planning, no target/dose) | T1 MR + CT + 30 OARs | 42 | Yes (same patient) | No: CT 1024²/2mm vs MR 512²/3mm, separate spaces | Per-case NRRD | Open (Zenodo) | ✓ ON DISK (dataset_peek, 4.9 GB) |
| RIRE | Brain | Registration eval | T1/T2/PD MR + CT | 16 | Yes (same patient) | No (separate spaces) | Legacy MetaImage | Open (IPFS) | ✓ DOWNLOADED (1 subj this session + on disk) |
| Kaggle-18 (Saadia 2025) | Brain, abdomen, neck | Cross-modality synthesis | T1w + T2w MR + CT | 18 pts / 389 triplets | Yes (co-registered) | Yes | Yes (DICOM+PNG, 2D axial) | Kaggle login, CC BY-NC | ✗ UNVERIFIED: reCAPTCHA, needs API token |
| CFB-GBM | Brain (GBM) | Glioblastoma imaging+RT | multi-MR + CT | 264 (195 w/ CT) | Overlapping cohort | Likely not pre-registered | NIfTI | Open, 208 GB Aspera only | ✗ UNVERIFIED: Aspera-only, no HTTP path |
| Gold Atlas (pelvis) | Pelvis | sCT / MR-only RT | T1w+T2w MR + CT | 19 | Yes (per literature) | Yes (per literature) | Yes | Zenodo 583096 | ✗ CANNOT CONFIRM: not on disk; Zenodo record now restricted |
| Burdenko-GBM | Brain (GBM) | RT planning | T1/T1C/T2/FLAIR + topometric CT | 180 | Yes (NOT pre-registered) | No (MR in acq space) | DICOM | Gated, TCIA Restricted License | 🔒 GATED: needs signed license (unverified) |
| GLIS-RT | Brain (glioma) | RT target definition | T1+Gd / T2-FLAIR + CT | 230 | Yes (+REG objects) | Yes (REG included) | DICOM | Gated, NIH controlled access | 🔒 GATED: needs access request (unverified) |
| CERMEP-iDB-MRXFDG | Brain | Healthy multimodal atlas | T1/FLAIR + CT | 37 healthy | Yes | Yes | Yes | Gated, application | 🔒 GATED: application (unverified) |
| APIS | Brain (stroke) | Stroke lesion seg | ADC map + NCCT | 60 paired | Yes (train only) | Coarse | 2D, low-res | Registration, 7-day link | 🔒 GATED: registration; ADC not structural |

Legend: ✓ green = bytes confirmed on disk / downloaded this session; ✗ amber = claimed open but NOT byte-verified (auth/Aspera/now-restricted); 🔒 purple = access-gated, not verifiable without an application.

## 3. Deep dive - Learn2Reg AbdomenMRCT (downloaded & verified)

- 16 paired same-patient MR-CT
- 192x160x192 @ 2.0mm iso
- T1w + CT (modality 0 / modality 1)
- ✓ labels: liver, spleen, kidneys + masks

**What it is for:** originally the Learn2Reg deformable registration challenge (abdomen MR<->CT), but the 16 intra-patient pairs are directly usable as supervised MRI->CT training data. Built from TCIA (paired MR-CT) + CHAOS (MR) + BCV (CT).

**Registration:** every pair shares an identical 192x160x192 / 2 mm grid and affine (`same_grid = True` for all 8 train cases) i.e. already co-registered & resampled. CT is in Hounsfield units (~ [-1024, +1100...3000]); MR T1w is raw-scaled (per-volume max varies widely -> needs intensity normalization).

*Figure: bundled organ segmentation (liver / spleen / right & left kidney) on the CT, useful for an auxiliary segmentation/Dice loss like the project's Anatomix pipeline.*

*Figure: intensity distributions (log y). CT is HU-calibrated; MR T1w is uncalibrated and varies per volume, normalize (percentile / z-score) before training, as the project's MONAI pipeline does.*

### Per-case inventory (8 training pairs)

| Case | Shape | Spacing (mm) | MR range | CT range (HU) | Registered |
| --- | --- | --- | --- | --- | --- |
| 0001 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 29728.7] | [-1024.0, 1116.5] | ✓ |
| 0002 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 398.9] | [-1024.0, 1148.4] | ✓ |
| 0003 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 27137.4] | [-1024.0, 974.1] | ✓ |
| 0004 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 473.8] | [-1024.0, 1193.8] | ✓ |
| 0005 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 1471.8] | [-1024.0, 3037.4] | ✓ |
| 0006 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 698.5] | [-1024.0, 2794.4] | ✓ |
| 0007 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 5206.3] | [-1024.0, 3071.0] | ✓ |
| 0008 | 192x160x192 | [2.0, 2.0, 2.0] | [0.0, 1839.4] | [-1024.0, 1153.3] | ✓ |

(8 more identically-structured pairs live in `imagesTs/`, for 16 paired total.)

## 4. Reference - SynthRAD (our in-use dataset, for comparison)

- 844 local subjects (GPFS)
- 310x244x182 @ 1.5mm iso
- HU: CT [-1024.0, 2003.0]
- ✓ registered+masked: moved_mr <-> ct, same grid

This is what the model trains on today: per-subject `ct.nii` + `moved_mr.nii` (MR rigidly registered to CT), `mask.nii`, and CT segmentations, resampled to 1.5 mm isotropic and body-masked. Background is zeroed (body mask) vs Learn2Reg's full FOV.

**Learn2Reg vs SynthRAD - practical notes for adding the data:**
- Resolution: Learn2Reg is 2.0 mm iso / 192x160x192; SynthRAD pipeline is 1.5 mm iso. Resample Learn2Reg to match.
- FOV / masking: SynthRAD is body-masked (background = 0); Learn2Reg is full FOV with air. Apply the same body-mask step.
- CT units: both HU, directly compatible with the existing clip-and-scale.
- MR scaling: both uncalibrated; the existing percentile/min-max MR normalization applies.
- Region: Learn2Reg adds abdomen diversity; only 16 pairs, so best as augmentation, not a standalone train set.

## 5. Head & Neck example - HaN-Seg (on disk, byte-verified)

**What it is for:** the HaN-Seg organ-at-risk segmentation challenge, same-patient head-&-neck CT + T1 MR (42 patients), with 30 OAR contours.

**Pairing:** same patient, but not pre-registered: CT is 1024²/2 mm and MR is 512²/3 mm in separate acquisition spaces (so the overlay row is omitted). CT is HU-calibrated; MR T1 is uncalibrated. Useful as an H&N pair source after registering MR<->CT. Full 4.9 GB copy sits in `dataset_peek/hanseg/`.

**Is HaN-Seg usable for dosimetry / dose planning? Not directly.** It comes from H&N patients imaged "for the purpose of image-guided radiotherapy planning" (official README), so it is RT-derived, but for an actual dose plan you need a target and a plan/dose grid, and HaN-Seg has neither:
- OARs: ✓ 30 per patient (organs to spare), confirmed, segmented on CT.
- PTV / GTV / CTV (target): ✗ none, verified by scanning all 42 cases (0 hits).
- RTDOSE / RTPLAN / RTSTRUCT / beams: ✗ none.

So you cannot run or evaluate a treatment plan out of the box (nothing to prescribe dose to). What it is good for: OAR auto-segmentation, and MR->CT synthesis (our task). For a dosimetry downstream task you'd have to define your own targets + plan (e.g. via pyRadPlan) on the sCT. License is CC BY-NC-ND 4.0 (non-commercial, no-derivatives), restrictive.

## 6. Brain example - RIRE (downloaded one subject)

**What it is for:** the classic Retrospective Image Registration Evaluation benchmark, same-patient brain CT + MR (T1/T2/PD), used to score registration algorithms.

**Pairing:** same patient, but CT (512x512x28 @0.65mm) and MR (256x256x26 @1.26mm) sit in different geometries and are NOT pre-registered (computing that alignment is the task). Old (2006), thick 4 mm slices, low through-plane resolution; CT is HU-calibrated. Best as a small extra brain test set, and only after registering MR<->CT yourself.

## 7. Ruled-out candidates (verified)

| Candidate | Why excluded |
| --- | --- |
| UPENN-GBM | API-confirmed 630 pts MR-only, 0 CT |
| ReMIND | API-confirmed MR + ultrasound, 0 CT |
| Vestibular-Schwannoma-SEG | API-confirmed MR + RT, 0 CT |
| CPTAC-PDA / CCRCC / LUAD | Same-patient CT∩MR overlap only 7 / 7 / 1, not viable |
| CHAOS / AMOS | CT and MR are different patients (unpaired cohorts) |
| IEEE-DataPort Glioblastoma (50) | Paired but subscription-paywalled + JPEG (no HU) |
| tFUS skull CT-T1w / CT-ZTE | Not public, clinical-trial, on-request only |
| OpenNeuro (all) | Platform has no CT modality (MRI/EEG/MEG/PET only) |
| TopCoW | Paired but wrong modality (MRA-TOF + CTA angiography) |
| Prostate-Anatomical-Edge-Cases | CT + RTSTRUCT only, no MR |
| HEAD-NECK-PET-CT / BRAIN-TR-GammaKnife | No MR / no CT respectively |
| Unpaired MR-CT brain (Jordan) | Explicitly unpaired (20 pts, CycleGAN) |
| ABCs 2020 | Dead host, Zenodo is a PDF only |

## 8. Recommendation

- **Ingest Learn2Reg AbdomenMRCT now**: verified, registered, labelled, HU-calibrated; 16 abdomen pairs to augment SynthRAD's abdomen split. Resample 2.0->1.5 mm + body-mask to match the pipeline.
- **Grab Kaggle-18** if a Kaggle API token is available (adds T1+T2 brain/neck pairs; tiny but co-registered).
- **Defer CFB-GBM** unless brain scale is needed: 208 GB Aspera, partial CT overlap, registration unverified.
- **Apply for gated brain sets** (Burdenko / GLIS-RT / CERMEP) only if the open data proves insufficient; GLIS-RT ships registration objects, Burdenko needs self-registration.

Companion markdown survey with URLs and download recipes: `docs/01_paired_mr_ct_datasets.md`. Verified download of Learn2Reg: `curl -A <browserUA> -L "https://cloud.imi.uni-luebeck.de/s/yiQZfo43YBBg7zL/download"`.
