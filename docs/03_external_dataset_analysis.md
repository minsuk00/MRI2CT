# External MR-CT Datasets: Technical Analysis

**Source HTML:** _html/external_dataset_analysis.html
**Date:** 2026-06-16
**TL;DR:** Of 6 external paired MR-CT sets on disk, CFB-GBM (195, brain, rigid) and Gold Atlas (19, pelvis, deformable) are training-ready (co-registered, same-grid, HU-calibrated; just resample to 1.5 mm). HaN-Seg, RIRE, and Learn2Reg are same-patient but not deformably aligned (need our registration step). TopCoW is angiography (OOD only). CFB-GBM is the only set with real GTV+RTDOSE for true RT dose-eval; no set supports PET-AC.

6 datasets verified on disk + 3 pending (literature). Every claim is tagged **data-verified** (measured from the on-disk volume) or **source-cited** (paper/Zenodo/TCIA). Per-dataset spec tables show one named example subject; exact shape/spacing/FOV can vary across subjects. Registration status was checked in world space (resample MR->CT by affine, measure anatomical overlap), not merely by grid-mismatch (different grid does not equal unregistered).

Figure windowing: CT uses a region-consistent window: head/brain & H&N [-100,100] HU (soft tissue), abdomen/pelvis [-1024,1024] HU; MR uses a per-volume 1-99th-percentile window.

Related: docs/01_paired_mr_ct_datasets.md, docs/02_dataset_analysis.md.

## Master comparison

| Dataset | Region | N paired (same-patient) | MR | CT | MR<->CT registered? | Same grid/FOV? | HU-calibrated? | Format | License |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Learn2Reg AbdomenMRCT | Abdomen | 16 (8 Tr + 8 Ts) | T1w | yes | co-gridded, NOT deformably aligned | yes (192x160x192 @2mm) | yes | NIfTI | TCIA/CHAOS/BCV |
| HaN-Seg | Head&Neck | 42 | T1 | yes | no | no (1024² vs 512², different) | yes | NRRD | CC BY-NC-ND 4.0 |
| RIRE | Brain | 16 (w/ CT) | T1/T2/PD/MP-RAGE | yes | no (alignment is the task) | no | yes | MetaImage | CC BY 3.0 |
| TopCoW | Brain (vessels) | 125 (+5 val) | MRA-TOF | CTA | no | no | yes (CTA) | NIfTI | OpenData.swiss |
| Gold Atlas | Pelvis | 19 | T1w + T2w | yes | yes - DEFORMABLE (B-spline DIR) | yes (deformed CT <-> T2) | yes | DICOM | Zenodo restricted; cite req. |
| CFB-GBM | Brain (GBM) | 195 | T1Gd (+6 more) | yes | yes - RIGID (to T1Gd) | yes (512²x208 @0.5/1mm) | yes | NIfTI | CC BY 4.0 |

"N paired" = same-patient subjects carrying both a structural MR and a CT, counted from disk. Grid/FOV/HU are data-verified; `registered?` is from a world-space overlap test (affine resample), and the registration method (rigid vs deformable) is source-cited.

Cross-cohort grid verification (the same-grid / separate-space classification was confirmed on every paired subject, not just the example shown, by comparing MR vs CT image geometry: size + spacing + origin + orientation):

| Dataset | pairs checked | result |
| --- | --- | --- |
| CFB-GBM | 195 | 195/195 same grid (CT == T1Gd) |
| Gold Atlas | 19 | 19/19 deformed-CT grid == an MR-T2 series |
| Learn2Reg | 16 | 16/16 share the 192³ grid (co-gridded) |
| HaN-Seg | 42 | 42/42 different grids (separate spaces) |
| RIRE | 16 | 16/16 different grids (separate spaces) |
| TopCoW | 125 | 125/125 different grids (separate spaces) |

## 1. Learn2Reg AbdomenMRCT (abdomen, 16 paired, NIfTI)

### Structure / navigation

Layout: `AbdomenMRCT/imagesTr/` and `imagesTs/` with nnU-Net naming `AbdomenMRCT_<case>_0000.nii.gz` = MR T1w, `_0001.nii.gz` = CT (modality map in `AbdomenMRCT_dataset.json`). Also `labelsTr/` (organ masks) + `masksTr/`.

Gotcha (data-verified): `imagesTr/` holds 97 case-IDs but only 8 are paired (have both `_0000` + `_0001`); the rest are unpaired CT-only (BCV) / MR-only (CHAOS) augmentation scans. Paired same-patient MR-CT = 8 (Tr) + 8 (Ts) = 16 (`numPairedTraining/Test` in dataset.json; verified by counting files).

### Verified specs (case 0001)

| volume | shape (x,y,z) | spacing mm | FOV mm | intensity [min,max] | [p1,p99] |
| --- | --- | --- | --- | --- | --- |
| MR T1w (_0000) | [192, 160, 192] | [2.0, 2.0, 2.0] | [384.0, 320.0, 384.0] | [0.0, 29728.7] | [0.0, 15278.6] |
| CT (_0001) | [192, 160, 192] | [2.0, 2.0, 2.0] | [384.0, 320.0, 384.0] | [-1024.0, 1116.5] | [-1024.0, 342.7] |

### Registration (data-verified, source-cited)

MR and CT share an identical [192, 160, 192] / 2 mm grid and affine (grid_match = True, verified). But `dataset.json` declares `registration_direction: fixed=MR, moving=CT`: this is a registration challenge, the pair is placed in a common frame but the deformable organ-level alignment is the unsolved task, not a property of the data. So MR<->CT voxel correspondence is only coarse/affine, not exact. Cite: Hering et al., Learn2Reg, IEEE TMI 2023 (arXiv:2112.04489) + dataset.json.

Ingest: already 1-grid & HU-calibrated, but not deformably registered, so using as supervised paired MR->CT data requires first solving the abdominal registration (our `register_single_subject.py` step). Resample 2->1.5 mm. Best treated as a small abdomen augmentation, not gold-standard pairs.

## 2. HaN-Seg (head & neck, 42 paired, NRRD)

### Structure / navigation

Per-case folder `set_1/case_XX/` containing `case_XX_IMG_CT.nrrd`, `case_XX_IMG_MR_T1.nrrd`, and ~30 `case_XX_OAR_<organ>.seg.nrrd` binary masks (organs-at-risk, drawn on CT).

Note (data-verified): 42 same-patient CT+T1 pairs. The OAR files are usable segmentation labels (29-30 organ-at-risk masks/case, drawn on CT), but there is no target structure (PTV/GTV/CTV) and no dose/plan (verified: only structure files are `IMG_*` and `OAR_*`). Great for organ-seg, not for a dose plan (which needs a target).

### Verified specs (case_01)

| volume | shape (x,y,z) | spacing mm | FOV mm | intensity [min,max] | [p1,p99] |
| --- | --- | --- | --- | --- | --- |
| MR T1 | [512, 512, 83] | [0.703, 0.703, 3.0] | [360.0, 360.0, 249.0] | [32768.0, 35447.0] | [32768.0, 33534.0] |
| CT | [1024, 1024, 202] | [0.558, 0.558, 2.0] | [571.0, 571.0, 404.0] | [-1000.0, 3000.0] | [-1000.0, 287.0] |

### Registration (data-verified, source-cited)

NOT co-registered, tested in world space (not just by grid): resampling the MR into the CT's coordinate frame (affine-aware) yields 0 anatomical overlap (the MR falls outside the CT's physical extent), they occupy disjoint world frames. (They also differ in grid: CT [1024, 1024, 202]@[0.558, 0.558, 2.0] mm vs MR [512, 512, 83]@[0.703, 0.703, 3.0] mm, but grid-mismatch alone wouldn't prove non-registration; the world-frame test does.) Cite: Podobnik et al., "HaN-Seg", Medical Physics 2023, 50(3):1917-1927; Zenodo 7442914.

Geometry varies widely (data-verified): across 42 cases there are 35 distinct CT shapes / 36 distinct spacings and 26 MR shapes; only the in-plane matrix is fixed (CT 1024², MR 512²). CT FOV (500-800 mm in-plane) > MR FOV (240-420 mm) in all 42. The case_01 example is not fully representative (its MR is one of 6/42 stored as 16-bit-unsigned with a +32768 offset, and its 0.703 mm MR spacing occurs in only 1 case).

Ingest: needs full MR->CT registration + resample + body-mask before use. CT HU up to 3000 (dental/bone) exceeds our [-1024,1024] clip.

## 3. RIRE (brain, 16 with CT, MetaImage legacy)

### Structure / navigation

Double-nested `patient_XXX/patient_XXX/` with one tar.gz per modality. RIRE is multi-modal: `ct` + structural MR `mr_T1`, `mr_T2`, `mr_PD` (and `mr_MP-RAGE` for patients 101-109) + `pet`, plus `mr_*_rectified` variants. Each unpacks to MetaImage (`.mhd` + `image.bin.Z`, LZW-compressed). The spec table / figure show T1 only as a representative MR.

"rectified" = geometric-distortion-corrected MR (static-field + scale distortion, the Vanderbilt/Chang-Fitzpatrick "rectification"; provided for patients 001-007), not intensity-corrected. Data-confirmed as a spatial resampling (spacing 1.25->1.265 / 4.0->4.129 mm, voxels resampled). RIRE's own pages don't define the term; this is from the associated distortion-correction literature.

Now extracted to `RIRE/nifti/<patient>/<modality>.nii.gz` (18 patients, 100 volumes; tarballs kept).

Gotchas (data-verified): (1) inner tars are not auto-extracted; (2) `image.bin.Z` needs `gzip -d` (Unix compress/LZW); (3) the `ct/` tar ships a stray second `.mhd` (patient_002) pointing at the same bin; (4) 16 of 18 patients have CT (patient_008/009 are MR+PET only); (5) 7 patients have PET (001/002/005/006/007/008/009) and 9 have MP-RAGE (101-109).

### Verified specs (patient_001)

| volume | shape (x,y,z) | spacing mm | FOV mm | intensity [min,max] | [p1,p99] |
| --- | --- | --- | --- | --- | --- |
| MR T1 | [256, 256, 26] | [1.25, 1.25, 4.0] | [320.0, 320.0, 104.0] | [2.0, 2053.0] | [4.0, 1130.0] |
| CT | [512, 512, 28] | [0.654, 0.654, 4.0] | [334.6, 334.6, 112.0] | [-1024.0, 2074.0] | [-1024.0, 1183.0] |

### Registration (data-verified, source-cited)

NOT registered, by design. RIRE is the Retrospective Image Registration Evaluation benchmark; the protocol tasks investigators with finding the CT->MR transform (gold standard came from bone fiducials, later erased), so the images are deliberately not pre-aligned. World-frame test agrees: affine-only overlap ~0.44 (incidental head-centering, same level as the unregistered baseline). Cite: West, Fitzpatrick et al., J Comput Assist Tomogr 1997, 21(4):554-566 (PMID 9216759).

Geometry, two groups (data-verified): patients 001-009 = CT 512²@0.65 mm / MR 256²@1.25 mm, 4 mm slices; patients 101-109 = CT 512²@~0.42 mm / MR 256²@~0.82 mm, 3 mm slices. Only CT 512²/MR 256²/thick-slice/different-grid are universal. The patient_001 example is the 001-009 group; "~1990s" is dated from the 1997 paper era (the files carry no acquisition date).

Ingest: small/old; only a tiny extra brain test set after registration.

PET note (data-verified): 7 patients have PET, but only 5 have CT+PET+MR together (001/002/005/006/007). The PET is a reconstructed image (e.g. 128²x15), no raw sinogram/listmode, so attenuation correction can't be re-run with a pseudo-CT mu-map. Not usable for the PET-AC downstream (same blocker as OASIS-3).

## 4. TopCoW (brain vessels, 125 paired, NIfTI)

### Structure / navigation

Flat `imagesTr/` with `topcow_ct_NNN_0000.nii.gz` + `topcow_mr_NNN_0000.nii.gz` (same NNN = same patient), `imagesVal/` (5 pairs), plus label folders `cow_seg_labelsTr/`, `roi_loc_labelsTr/`, `antpos_edges_labelsTr/` (Circle-of-Willis vessel annotations).

Modality caveat (data-verified, source-cited): this is angiography, MR is MRA-TOF and CT is CTA, not structural MR / soft-tissue CT. Off-distribution for HU sCT; useful only as an OOD / vessel experiment. Cite: Yang et al., TopCoW (arXiv:2312.17670); release README.

### Verified specs (case 001)

| volume | shape (x,y,z) | spacing mm | FOV mm | intensity [min,max] | [p1,p99] |
| --- | --- | --- | --- | --- | --- |
| MRA-TOF | [508, 585, 189] | [0.297, 0.297, 0.6] | [150.8, 173.7, 113.4] | [0.0, 1331.0] | [0.0, 253.0] |
| CTA | [284, 327, 243] | [0.508, 0.508, 0.625] | [144.2, 166.1, 151.9] | [-3024.0, 3071.0] | [-1016.0, 1425.0] |

### Registration (data-verified, source-cited)

Pairing is documentation-confirmed: the README calls them "125 pairs" and names files `topcow_{modality}_{pat_id}` (same pat_id = same patient). Non-registration is data-derived: the README says nothing about cross-modality registration, and resampling MR into the CT world frame gives only ~0.45 foreground overlap, the same incidental level as the known-unregistered RIRE benchmark (i.e. no true alignment). Each modality is in its own braincase-cropped native space (CT [284, 327, 243]@[0.508, 0.508, 0.625] mm vs MR [508, 585, 189]@[0.297, 0.297, 0.6] mm). CTA HU spans [-3024, 3071].

Ingest: per-subject registration needed; modality mismatch makes it an OOD probe rather than training data for structural sCT.

## 5. Gold Atlas (Male Pelvis, pelvis, 19 paired, DICOM)

### Structure / navigation

One folder per patient (`1_01_P` ... `3_04_P`; leading digit = registration site 1/2/3) with all DICOM files dumped flat. You must group by `SeriesInstanceUID`; each patient holds 8 series:

| Modality | SeriesDescription | slices (1_01_P) | role |
| --- | --- | --- | --- |
| CT | "RT p+ Buk 3.0 B30f" | 140 | original planning CT (native CT space) |
| CT | "Deformed 'CT 1' using 'CTtoMR DIR site1'" | 84 | deformed CT, resampled onto the MR grid |
| MR | "Ax T2 FRFSE GOLD ATLAS" | 84 | MR T2 (matches deformed-CT grid) |
| MR | "Ax T1 FSE GOLD ATLAS" | 70 | MR T1 (separate grid) |
| REG | - | 1 | DICOM spatial-registration object |
| RTSTRUCT (2-3) | "RS: Unapproved Structure Set" | 1 each | one set carries 5 observers + consensus + STAPLE (verified) |

The "hard" part (data-verified): there are two CT series, the original CT (native CT space) and a "Deformed CT using CTtoMR DIR". For paired training you want the deformed CT (carries the same [512, 512, 84] grid as MR T2). Confirmed across all 19 (each has exactly 2 CT series; deformed-CT grid == an MR-T2 series).

What's registered to what (data-verified, corrected): the deformed CT, MR T2 and MR T1 all share one DICOM `FrameOfReferenceUID` (== co-registered), while the original CT has a different one. So MR T1 is co-registered too, it just sits on a different grid (70 slices @3 mm vs T2's 84 @2.5 mm); different grid does not equal unregistered. World-frame overlap confirms it: T1<->T2 Dice 0.85, deformed-CT<->T2 0.80, but original-CT<->T2 only 0.14 (the original CT is the one not aligned to MR). The REG object is a DICOM spatial-registration object storing the (rigid/affine) transform between series, not an image.

Structures (data-verified): the multi-observer RTSTRUCT carries 5 observers (User1-User5) + consensus + STAPLE delineations of 9 pelvic OARs (prostate, bladder, rectum, seminal vesicles, anal canal, penile bulb, neurovascular bundles, L/R femoral heads) + an "External" body contour, and zero targets (no PTV/GTV/CTV; the prostate is delineated as an organ, not a target). The contours live in the MR/registered frame (align to T1/T2/deformed-CT, not the original CT). Usable as OAR-segmentation ground truth (rasterize the contours to masks first); not a dose plan (no target/dose).

### Verified specs (1_01_P)

| volume | shape (x,y,z) | spacing mm | FOV mm | intensity [min,max] | [p1,p99] |
| --- | --- | --- | --- | --- | --- |
| MR T2 (FRFSE) | [512, 512, 84] | [0.875, 0.875, 2.5] | [448.0, 448.0, 210.0] | [0.0, 10066.0] | [0.0, 2645.0] |
| deformed CT | [512, 512, 84] | [0.875, 0.875, 2.5] | [448.0, 448.0, 210.0] | [-1000.0, 2651.0] | [-1000.0, 365.0] |
| original CT (native) | [512, 512, 140] | [0.977, 0.977, 3.0] | [500.0, 500.0, 420.0] | [-1000.0, 3071.0] | [-1000.0, 230.0] |

### Registration (data-verified, source-cited)

DEFORMABLE (non-rigid). The dataset ships a CT B-spline-deformably registered to the MR; data-verified: the deformed-CT series shares the exact MR-T2 grid ([512, 512, 84], grid_match = True) and the checkerboard shows seamless bone/soft-tissue continuity. The Zenodo record explicitly lists the ELASTIX B-spline transform parameters. Cite: Nyholm et al., Medical Physics 2018; Zenodo 583096.

Ingest: the cleanest pelvis pair we have, deformed CT + MR T2 (or T1) are already aligned & HU-calibrated; only resample 0.875x0.875x2.5 -> 1.5 mm iso + body-mask. Training-ready after resample. Now converted to grouped NIfTI at `GoldAtlas/nifti/<patient>/{ct, ct_deformed, mr_T1, mr_T2}.nii.gz` (raw DICOM kept). (Note T1/T2 sequence-matching caveat vs SynthRAD pelvis, see project memory.)

## 6. CFB-GBM (brain GBM, 195 paired, NIfTI)

### Structure / navigation

Per-patient `<id>/<timepoint>/<id>_<tp>_<modality>.nii.gz`, timepoints `t0` (baseline) / `t1` / `t2`, modalities `ct, t1gd, t1eg, t1tse, t2tse, t2star, flair, adc, rtdose, gtv`. Separate CSV availability tables (`*_ct_availability`, `*_mri_availability`) list per-patient coverage.

Gotchas (data-verified): (1) folder id has leading zeros (`001/`) but the filename does not (`1_t0_ct.nii.gz`); (2) CT, RTDOSE and GTV exist only at t0; (3) MR coverage is sparse/heterogeneous, only T1Gd is present for all 264 (FLAIR 255, T2* 241, T1-EG 215, T2-TSE 125, ADC 129 at t0; T1-TSE absent at t0); (4) CT is defaced. Paired t0 CT+T1Gd = 195.

### Verified specs (patient 1, t0)

| volume | shape (x,y,z) | spacing mm | FOV mm | intensity [min,max] | [p1,p99] |
| --- | --- | --- | --- | --- | --- |
| MR T1Gd | [512, 512, 208] | [0.5, 0.5, 1.0] | [256.0, 256.0, 208.0] | [0.0, 1235.0] | [0.0, 384.0] |
| CT | [512, 512, 208] | [0.5, 0.5, 1.0] | [256.0, 256.0, 208.0] | [-1024.0, 3067.0] | [-1024.0, 971.5] |

### Registration (data-verified, source-cited, correction)

RIGID, and the CT IS co-registered to the MR. Data-verified: CT and T1Gd share an identical [512, 512, 208] @ [0.5, 0.5, 1.0] mm grid + affine (grid_match = True) and the checkerboard shows the skull (CT) wrapping the brain (MR) seamlessly. The CFB-GBM methods state: "All the images were translated and resampled in the same space using rigid registration, the T1Gd (at t0) as reference", i.e. the CT was rigidly resampled into the T1Gd frame. Cite: Moreau et al., CFB-GBM, TCIA DOI 10.7937/v9pn-2f72; GitHub AurelienCD/CFB-GBM.

Ingest: already co-registered & same-grid, training-ready after resample (0.5x0.5x1 -> 1.5 mm iso) + body-mask. CT HU to 3067 (dense bone) exceeds our [-1024,1024] clip, saturates above. Bonus: RTDOSE+GTV at t0 enable a real RT dose-eval downstream.

## Appendix: 3 datasets pending access (literature only, NOT on disk)

Source-cited from publications; none verified from data (we do not yet hold the bytes).

| Dataset | Region · N | Modalities | Why it matters | Access |
| --- | --- | --- | --- | --- |
| CERMEP-IDB-MRXFDG | Brain · 37 healthy | [18F]FDG PET + T1 + FLAIR + CT (same-patient PET/CT+MR) | The only PET-AC candidate, has same-patient PET + real CT + MR; could validate sCT-based attenuation correction (pending confirmation of raw/NAC PET availability). | email DUA (merida@cermep.fr); request sent |
| Burdenko-GBM | Brain · 180 | T1/T1C/T2/FLAIR + topometric CT (+RTSTRUCT/DOSE) | Large brain MR-CT; MR in acquisition space (register yourself). | dbGaP phs004225, needs PI + institutional sign-off |
| GLIS-RT | Brain · 230 | T1+Gd / T2-FLAIR + CT + REG objects | Ships registration objects (best-aligned gated brain set). | dbGaP phs004225 (same request as Burdenko) |

Cite: Mérida et al., EJNMMI Research 2021 (DOI 10.1186/s13550-021-00830-6); TCIA Burdenko-GBM-Progression; TCIA GLIS-RT (DOI 10.7937/TCIA.T905-ZQ20).

## Downstream-task suitability (summary)

How each on-disk dataset maps to our pipeline's capabilities. "Train" = usable as supervised paired MR->CT data; RT-dose = pyRadPlan dose-eval (`pyradplan/eval_real_sct.py` optimizes a plan on GT-CT, delivers the same plan on the model sCT, and compares MAE dose/DVH/gamma 2%/2mm; `sct_dose_eval.py` is a proxy-sCT variant with no gamma); Seg = TotalSegmentator organ-Dice (`src/evaluate/total_seg.py`); OOD-val = held-out qualitative/metric test; PET-AC = synthetic-CT attenuation correction.

Note: the current dose-eval code plants a synthetic spherical PTV (TotalSeg has no tumor), so "RT-dose yes" today means body/OAR dose fidelity; a real-target plan needs a dataset that ships tumor structures.

| Dataset | Train (paired) | RT dose-eval | Organ-seg | OOD validation | PET-AC |
| --- | --- | --- | --- | --- | --- |
| Learn2Reg Abd | △ after registration | △ synthetic target | yes | yes abdomen | no PET |
| HaN-Seg | △ after registration | no target/dose | yes (+OARs) | yes H&N | no PET |
| RIRE | △ after registration (small/old) | △ synthetic | yes | yes brain (tiny) | PET old/recon-only |
| TopCoW | angiography | no | △ vessels | yes OOD modality | no PET |
| Gold Atlas | yes ready (resample) | △ OARs, synthetic target | yes | yes pelvis | no PET |
| CFB-GBM | yes ready (resample) | yes real GTV+RTDOSE | yes | yes brain | no PET |

Headline findings:
- Training-ready now (already co-registered + HU CT): CFB-GBM (195, brain, rigid) and Gold Atlas (19, pelvis, deformable), just resample to 1.5 mm + body-mask.
- Need our registration step first: HaN-Seg, RIRE, and Learn2Reg (co-gridded but not deformably aligned). TopCoW also needs it but is off-modality.
- RT dose-eval: CFB-GBM is the standout, the only on-disk set carrying real tumor targets (GTV, 191 pts) + recorded RTDOSE (194 pts), so it could replace the pipeline's synthetic spherical PTV with a real clinical target. Gold Atlas (5-observer OARs, verified) and HaN-Seg (30 OARs) have organs but no targets and no dose (verified: no RTDOSE/RTPLAN).
- PET-AC: no for all 6. None contains usable PET (RIRE's is old reconstructed registration-benchmark PET), and our repo has no PET-AC code (verified). PET-AC needs CERMEP (pending) + raw/NAC PET.

## Pipeline-fit reference

Our trainer (`src/common/data.py`) consumes pre-processed subjects laid out as `<subject>/ct.nii` + `moved_mr.nii` (MR already registered into CT space) + `mask.nii`, then: enforces RAS, clips CT to [-1024, 1024] HU -> [0,1], per-volume min-max MR, and assumes data is resampled to 1.5 mm isotropic. There is no runtime registration; alignment must be done upfront (`src/preprocess/register_single_subject.py` -> `resample_and_split_dataset.py`). So for each external set the gap to ingestion is:

| Dataset | Register MR<->CT? | Resample to 1.5mm? | Body-mask? | Cohort CT-max HU (>1024 is clipped) |
| --- | --- | --- | --- | --- |
| Learn2Reg | yes (deformable, unsolved) | 2->1.5mm | yes | 15/16 paired CTs >1024; up to 7701 (metal) |
| HaN-Seg | yes | yes | yes | up to 3000 (dental/bone) |
| RIRE | yes | yes | yes | all 16 >1024; up to 2617 (skull) |
| TopCoW | yes | yes | yes | CTA up to 3071 |
| Gold Atlas | no (deformed CT supplied) | 0.875->1.5mm | yes | native CT up to 19002 (hip implants; all 19 >2798) |
| CFB-GBM | no (rigid, supplied) | 0.5/1->1.5mm | yes (CT defaced) | up to 3071 (dense bone) |

CT-max values are cohort-verified (max over every paired subject; GoldAtlas/RIRE loaded in full). Every dataset contains real CT with bone/metal >1024 HU, so our [-1024,1024] clip saturates the densest material in all of them; magnitude varies as shown. (CFB-GBM defacing is per the dataset's stated CTA-DEFACE step, source-cited, not independently re-verified here.)

## Sources

- Learn2Reg: Hering et al., "Learn2Reg...", IEEE TMI 2023; 42(3):697-712; arXiv:2112.04489; challenge; on-disk AbdomenMRCT_dataset.json.
- HaN-Seg: Podobnik et al., "HaN-Seg...", Medical Physics 2023; 50(3):1917-1927, DOI 10.1002/mp.16197; Zenodo 7442914.
- RIRE: West, Fitzpatrick et al., J Comput Assist Tomogr 1997; 21(4):554-566, PMID 9216759; RIRE / Vanderbilt.
- TopCoW: Yang, Musio et al., "Benchmarking the CoW with the TopCoW Challenge", arXiv:2312.17670; Zenodo 15692630.
- Gold Atlas: Nyholm et al., Medical Physics 2018; 45(3):1295-1300, DOI 10.1002/mp.12748; Zenodo 583096 (ELASTIX B-spline params in record).
- CFB-GBM: Moreau et al., CFB-GBM, TCIA DOI 10.7937/v9pn-2f72; GitHub AurelienCD/CFB-GBM.
- CERMEP: Mérida et al., EJNMMI Research 2021, DOI 10.1186/s13550-021-00830-6.
- Burdenko / GLIS-RT: TCIA collections (NIH controlled-access, dbGaP phs004225); GLIS-RT DOI 10.7937/TCIA.T905-ZQ20.
- Our pipeline / downstream: src/common/data.py (req. pre-registered moved_mr* / ct / mask, RAS, CT clip (-1024,1024), no runtime registration), src/preprocess/{register_single_subject,resample_and_split_dataset}.py (offline register + 1.5mm resample), pyradplan/eval_real_sct.py (real-sCT plan transfer + gamma) & sct_dose_eval.py (proxy), src/evaluate/total_seg.py. PET-AC: no such code in the core repo (grep-verified).

On-disk paths under `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/`. Figures rendered from one representative subject per dataset; reproduce via `~/ext_ds_analysis/analyze.py`.
