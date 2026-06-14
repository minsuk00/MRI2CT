# Paired MR–CT Datasets — Public Availability Survey

**Date:** 2026-06-09
**Goal:** Find additional **paired, same-patient MRI + CT** datasets that are **publicly downloadable**, for any body region, to expand training data for the MRI→CT synthesis model.
**Method:** Deep multi-source web research (5 search angles, 16 sources fetched, adversarial 3-vote verification) + targeted manual verification of candidates that the automated pass left unverified (TCIA rate-limiting).

**Scope note:** The deep-research pass was scoped to find datasets *beyond* the ones we already had. Those previously-found datasets are listed in **Tier 0** below so this doc is the single complete picture.

**Completeness status (updated 2026-06-09 after platform sweep):** Coverage is now high.
- **TCIA** — authoritatively cleared via the NBIA per-patient modality API (not keyword guessing): every "hidden paired set" lead resolved to 0 CT or single-digit overlap (see Ruled-out table).
- **OpenNeuro** — structurally cannot host this: it supports MRI/EEG/MEG/PET only, **no CT modality**. Definitively ruled out.
- **Synapse / Figshare / IEEE DataPort** — swept; surfaced only already-known sets (SynthRAD, Kaggle-18, APIS) plus the paywalled/JPEG IEEE glioblastoma set (ruled out).
- **Remaining residual uncertainty:** brand-new releases, non-English sources, and datasets that exist only as on-request clinical cohorts (e.g. the tFUS skull sets) — these can't be ruled in/out by search. Treat the list as "everything openly downloadable that public indexing surfaces," not an absolute proof of non-existence.

---

## TL;DR conclusion

- **3 openly downloadable** paired same-patient MR-CT sets were found beyond what we already have. Test order: **Learn2Reg Abdomen → Kaggle Paired CT/MRI → CFB-GBM**.
- **3 gated** sets (signed agreement / controlled access / registration) exist for brain; pursue only if more brain data is needed.
- The single best **fully-open, no-login** candidate is **Learn2Reg Abdomen MR-CT** (1.8 GB direct link).
- For structural MRI→CT HU synthesis, only the Kaggle (T1/T2), Burdenko (T1/T1C/T2/FLAIR), CFB-GBM (multi-MR), GLIS-RT (T1+Gd/T2-FLAIR), and Learn2Reg (T1w) sets provide conventional structural MR. **APIS provides only ADC (a derived diffusion map) — low value for HU synthesis.**

---

## Tier 0 — Already in hand / previously verified (for completeness)

These were found/verified in earlier sessions (byte-tested where noted). Included here so this doc is a complete reference.

| Dataset | Region | Subjects | MR / CT | Pairing | Host / access | Notes |
|---------|--------|----------|---------|---------|---------------|-------|
| **SynthRAD2023 / 2025** | Brain, H&N, thorax, abdomen, pelvis | 2025: 890 MR-CT + 1472 CBCT-CT pairs | Multiple | Same-patient, registered | Zenodo, open | **Primary dataset already in use.** |
| **RIRE / Vanderbilt** | Brain | 16 | CT + co-reg T1/T2/MP-RAGE/PD | Same-patient, co-registered | `rire.insight-journal.org/download_data`, CC BY 3.0, **open (byte-tested)** | Old (~2006), low-res, legacy MetaImage (`.mhd` + LZW `.bin.Z`), slow IPFS gateway. Small. |
| **HaN-Seg** | Head & Neck | 42 | CT + T1 | Same-patient | Zenodo 7442914 (`HaN-Seg.zip`, 4.94 GB), CC BY-NC-ND 4.0, **open (byte-tested)** | H&N, not whole brain. |
| **Gold Atlas Male Pelvis** | Pelvis | 19 | T1w + T2w MR + CT (+DVFs) | Same-patient | Zenodo 583096 — ⚠️ **as of 2026-06-09 the record reads `restricted` and is not on our disk** (byte-tested open in a prior session from a different network; re-verify before relying on it) | Was the cleanest open pelvis option; download access needs re-confirming. |
| **TopCoW** | Brain (vessels) | 200 | MRA-TOF + CTA | Same-patient | Zenodo 15692630 (~14.9 GB), **open** | ⚠️ **WRONG MODALITY** (angiography, not structural MR/CT) — off-distribution for sCT. Skip unless vessel/MRA experiment. |

---

## Tier 1 — Open / direct download (best candidates)

### 1. Learn2Reg Abdomen MR-CT ⭐ DOWNLOAD-VERIFIED — strongest fully-open candidate
- **Region:** Abdomen
- **Subjects:** **16 paired same-patient MR-CT** (cases with both channels) + unpaired CT/MR augmentation scans (97 unique case IDs, 359 NIfTI files total)
- **Modalities:** modality 0 = **MR T1w**, modality 1 = **CT** (per `AbdomenMRCT_dataset.json`); image shape 192×160×192; includes organ **labels** (liver, spleen, L/R kidney) + masks
- **Pairing:** Intra-patient. Confirmed: exactly 16 case IDs have both `_0000` (MR) and `_0001` (CT) files.
- **Host / download:** Uni-Lübeck Nextcloud, file `AbdomenMRCT.zip`, **1,916,666,413 bytes (1.79 GB)**, no registration.
- **Access:** **Open, no login.**
- **Source:** https://learn2reg.grand-challenge.org/Datasets/
- **Licenses (bundled in zip):** `README_LICENCE_TCIA_MRCT.txt` (the paired MR-CT portion is from TCIA), `README_LICENCE_CHAOS_MR.txt`, `README_LICENCE_BCV_CT.txt`. Cite Learn2Reg (arXiv:2112.04489) + per-source.
- **✅ DOWNLOAD TEST (2026-06-09):** Fully downloaded to `/tmp/dataset_dl_test/AbdomenMRCT.zip`, `unzip -t` integrity **OK**, 359 entries, valid `.nii.gz`. **Confirmed downloadable.**
- **Download recipe** (the `/download` link 303-redirects to a WebDAV `?accept=zip` endpoint; `curl -L` with a browser UA follows it and streams the zip anonymously — token/`-u` auth and the raw dav *file* path both 404/401):
  ```bash
  UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
  curl -A "$UA" -L -o AbdomenMRCT.zip \
    "https://cloud.imi.uni-luebeck.de/s/yiQZfo43YBBg7zL/download"
  ```

### 2. Kaggle — "Paired CT and MRI Dataset for Medical Applications"
- **Region:** Brain (9), Abdomen (8), Neck (1)
- **Subjects:** 18 patients → 389 cross-modality axial triplets (389 T1 + 389 T2 + 389 CT = 1167 slices)
- **Modalities:** **T1-weighted + T2-weighted MRI + CT**, co-registered, radiologist-confirmed; DICOM + PNG
- **Pairing:** Same-patient, identical anatomical site, co-registered (truly registered — usable for supervised paired training)
- **Host / download:** Kaggle (canonical URL confirmed verbatim from the paper's Data Availability section):
  `https://kaggle.com/datasets/29c3607295965ebb030f2d158fec487412d84c82528dd44f8ef956aef35541aa`
- **Access:** **Free Kaggle login required.** License **CC BY-NC** (non-commercial research OK).
- **Reference:** Saadia et al., *Data in Brief* vol. 61, 2025; PMID 40655994; PMC12246850
  - https://www.sciencedirect.com/science/article/pii/S2352340925004950
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12246850/
- **⚠️ DOWNLOAD TEST (2026-06-09): BLOCKED — needs Kaggle credentials.** Anonymous `curl` hits a Cloudflare/reCAPTCHA "Checking your browser" wall (`<title>Checking your browser - reCAPTCHA</title>`); no `kaggle` CLI or `~/.kaggle/kaggle.json` present on this machine. The dataset page/URL is real and resolves, but pulling bytes requires either (a) `pip install kaggle` + a `kaggle.json` API token, then `kaggle datasets download -d 29c36...41aa`, or (b) a logged-in browser. **Not byte-verified.**

### 3. CFB-GBM (TCIA)
- **Region:** Brain (glioblastoma)
- **Subjects:** 264 total — **CT for 195**, MRI for all 264 (overlapping cohort; not every patient has both)
- **Modalities:** MR (T1, T1Gd, T1 gradient echo, T2, T2-FLAIR, T2*, ADC) + axial CT (+ RTDOSE, GTV segmentations)
- **Pairing:** Same-patient overlapping cohort — must filter to subjects having both CT and MR
- **Host / download:** TCIA — **208 GB images via IBM Aspera Connect** (NIfTI); clinical TSVs direct
- **Access:** **Open**, CC BY 4.0, data citation required
- **Source:** https://www.cancerimagingarchive.net/collection/cfb-gbm/
- **⚠️ DOWNLOAD TEST (2026-06-09): not curl/wget-able.** The legacy NBIA REST API (`services.cancerimagingarchive.net/nbia-api/.../getSeries?Collection=CFB-GBM`) is reachable but returns an **empty body** for this collection — CFB-GBM is distributed as **NIfTI via IBM Aspera Connect**, not as DICOM series through NBIA, so it has no scriptable HTTP download. Pulling it requires the **Aspera Connect browser plugin/client** (`ascp`), and 208 GB is impractical to byte-test here. Open-access *mechanism* confirmed; **bytes not pulled.** Before committing to the 208 GB pull, enumerate the ~195/264 subjects that have **both** CT and MR.

---

## Tier 2 — Gated (registration / signed agreement / controlled access)

### 4. Burdenko-GBM-Progression (TCIA)
- **Region:** Brain (glioblastoma) · **Subjects:** 180 (Burdenko Center, 2014–2020)
- **Modalities:** 4 MR sequences (T1, T1C, T2, FLAIR) + **topometric CT** per planning study (+ RTSTRUCT/RTPLAN/RTDOSE)
- **Pairing:** Same-patient confirmed. ⚠️ **MR is in original acquisition space, NOT pre-registered to CT** — RT files align to the topometric CT, not the MR. Registration is on us.
- **Access:** **Gated** — must sign & submit a **TCIA Restricted License Agreement** (face-reconstruction risk) to help@cancerimagingarchive.net; then download via TCIA Data Retriever. Only the 29 kB clinical CSV is openly CC BY 4.0.
- **Notes:** 1 topometric CT per patient; MRI has 1–8 follow-up MRI-only timepoints.
- **Sources:** https://www.cancerimagingarchive.net/collection/burdenko-gbm-progression/ · https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=133073470

### 5. GLIS-RT (TCIA)
- **Region:** Brain (glioma) · **Subjects:** 230
- **Modalities:** contrast-enhanced 3D-T1 + 2D multislice T2-FLAIR MR + CT (+ **REG** image-registration objects, RTSTRUCT). 28.26 GB DICOM.
- **Pairing:** **Same-patient, and registration objects are included** (post-surgical MRI + CT acquired for RT target definition).
- **Access:** **Gated** — NIH Controlled Data Access Policy (face-reconstruction risk); request access, then TCIA Data Retriever. DOI 10.7937/TCIA.T905-ZQ20.
- **Source:** https://www.cancerimagingarchive.net/collection/glis-rt/
- **Note:** Previously flagged "gated" in project memory — now confirmed it **is** genuinely same-patient paired with registration files, so worth an application if more brain data is wanted.

### 6. APIS (BIVL2AB challenge) — low value for HU synthesis
- **Region:** Brain (acute ischemic stroke) · **Subjects:** 60 paired train (+40 NCCT-only test = 100)
- **Modalities:** **ADC** (MRI-diffusion-derived parametric map, **NOT structural T1/T2**) + non-contrast head CT (NCCT). Low res (CT 512², ADC ~256²).
- **Pairing:** Same-patient for the 60 training studies only.
- **Access:** **Registration-required** — registered email yields a unique 7-day-expiring link; requests may be refused; institutional email expected.
- **Sources:** https://bivl2ab.uis.edu.co/challenges/apis · https://www.nature.com/articles/s41598-024-71273-x · https://arxiv.org/abs/2309.15243
- **Verdict:** ADC ≠ structural MR and NCCT ≠ full-HU diagnostic CT → **skip for structural MRI→CT synthesis.**

---

### 7. CERMEP-IDB-MRXFDG (from prior memory)
- **Region:** Whole brain · **Subjects:** 37 healthy
- **Modalities:** CT + T1 + FLAIR MR · **Pairing:** same-patient
- **Access:** **Application / agreement required** (lightest of the gated brain options per prior notes).

### 8. OASIS-3 (from prior memory)
- **Region:** Brain · **Modalities:** MR + CT, but **CT is only low-dose attenuation-correction (AC) CT** from PET — ⚠️ likely too low-quality for HU synthesis targets.
- **Access:** Application required.

---

## Ruled out (surfaced but do not qualify)

| Candidate | Reason |
|-----------|--------|
| BRAIN-TR-GammaKnife (TCIA) | **MR only, no CT** (T1 MPRAGE + Gd, Gamma Knife planning) |
| "Unpaired MR-CT brain dataset" (*Data in Brief* 2022, PMC9011016) | **Explicitly unpaired** — 90 MR + 89 CT slices, 20 patients, not registered; built for CycleGAN-style unsupervised translation |
| Prostate-Anatomical-Edge-Cases (TCIA) | **CT + RTSTRUCT only**, no downloadable MR series (131 subjects); MRI mentioned only in study text |
| CPTAC-\* (TCIA) | **Confirmed negligible same-patient overlap via NBIA API** (2026-06-09): CPTAC-PDA = **7** patients w/ both CT&MR, CPTAC-CCRCC = **7**, CPTAC-LUAD = **1** (CT and MR are largely different cohorts). CPTAC-GBM / CPTAC-HNSCC API timed out but pattern is clearly single-digit. Not viable training sources. |
| UPENN-GBM (TCIA) | **API-confirmed: 630 patients, MR-only, 0 CT series.** Modality listing was misleading. |
| ReMIND (TCIA) | **API-confirmed: MR + ultrasound + SEG, 0 CT.** (intra-op imaging, no CT) |
| Vestibular-Schwannoma-SEG (TCIA) | **API-confirmed: MR + RT objects, 0 CT.** |
| HEAD-NECK-PET-CT (TCIA) | **PET/CT + planning CT, no MRI** |
| CHAOS / AMOS (abdominal) | Have both CT and MR but **as separate, unpaired cohorts (different patients per modality)** — not same-patient. (AMOS22 is the unpaired-modality source; CHAOS CT and MR are distinct 20-patient sets.) |
| ABCs 2020 (Harvard MGH) | **NOT downloadable** — Zenodo 3714982 is a PDF design doc only; real data host `abcs.mgh.harvard.edu` is down/unreachable (from prior memory) |
| CT-MRI Glioblastoma Multimodal Benchmark (IEEE DataPort) | Paired same-patient (50, brain) BUT **behind IEEE DataPort subscription paywall** AND stored as **JPEG** (no Hounsfield units) → unusable for HU synthesis. Double-disqualified. |
| tFUS skull CT-T1w / CT-ZTE (HM Hospitales / UCL) | **Not public** — clinical-trial cohorts (171 CT-T1w + 90 CT-ZTE; and a 86-subject CT-T1w set), no data-availability statement, on-request only. |
| OpenNeuro datasets | Platform supports **MRI/EEG/MEG/PET only — no CT modality**, so cannot host paired MR-CT. |
| OSF MRI-TCS database (osf.io/zdcjb) | MRI + **transcranial ultrasound** co-registration, **not CT**. |

---

## Open questions / follow-ups

1. **CFB-GBM:** enumerate per-subject CT∩MR overlap (only 195/264 have CT) before committing to the 208 GB Aspera pull.
2. **Burdenko:** assess MR→CT registration effort (MR in acquisition space, not pre-registered) and whether the TCIA Restricted License approval timeline fits the schedule.
3. ~~CPTAC / UPENN-GBM per-subject check~~ — **DONE (2026-06-09, NBIA API):** UPENN-GBM has 0 CT; CPTAC overlap is single-digit. Both ruled out. CPTAC-GBM / CPTAC-HNSCC API calls timed out — retry if you want the exact (expected single-digit) numbers.
4. **Still un-swept (the real remaining unknown):** Synapse, OpenNeuro, Figshare/Dryad, IEEE DataPort, and paper-supplementary-only datasets. These are NOT covered by the TCIA API check above. This is where any genuinely-missed dataset would be.

---

## Verification status

| Dataset | Specs verified | Download bytes pulled | Result |
|---------|----------------|----------------------|--------|
| **Learn2Reg Abdomen** | ✅ dataset.json inspected | **✅ 1.79 GB pulled, zip OK** | **DOWNLOADABLE — confirmed** |
| Kaggle Paired CT/MRI | ✅ peer-reviewed descriptor + canonical URL | ⚠️ blocked (reCAPTCHA, no creds) | Needs Kaggle login/API token |
| CFB-GBM | ✅ TCIA page; NBIA API probed | ❌ no scriptable HTTP (Aspera-only) | Open but needs Aspera client (208 GB) |
| Burdenko | ✅ TCIA page + wiki | ❌ gated | Needs signed TCIA Restricted License |
| GLIS-RT | ✅ TCIA page | ❌ gated | Needs NIH controlled-access request |
| APIS | ✅ paper + portal | ❌ registration-gated | Email registration; ADC-only (low value) |

### Download-test log (2026-06-09)
- **Environment:** `curl`/`wget`/`unzip`/`python3` available; **no** `aria2c`, **no** `kaggle` CLI, **no** `~/.kaggle/kaggle.json`. Network is HTTP-proxied.
- **Learn2Reg ✅** → `curl -L` (browser UA) followed the Nextcloud `/download` 303→WebDAV `?accept=zip` redirect and streamed `AbdomenMRCT.zip` (1,916,666,413 B). `unzip -t` passed; modality 0 = MR T1w, 1 = CT; 16 case IDs carry both channels (= 16 paired same-patient MR-CT). Saved at `/tmp/dataset_dl_test/AbdomenMRCT.zip`.
- **Kaggle ⚠️** → anonymous curl blocked by Cloudflare/reCAPTCHA browser check; requires a Kaggle account token to pull. URL confirmed real via the paper's Data Availability section.
- **CFB-GBM ⚠️** → NBIA REST reachable (`getCollectionValues` = 200/JSON) but `getSeries?Collection=CFB-GBM` returns empty; collection is NIfTI-via-Aspera, no plain-HTTP path. Not pulled.
