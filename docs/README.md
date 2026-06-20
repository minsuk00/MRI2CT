# docs/ — agent-facing research record

Markdown mirrors of the project's HTML reports in `_html/`. The HTML versions (with embedded figures) are for humans; these `.md` files are for LLM agents: same data tables and conclusions, figures reduced to one-line descriptions, no base64 image blobs.

Each doc starts with a header block: source HTML path, date, and a TL;DR. Data tables are preserved verbatim. To see a figure, open the corresponding `_html/*.html`.

## How to use this directory
- Need the current model standings? Read the latest eval snapshot: **[21_full_eval_20260617](21_full_eval_20260617.md)**.
- Need the bone-failure story (the project's central finding)? Read **[09](09_unet_error_anatomy.md) -> [13](13_mr_bone_information_limit.md) -> [17](17_bone_solution_study.md)** in that order.
- Need available training data? Read **[01](01_paired_mr_ct_datasets.md)** (survey) and **[03](03_external_dataset_analysis.md)** (on-disk specs).

## Datasets
| doc | source HTML | summary |
|-----|-------------|---------|
| [01_paired_mr_ct_datasets](01_paired_mr_ct_datasets.md) | (deep-research survey) | Public paired same-patient MR-CT availability survey; best open candidate Learn2Reg Abdomen. |
| [02_dataset_analysis](02_dataset_analysis.md) | dataset_analysis.html | Byte-verified usable paired data: SynthRAD (in use), Learn2Reg AbdomenMRCT, RIRE. |
| [03_external_dataset_analysis](03_external_dataset_analysis.md) | external_dataset_analysis.html | Technical specs for 6 on-disk external sets; CFB-GBM and Gold Atlas are training-ready. |

## Survey & novelty
| doc | source HTML | summary |
|-----|-------------|---------|
| [04_sota_mri2ct_translation](04_sota_mri2ct_translation.md) | 01_sota_mri2ct_translation.html | SOTA survey 2022-2025 + the 11-experiment bone-ceiling campaign. Plain U-Net is the hardest baseline. |
| [05_amix_bone_findings](05_amix_bone_findings.md) | 01a_amix_bone_findings.html | What the 11 experiments measured: anatomix does not help; bone is information-limited. |
| [08_translation_novelty](08_translation_novelty.md) | 03_3d_medical_translation_novelty.html | Methodological-novelty map; diffusion is NOT SOTA for MR->CT; live lanes are decision-aware objective/uncertainty. |

## MAISI VAE audit
| doc | source HTML | summary |
|-----|-------------|---------|
| [06_maisi_vae_brain_hu_audit](06_maisi_vae_brain_hu_audit.md) | 02_maisi_vae_brain_hu_audit.html | Brain HU-inflation audit. **Original skull-bleed conclusion RETRACTED** (sliding-window harness artifact); cause reopened. |
| [07_maisi_latent_fix_verification](07_maisi_latent_fix_verification.md) | 02a_maisi_latent_fix_verification.html | The dynamic_infer fix removes the VAE GroupNorm-over-zero-padding artifact. |

## Bone-failure / error-anatomy series
| doc | source HTML | summary |
|-----|-------------|---------|
| [09_unet_error_anatomy](09_unet_error_anatomy.md) | 04_unet_error_anatomy.html | Where the U-Net fails: cortical-bone undershoot, information-limited, invisible to reported PSNR. |
| [10_unet_failure_anatomy](10_unet_failure_anatomy.md) | 05_unet_failure_anatomy.html | Per-CADS-label, per-region failure diagnosis; bone carries error mass beyond its volume. |
| [11_unet_bone_deepdive](11_unet_bone_deepdive.md) | 06_unet_bone_deepdive.html | Is bone the biggest problem? Worst per-voxel, but air/soft are bigger aggregate PSNR levers. |
| [12_crossmodel_bone](12_crossmodel_bone.md) | 07_crossmodel_bone.html | Bone undershoot is universal across all 6 models (capped and uncapped). |
| [13_mr_bone_information_limit](13_mr_bone_information_limit.md) | 08_mr_bone_information_limit.html | Conclusive test that standard MR lacks bone-density info (model-free k-NN + literature). |
| [14_unet_seg_downstream](14_unet_seg_downstream.md) | 09_unet_seg_downstream.html | Seg-downstream view: bone fails two ways (HU undershoot + blur); four falsification tests. |
| [15_cads_error_decomposition](15_cads_error_decomposition.md) | 10_cads_error_decomposition.html | U-Net error by GT CADS label; ~18% of "body" is loose-mask external air. |
| [16_cads_multimodel_decomposition](16_cads_multimodel_decomposition.md) | 11_cads_multimodel_decomposition.html | The CADS decomposition replicated across all six models; shared failure structure. |
| [17_bone_solution_study](17_bone_solution_study.md) | 12_bone_solution_study.html | Toy + real-data proof bone is information-limited; bounds what any fix can achieve. |

## Pipeline & infrastructure
| doc | source HTML | summary |
|-----|-------------|---------|
| [18_cads_merge_report](18_cads_merge_report.md) | cads_merge_report.html | Merge 9 CADS task models per subject by priority painting, no region gating (gating reverted). |
| [26_compile_mode_benchmark](26_compile_mode_benchmark.md) | compile_mode_benchmark.html | `compile_mode="model"` wins on both trainers; `"full"` shatters into broken sub-graphs. |
| [27_training_budget](27_training_budget.md) | training_budget.html | Reference budget = amix/unet ~3.20M 128^3 patches (wandb-derived); per-baseline parity. |

## Full model evaluations (dated snapshot series)
| doc | source HTML | summary |
|-----|-------------|---------|
| [19_full_eval_20260601](19_full_eval_20260601.md) | full_eval_20260601.html | Six-model eval, center-wise OOD val (207 subj). |
| [20_full_eval_20260609](20_full_eval_20260609.md) | full_eval_20260609.html | Same, newer MAISI + cWDM checkpoints. |
| [21_full_eval_20260617](21_full_eval_20260617.md) | full_eval_20260617.html | Latest snapshot. Macro MAE: unet < amix < koalAI < mcddpm < cwdm < maisi. |
| [22_full_eval_ep600_parity](22_full_eval_ep600_parity.md) | full_eval_ep600_parity.html | Budget-parity: unet@ep600 beats koalAI at matched ~2.4-2.5M samples. |

## Perceptual / anatomix-injection ablations (dated snapshot series)
| doc | source HTML | summary |
|-----|-------------|---------|
| [23_perc_ablation_20260603](23_perc_ablation_20260603.md) | perc_ablation_20260603.html | ep400/200k-step; perceptual marginally helps body MAE. |
| [24_perc_ablation_100k_20260609](24_perc_ablation_100k_20260609.md) | perc_ablation_100k_20260609.html | ep200/100k-step single-axis perceptual on/off. |
| [25_perc_ablation_20260617](25_perc_ablation_20260617.md) | perc_ablation_20260617.html | Anatomix injection point: nowhere vs loss vs backbone. |
