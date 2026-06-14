# MONAI Migration Plan

## 1. Analysis
**Goal:** Migrate the MRI-to-CT data loading pipeline from `torchio` to `monai` while maintaining mathematical parity with the current 128³ patch sampling and optimizing for a system with 48GB RAM and 44GB VRAM. We must ensure high inter-patient diversity in every training step.

**Technical Approach & Architectural Decisions:**
1. **Dictionary Representation:** Shift from `torchio.Subject` to standard Python dictionaries containing file paths, which natively integrates with MONAI's dictionary transforms (`*d`).
2. **Two-Stage Transform Architecture (Preprocessing vs. Augmentation):**
   - **Stage 1 - Preprocessing (CPU / PersistentDataset):** Deterministic, heavy operations (Load, ChannelFirst, Normalization, Padding). 
     - **Custom Parity Transforms:** To guarantee 100% parity with the current inference unpadding logic, we will write a custom MONAI `MapTransform` class (`PadToMultipleEndD`) to wrap the exact `pad to multiple of 32 at the end` logic. For normalization, we will use MONAI's built-in transforms conditionally based on the model: for AMIX/U-Net, MRI uses `ScaleIntensityRanged` (min-max) and CT uses `ScaleIntensityRanged` (-1024 to 1024); for MAISI, MRI uses `ScaleIntensityRangePercentilesd` (0-99.5%) and CT uses `ScaleIntensityRanged` (-1000 to 1000).
     - By limiting PyTorch dataloader workers to these deterministic tasks, we can aggressively cache the padded, normalized full-volume float16/float32 arrays to an NVMe drive via `monai.data.PersistentDataset`. This completely bypasses GPFS I/O bottlenecks.
   - **Stage 2 - Augmentation (GPU Main Thread):** With 44GB VRAM, we move the cached full volumes to the GPU. We then apply volume-level physics augmentations (`RandBiasFieldd`) $\rightarrow$ extract patches (`RandCropByPosNegLabeld`) $\rightarrow$ apply patch-level spatial/intensity augmentations (`RandAffined`, `RandGaussianNoised`, etc.) entirely on the GPU in milliseconds.
3. **Multi-Volume Batching (The Diversity Solution):**
   - **The Strategy (4 Volumes, 1 Patch Each):** To maintain the diversity of `torchio.Queue` without blowing up VRAM or using slow CPU patch-queuing, we will use a `DataLoader` with `batch_size=4` to load 4 distinct patients per step. 
   - **Crop:** `RandCropByPosNegLabeld` will extract `num_samples=1` patch per volume.
   - **Result:** The model receives a batch of 4 patches, where each patch is guaranteed to be from a completely different patient. This matches your current batch size of 4 but ensures maximum gradient stability and diversity.
4. **Collation & Dimension Handling (The Batching Issue):** 
   - **The Problem:** PyTorch Dataloaders yield a 5D tensor `(B=4, C, H, W, D)`. MONAI's spatial dictionary transforms strictly expect unbatched 4D tensors `(C, H, W, D)`.
   - **The Solution (Decollate -> Augment -> Recollate):**
     1. **Un-batch:** Inside the training loop, we use MONAI's `decollate_batch(batch)` to separate the 5D batch into a list of 4 individual 4D volumes, then move them to the GPU.
     2. **Crop & Augment:** We apply the GPU transforms to each 4D volume. `RandCropByPosNegLabeld` outputs a list of 1 patch per volume.
     3. **Re-batch:** We use MONAI's `list_data_collate` to stack the 4 resulting patch dictionaries back into a single batched 5D tensor `(4, C, 128, 128, 128)` for the forward pass.

## 2. Numbered Steps
1. **Refactor `src/common/data.py` (Est: 30-45 mins)**
   - Replace `build_tio_subjects` with a generic `build_data_dicts` function.
   - **Custom Transforms:** Implement a `PadToMultipleEndD` class extending `monai.transforms.MapTransform` to perfectly match the current custom padding logic. We will use MONAI's `ScaleIntensityRanged` for normalization.
   - Implement `get_cached_transforms()` (the deterministic CPU pipeline).
   - Implement `get_gpu_transforms()` containing `RandBiasFieldd` (volume), `RandCropByPosNegLabeld` (patch extraction, `num_samples=1`), and GPU patch augmentations.
2. **Update `src/amix/trainer.py` (Est: 30-45 mins)**
   - Swap `torchio.SubjectsDataset` with `monai.data.PersistentDataset`.
   - Swap `torchio.SubjectsLoader` with `monai.data.DataLoader` (`batch_size=4`, `shuffle=True`).
   - Inject the `decollate_batch` $\rightarrow$ `gpu_transforms` $\rightarrow$ `list_data_collate` execution directly into the training loop `for batch in self.train_loader:`.
3. **Update `src/unet_baseline/train.py` (Est: 15-20 mins)**
   - Replicate the dataloader and training loop changes established in the AMIX trainer.
4. **Update `src/maisi_baseline/trainer.py` (Est: 15-20 mins)**
   - Replicate the dataloader and training loop changes.
5. **Testing & Validation (Est: 30-60 mins)**
   - Run a short test run to verify tensor shapes (`(4, C, 128, 128, 128)`), verify GPU memory utilization scaling, ensure the NVMe cache generates correctly, and confirm the loss calculates properly (and unpadding works during validation).

## 3. File Impact
- **Modify:** `src/common/data.py`
- **Modify:** `src/amix/trainer.py`
- **Modify:** `src/unet_baseline/train.py`
- **Modify:** `src/maisi_baseline/trainer.py`

## 4. Risks / Assumptions
- **Assumption:** The cache directory (`/home/minsukc/MRI2CT/dataset/`) has sufficient capacity. Caching 100+ subjects uncompressed can consume 10-30GB.
- **Risk:** Implementing the custom `PadToMultipleEndD` MapTransform requires careful handling of metadata keys (like `original_shape`) to ensure the downstream sliding window inference unpads correctly.
- **Risk:** We must carefully configure the parameters of MONAI's `ScaleIntensityRanged` and `ScaleIntensityRangePercentilesd` to perfectly match the distinct normalization strategies (min-max vs. percentile) used by the different models in the legacy `anatomix_normalize` logic.

## 5. Approval
I am standing by for approval. I will not modify any files until I receive explicit confirmation from you to proceed.om you to proceed.