import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim3d


def get_class_dice(logits, target_mask, mask=None, bone_idx=5):
    """
    Soft dice per class. Optionally restricted to body voxels.
    logits:      [B, C, X, Y, Z] raw logits from teacher
    target_mask: [B, 1, X, Y, Z] integer GT labels
    mask:        [1, 1, X, Y, Z] binary body mask (optional); when provided,
                 dice is computed only on body voxels (B=1 assumed)
    Returns: class_dices [C], bone_dice scalar (or None if bone_idx out of range)
    """
    smooth = 1e-5
    # Cast to fp32 before softmax: fp16 probs.sum() over millions of voxels
    # overflows (fp16 max ≈ 65504), zeroing high-mass classes' Dice (e.g. bone).
    # Triggered when val data is fp16 via use_float16_storage. bf16 is safe
    # (fp32-range exponent), but fp32 is cheap insurance.
    probs = F.softmax(logits.float(), dim=1)  # [B, C, X, Y, Z]
    if target_mask.ndim == 5:
        target_mask = target_mask.squeeze(1)  # [B, X, Y, Z]

    if mask is not None:
        # Body-voxel path: extract masked voxels, compute dice on flattened body region
        mask_bool = mask.squeeze().bool()  # [X, Y, Z]
        probs_flat = probs.squeeze()[:, mask_bool]  # [C, N_body]
        seg_flat = target_mask.squeeze()[mask_bool].long()  # [N_body]
        seg_oh = F.one_hot(seg_flat, num_classes=probs.shape[1]).float().T  # [C, N_body]
        intersection = (probs_flat * seg_oh).sum(dim=1)
        union = probs_flat.sum(dim=1) + seg_oh.sum(dim=1)
        class_dices = (2.0 * intersection + smooth) / (union + smooth)  # [C]
    else:
        # Full-volume path (original behavior, supports batch > 1)
        num_classes = probs.shape[1]
        target_one_hot = F.one_hot(target_mask.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        p = probs.view(probs.shape[0], num_classes, -1)
        t = target_one_hot.view(target_one_hot.shape[0], num_classes, -1)
        intersection = (p * t).sum(dim=2)
        union = p.sum(dim=2) + t.sum(dim=2)
        class_dices = ((2.0 * intersection + smooth) / (union + smooth)).mean(dim=0)  # [C]

    bone_dice = class_dices[bone_idx] if class_dices.shape[0] > bone_idx else None
    return class_dices, bone_dice


class AnatomixPerceptualLoss(nn.Module):
    """Perceptual loss between Anatomix decoder features of pred vs target.

    Runs a frozen Anatomix U-Net on both images and compares multi-scale decoder feature maps.
    Default layers [38, 45, 52, 65] are the three decoder convs + the final 16-D anatomix
    descriptor, i.e. the decoder subset of the layers the contrastive (NCE) objective was
    applied to during v1_4 pretraining (27,31,38,45,52,65). pred/target are CT in [0, 1],
    shape (B, 1, D, H, W). Gradients flow through pred only.

    metric:
      "ncc" — (default) 1 − local squared normalized cross-correlation (LNCC): squared
              Pearson correlation within sliding 7³ box windows, averaged over windows,
              channels, and samples. Compares local feature *patterns* invariant to a
              per-window affine change (incl. sign); a strong spatial-structure term. In [0, 1].
      "l1"  — mean L1 distance between feature maps (AFP-loss convention).

    NOTE: when this perceptual loss is active (perceptual_w > 0), set ssim_w = 0.
    Perceptual REPLACES SSIM as the structural-similarity term on top of L1 — using
    both double-counts structure. (Our perceptual-vs-baseline arms differ by exactly
    this swap: baseline = L1+SSIM+dice, perceptual = L1+perceptual+dice.)
    """

    CKPT = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
    DEFAULT_LAYERS = [38, 45, 52, 65]

    def __init__(self, layers=None, device="cuda", metric="ncc", separable=True, compile_lncc=False):
        super().__init__()
        from anatomix.model.network import Unet

        from common.utils import clean_state_dict

        self.extractor = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(device)
        self.extractor.load_state_dict(clean_state_dict(torch.load(self.CKPT, map_location=device)), strict=True)
        for p in self.extractor.parameters():
            p.requires_grad = False
        self.extractor.eval()
        self.layers = sorted(layers) if layers else list(self.DEFAULT_LAYERS)
        self.metric = metric.lower()
        assert self.metric in ("l1", "ncc"), f"unknown perceptual metric {metric!r} (use 'l1' or 'ncc')"
        self.dist = nn.L1Loss()
        # separable: factor the k^3 box sums into 3x 1-D convs (algebraically identical, ~3.5x
        # faster). compile_lncc: fuse the elementwise variance/correlation arithmetic via
        # torch.compile (convs stay on cuDNN); numerically equivalent up to fp reordering (~1e-7).
        self.separable = separable
        self._lncc = torch.compile(self._ncc_loss) if compile_lncc else self._ncc_loss
        print(f"[DEBUG] AnatomixPerceptualLoss (v1_4) metric={self.metric} layers {self.layers} separable={self.separable} compiled={compile_lncc}")

    @staticmethod
    def _ncc_loss(pred, target, *, kernel_size=7, eps=1e-5, separable=True):
        """Local zero-normalized cross-correlation (LNCC) loss for 2D or 3D features.

        Slides a rectangular box window over the spatial dimensions and, within each
        window, computes the *squared* Pearson correlation between ``pred`` and
        ``target`` per (sample, channel). The result is averaged over all windows,
        channels, and samples, and returned as ``1 - mean(ncc)``.

        Because the correlation is computed locally and is squared, the loss is
        invariant to a per-window affine change (mean shift and rescaling, including
        sign) of either input within the image patch.

        Args:
            pred:        Raw (un-normalized) features, shape ``(N, C, H, W)`` for 2D
                         or ``(N, C, D, H, W)`` for 3D.
            target:      Reference features, same shape as ``pred``.
            kernel_size: Side length of the cubic/square box window. Uses ``same``
                         padding, so the output covers every input location.
            eps:         Small constant added to variances and the correlation ratio
                         for numerical stability.

        Returns:
            Scalar tensor in ``[0, 1]``. ``0`` means every local window is perfectly
            correlated (up to affine scaling); larger values mean weaker local
            agreement.
        """
        C = pred.shape[1]
        ndim = pred.dim() - 2  # number of spatial dims (2 or 3)
        if ndim not in (2, 3):
            raise ValueError(f"expected 4D or 5D input, got {pred.dim()}D")
        conv = F.conv2d if ndim == 2 else F.conv3d
        n_win = kernel_size**ndim
        pad = kernel_size // 2

        # Force fp32 with autocast disabled: this LNCC computes variance/covariance via
        # the unstable "sum of squares minus square of sums" identity (a difference of two
        # large, near-equal box sums). Under the bf16 autocast used in training, conv3d
        # would downcast and that subtraction suffers catastrophic cancellation -> garbage
        # variances clamp to eps and the squared-correlation ratio blows up (~1e7). Merely
        # .float()-ing the inputs is not enough: autocast re-downcasts conv operands, so we
        # must disable it for this block. Same precision guard as get_class_dice() above.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.float()
            target = target.float()

            if separable:
                # Separable box: a k^ndim all-ones kernel is rank-1 = ones_k (x) ... (x) ones_k,
                # so the window sum factors into ndim successive 1-D convs (k^ndim -> ndim*k taps).
                # Algebraically identical to the dense box (fp32 agrees to ~1e-7); ~3.5x faster
                # because the grouped 3-D conv is memory-bound.
                axis_kernels, axis_pads = [], []
                for a in range(ndim):
                    ksz = [1] * ndim
                    ksz[a] = kernel_size
                    axis_kernels.append(pred.new_ones(C, 1, *ksz))  # 1-D ones-kernel along axis a
                    ap = [0] * ndim
                    ap[a] = pad
                    axis_pads.append(tuple(ap))

                def box(x):
                    for k, ap in zip(axis_kernels, axis_pads):
                        x = conv(x, k, padding=ap, groups=C)  # group=C -> no cross-channel mixing
                    return x
            else:
                kernel = pred.new_ones(C, 1, *([kernel_size] * ndim))  # per-channel box sum

                def box(x):
                    return conv(x, kernel, padding=pad, groups=C)  # group=C -> no cross-channel mixing

            t, p = target, pred
            t_sum, p_sum = box(t), box(p)
            t2_sum, p2_sum = box(t * t), box(p * p)
            tp_sum = box(t * p)
            cross = tp_sum - p_sum * t_sum / n_win  # n_win * local covariance
            t_var = (t2_sum - t_sum * t_sum / n_win).clamp_min(eps)
            p_var = (p2_sum - p_sum * p_sum / n_win).clamp_min(eps)
            ncc = (cross * cross + eps) / (t_var * p_var + eps)  # squared correlation
            return 1.0 - ncc.mean()

    def _dist(self, p, t):
        return self.dist(p, t) if self.metric == "l1" else self._lncc(p, t, separable=self.separable)

    def forward(self, pred, target):
        with torch.no_grad():
            tgt = self.extractor(target, layers=self.layers, encode_only=True)
        prd = self.extractor(pred, layers=self.layers, encode_only=True)
        return sum(self._dist(p, t) for p, t in zip(prd, tgt)) / len(prd)


class CompositeLoss(nn.Module):
    def __init__(self, weights, perceptual=None):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.perceptual = perceptual

    def forward(self, pred, target, pred_probs=None, target_mask=None, compute_perceptual=True):
        """Returns (total_loss, loss_components).

        total_loss is the WEIGHTED sum of the active terms. The values in loss_components
        (loss_l1, loss_ssim, loss_perceptual, loss_dice, ...) are RAW, unweighted per-term
        losses — these are what get logged to wandb (train/l1, train/ssim, train/perceptual,
        ...). Only train/total is scaled by `weights`. Read the raw values to pick weights:
        e.g. set perceptual_w so perceptual_w * loss_perceptual is comparable to l1_w * loss_l1.

        compute_perceptual: set False during validation. The perceptual extractor's
        full-volume decoder forward OOMs on val-size volumes, so it's skipped there.
        (Dice is excluded from val separately, by not passing pred_probs — see validate_dice.)
        """
        total_loss = torch.tensor(0.0, device=pred.device)
        loss_components = {}

        # 1. L1 Loss
        l1_val = self.l1(pred, target)
        if self.weights.get("l1", 0) > 0:
            loss_components["loss_l1"] = l1_val
            total_loss += self.weights["l1"] * l1_val

        # 2. L2 Loss
        if self.weights.get("l2", 0) > 0:
            l2_val = self.l2(pred, target)
            loss_components["loss_l2"] = l2_val
            total_loss += self.weights["l2"] * l2_val

        # 3. SSIM Loss
        if self.weights.get("ssim", 0) > 0:
            ssim_val = 1.0 - fused_ssim3d(pred.float(), target.float(), train=True)
            loss_components["loss_ssim"] = ssim_val
            total_loss += self.weights["ssim"] * ssim_val

        # 4. Anatomix Perceptual Loss
        if compute_perceptual and self.perceptual is not None and self.weights.get("perceptual", 0) > 0:
            perc_val = self.perceptual(pred, target)
            loss_components["loss_perceptual"] = perc_val
            total_loss += self.weights["perceptual"] * perc_val

        # 5. Dice Losses (Teacher-guided)
        if pred_probs is not None and target_mask is not None:
            bone_idx = self.weights.get("dice_bone_idx", 5)
            class_dices, bone_dice = get_class_dice(pred_probs, target_mask, bone_idx=bone_idx)
            C = class_dices.shape[0]

            # Per-class weight vector over ALL classes (background included): every
            # class defaults to dice_w; bone is REPLACED by dice_bone_w (not added).
            # Dice is applied once as a mean over all classes, so when
            # dice_bone_w == dice_w this reduces exactly to dice_w * mean(1 - dice).
            w = torch.full((C,), self.weights.get("dice_w", 0.0), device=class_dices.device, dtype=class_dices.dtype)
            if bone_idx < C:
                w[bone_idx] = self.weights.get("dice_bone_w", 0.0)

            if torch.any(w > 0):
                total_loss += (w * (1.0 - class_dices)).sum() / C

            # Diagnostics (unweighted; logged to wandb, no gradient impact).
            gen_dice = class_dices.mean()
            loss_components["dice_score_all"] = gen_dice
            loss_components["loss_dice"] = 1.0 - gen_dice
            if bone_dice is not None:
                loss_components["dice_score_bone"] = bone_dice
                loss_components["loss_dice_bone"] = 1.0 - bone_dice

        return total_loss, loss_components
