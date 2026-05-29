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
        probs_flat = probs.squeeze()[:, mask_bool]         # [C, N_body]
        seg_flat = target_mask.squeeze()[mask_bool].long() # [N_body]
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
    """Perceptual loss: mean L1 distance between Anatomix encoder features of pred vs target.

    Runs a frozen Anatomix U-Net on both images and compares multi-scale encoder feature maps.
    pred/target are CT in [0, 1], shape (B, 1, D, H, W). Gradients flow through pred only.
    """

    CKPT = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"

    def __init__(self, layers=None, device="cuda"):
        super().__init__()
        from anatomix.model.network import Unet

        from common.utils import clean_state_dict

        self.extractor = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(device)
        self.extractor.load_state_dict(clean_state_dict(torch.load(self.CKPT, map_location=device)), strict=True)
        for p in self.extractor.parameters():
            p.requires_grad = False
        self.extractor.eval()
        self.layers = sorted(layers) if layers else list(self.extractor.encoder_idx)
        self.dist = nn.L1Loss()
        print(f"[DEBUG] AnatomixPerceptualLoss (v1_4) using encoder layers {self.layers}")

    def forward(self, pred, target):
        with torch.no_grad():
            tgt = self.extractor(target, layers=self.layers, encode_only=True)
        prd = self.extractor(pred, layers=self.layers, encode_only=True)
        return sum(self.dist(p, t) for p, t in zip(prd, tgt)) / len(prd)


class CompositeLoss(nn.Module):
    def __init__(self, weights, perceptual=None):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()
        self.perceptual = perceptual

    def forward(self, pred, target, pred_probs=None, target_mask=None):
        """Returns (total_loss, loss_components).

        total_loss is the WEIGHTED sum of the active terms. The values in loss_components
        (loss_l1, loss_ssim, loss_perceptual, loss_dice, ...) are RAW, unweighted per-term
        losses — these are what get logged to wandb (train/l1, train/ssim, train/perceptual,
        ...). Only train/total is scaled by `weights`. Read the raw values to pick weights:
        e.g. set perceptual_w so perceptual_w * loss_perceptual is comparable to l1_w * loss_l1.
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
        ssim_val = 1.0 - fused_ssim3d(pred.float(), target.float(), train=True)
        if self.weights.get("ssim", 0) > 0:
            loss_components["loss_ssim"] = ssim_val
            total_loss += self.weights["ssim"] * ssim_val

        # 4. Anatomix Perceptual Loss
        if self.perceptual is not None and self.weights.get("perceptual", 0) > 0:
            perc_val = self.perceptual(pred, target)
            loss_components["loss_perceptual"] = perc_val
            total_loss += self.weights["perceptual"] * perc_val

        # 5. Dice Losses (Teacher-guided)
        if pred_probs is not None and target_mask is not None:
            bone_idx = self.weights.get("dice_bone_idx", 5)
            class_dices, bone_dice = get_class_dice(pred_probs, target_mask, bone_idx=bone_idx)

            # --- General Dice (All classes or Foreground only) ---
            if self.weights.get("dice_exclude_background", True):
                gen_dice = class_dices[1:].mean()
            else:
                gen_dice = class_dices.mean()

            loss_components["dice_score_all"] = gen_dice
            if self.weights.get("dice_w", 0) > 0:
                loss_components["loss_dice"] = 1.0 - gen_dice
                total_loss += self.weights["dice_w"] * (1.0 - gen_dice)

            # --- Bone-Specific Dice ---
            if bone_dice is not None:
                loss_components["dice_score_bone"] = bone_dice
                if self.weights.get("dice_bone_w", 0) > 0:
                    loss_components["loss_dice_bone"] = 1.0 - bone_dice
                    total_loss += self.weights["dice_bone_w"] * (1.0 - bone_dice)

        return total_loss, loss_components
