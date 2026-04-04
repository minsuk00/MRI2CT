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
    probs = F.softmax(logits, dim=1)  # [B, C, X, Y, Z]
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


class CompositeLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target, pred_probs=None, target_mask=None):
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

        # 4. Dice Losses (Teacher-guided)
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
