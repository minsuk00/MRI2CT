import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim3d


class CompositeLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def soft_dice_loss(self, logits, target_mask):
        """
        logits: [B, C, X, Y, Z] (Raw logits from Teacher)
        target_mask: [B, 1, X, Y, Z] (Integer Hard Labels from GT)
        """
        smooth = 1e-5
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]

        # Squeeze channel dim from mask if present: [B, 1, X, Y, Z] -> [B, X, Y, Z]
        if target_mask.ndim == 5:
            target_mask = target_mask.squeeze(1)

        # Convert target_mask to one-hot: [B, X, Y, Z] -> [B, C, X, Y, Z]
        target_one_hot = F.one_hot(target_mask.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        # Flatten: [B, C, N]
        p = probs.view(probs.shape[0], probs.shape[1], -1)
        t = target_one_hot.view(target_one_hot.shape[0], target_one_hot.shape[1], -1)

        # Select Classes based on config
        if self.weights.get("dice_bone_only", False):
            if p.shape[1] <= 5:
                raise RuntimeError(f"Dice Bone Only requested (idx 5), but input has only {p.shape[1]} channels.")
            # Select just channel 5
            p = p[:, 5:6, :]
            t = t[:, 5:6, :]
        elif self.weights.get("dice_exclude_background", True):
            # Exclude index 0

            p = p[:, 1:, :]
            t = t[:, 1:, :]

        intersection = (p * t).sum(dim=2)
        union = p.sum(dim=2) + t.sum(dim=2)

        # Dice per class per batch
        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Average over classes and batch
        return 1.0 - dice.mean()

    def forward(self, pred, target, pred_probs=None, target_mask=None):
        total_loss = torch.tensor(0.0, device=pred.device)
        loss_components = {}

        # 1. L1 Loss
        l1_val = self.l1(pred, target)
        loss_components["loss_l1"] = l1_val
        if self.weights.get("l1", 0) > 0:
            total_loss += self.weights["l1"] * l1_val

        # 2. L2 Loss
        if self.weights.get("l2", 0) > 0 or "loss_l2" in self.weights:  # Keep if weight exists
            l2_val = self.l2(pred, target)
            loss_components["loss_l2"] = l2_val
            if self.weights.get("l2", 0) > 0:
                total_loss += self.weights["l2"] * l2_val

        # 3. SSIM Loss
        ssim_val = 1.0 - fused_ssim3d(pred.float(), target.float(), train=True)
        loss_components["loss_ssim"] = ssim_val
        if self.weights.get("ssim", 0) > 0:
            total_loss += self.weights["ssim"] * ssim_val

        # 4. Dice Loss (Only if probabilities are provided)
        if pred_probs is not None and target_mask is not None:
            dice_loss_val = self.soft_dice_loss(pred_probs, target_mask)
            loss_components["loss_dice"] = dice_loss_val
            if self.weights.get("dice_w", 0) > 0:
                total_loss += self.weights["dice_w"] * dice_loss_val

        return total_loss, loss_components
