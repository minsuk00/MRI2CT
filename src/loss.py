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

    def soft_dice_loss(self, probs, target_mask):
        """
        probs: [B, C, X, Y, Z] (Softmax output from Teacher)
        target_mask: [B, 1, X, Y, Z] (Integer Hard Labels from GT)
        """
        smooth = 1e-5
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
        dice = (2. * intersection + smooth) / (union + smooth)
        
        # Average over classes and batch
        return 1.0 - dice.mean()

    def forward(self, pred, target, feat_extractor=None, use_sliding_window=False, pred_probs=None, target_mask=None):
        total_loss = 0.0
        loss_components = {}
        
        if self.weights.get("l1", 0) > 0:
            val = self.l1(pred, target)
            total_loss += self.weights["l1"] * val
            loss_components["loss_l1"] = val.item()
            
        if self.weights.get("l2", 0) > 0:
            val = self.l2(pred, target)
            total_loss += self.weights["l2"] * val
            loss_components["loss_l2"] = val.item()

        if self.weights.get("ssim", 0) > 0:
            # b, c, d, h, w = pred.shape
            # pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).float()
            # targ_2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).float()
            # val = 1.0 - fused_ssim(pred_2d, targ_2d, train=True)
            
            val = 1.0 - fused_ssim3d(pred.float(), target.float(), train=True)
            total_loss += self.weights["ssim"] * val
            loss_components["loss_ssim"] = val.item()

        if self.weights.get("perceptual", 0) > 0:
            if feat_extractor is None: 
                raise ValueError("Feat extractor missing for perceptual loss")
            if use_sliding_window:
                print("Skipping perceptual loss calculation during validation. NOTE: val loss will differ from train loss.")
            else:
                pred_feats = feat_extractor(pred)
                with torch.no_grad(): target_feats = feat_extractor(target)
                val = self.l1(pred_feats, target_feats)
                total_loss += self.weights["perceptual"] * val
                loss_components["loss_perceptual"] = val.item()
        
        if self.weights.get("dice_w", 0) > 0 and pred_probs is not None and target_mask is not None:
            val = self.soft_dice_loss(pred_probs, target_mask)
            total_loss += self.weights["dice_w"] * val
            loss_components["loss_dice"] = val.item()
            
        return total_loss, loss_components
