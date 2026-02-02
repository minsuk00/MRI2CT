import torch
import torch.nn as nn
from fused_ssim import fused_ssim3d

class CompositeLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target, feat_extractor=None, use_sliding_window=False):
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
            
        return total_loss, loss_components
