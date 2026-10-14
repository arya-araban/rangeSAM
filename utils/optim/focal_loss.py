import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        # Ensure pred is in logit form
        if pred.size(1) > 1:
            pred = pred[:, 1, ...].unsqueeze(1)  # Take the foreground channel for binary case

        # Ensure target has the same shape as pred
        target = target.unsqueeze(1).float()

        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        pt = torch.exp(-bce_loss)

        # Apply alpha weighting based on the target class
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if mask is not None:
            mask = mask.unsqueeze(1)  # Add channel dimension to mask
            focal_loss = focal_loss * mask

        return focal_loss.mean()
