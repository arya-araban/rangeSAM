import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask=None):
        # Ensure pred is in logit form
        if pred.size(1) > 1:
            pred = pred[:, 1, ...].unsqueeze(1)  # Take the foreground channel for binary case

        pred = torch.sigmoid(pred)  # Use sigmoid instead of softmax for binary case

        # Ensure target has the same shape as pred
        target = target.unsqueeze(1).float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # Add channel dimension to mask
            pred = pred * mask
            target = target * mask

        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return torch.clamp(1 - dice, min=0.0, max=1.0)  # Ensure non-negative loss
