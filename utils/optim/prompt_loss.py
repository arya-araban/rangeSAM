import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptLoss(nn.Module):
    """Binary cross-entropy restricted to a kernel_size x kernel_size window around each
    prompt, averaged over the prompts. Encourages accurate predictions near the clicks."""

    def __init__(self, kernel_size=3):
        super(PromptLoss, self).__init__()
        self.kernel_size = kernel_size
        self.offset = kernel_size // 2

    def forward(self, pred, target, point_coords, point_labels):
        B, C, H, W = pred.shape

        # Ensure pred is in logit form for binary classification
        if C > 1:
            pred = pred[:, 1, ...].unsqueeze(1)  # Take the foreground channel for binary case

        point_loss = torch.zeros(B, device=pred.device)

        for b in range(B):
            coords = point_coords[b]
            labels = point_labels[b]

            valid_mask = labels != -1
            valid_coords = coords[valid_mask]
            valid_labels = labels[valid_mask]

            for coord, label in zip(valid_coords, valid_labels):
                y, x = coord
                y, x = int(y), int(x)
                if not (0 <= y < H and 0 <= x < W):
                    continue  # Skip if coordinate is out of bounds

                y_start = max(0, y - self.offset)
                y_end = min(H, y + self.offset + 1)
                x_start = max(0, x - self.offset)
                x_end = min(W, x + self.offset + 1)

                area_pred = pred[b, 0, y_start:y_end, x_start:x_end].reshape(-1)
                area_target = target[b, y_start:y_end, x_start:x_end].reshape(-1)

                point_loss[b] += F.binary_cross_entropy_with_logits(area_pred, area_target.float())

        num_valid_points = (point_labels != -1).sum().float()
        return point_loss.sum() / (num_valid_points + 1e-8)
