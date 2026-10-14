import numpy as np
import torch


def expanded_bbox(fg_indices, shape, expand=0.2):
    """Bounding box (y_min, y_max, x_min, x_max) of the foreground pixels, enlarged by
    `expand` of its size on each side and clipped to the image."""
    y_min, x_min = fg_indices.min(0)
    y_max, x_max = fg_indices.max(0)
    expand_h, expand_w = int(expand * (y_max - y_min)), int(expand * (x_max - x_min))
    return (max(0, y_min - expand_h), min(shape[0] - 1, y_max + expand_h),
            max(0, x_min - expand_w), min(shape[1] - 1, x_max + expand_w))


def batch_sample_points(labels, fg_range=(1, 12), bg_range=(0, 8)):
    """Random prompt sampling used for training.

    For every range image in the batch, samples between fg_range[0] and fg_range[1]
    foreground clicks on the instance and between bg_range[0] and bg_range[1] background
    clicks inside the instance's bounding box enlarged by 20%.

    labels: (B, H, W) binary instance masks.
    Returns point_coords (B, N, 2) as (row, col) and point_labels (B, N) with
    1 = foreground, 0 = background and -1 = padding.
    """
    if labels.dim() == 2:
        labels = labels.unsqueeze(0)
    B, H, W = labels.shape
    device = labels.device

    max_points = fg_range[1] + bg_range[1]
    point_coords = torch.full((B, max_points, 2), -1, dtype=torch.long, device=device)
    point_labels = torch.full((B, max_points), -1, dtype=torch.long, device=device)

    for b in range(B):
        foreground_mask = (labels[b] == 1)
        fg_indices = torch.nonzero(foreground_mask, as_tuple=False)
        num_fg_available = fg_indices.shape[0]

        num_fg_points = min(torch.randint(fg_range[0], fg_range[1] + 1, (1,)).item(), num_fg_available)
        num_bg_points = torch.randint(bg_range[0], bg_range[1] + 1, (1,)).item()

        if num_fg_available > 0:
            fg_sampled = fg_indices[torch.randperm(num_fg_available)[:num_fg_points]]
            point_coords[b, :num_fg_points] = fg_sampled
            point_labels[b, :num_fg_points] = 1

            y_min, y_max, x_min, x_max = expanded_bbox(fg_indices.cpu().numpy(), (H, W))
            bg_mask = torch.zeros_like(foreground_mask)
            bg_mask[y_min:y_max + 1, x_min:x_max + 1] = True
            bg_mask[foreground_mask] = False
        else:
            bg_mask = ~foreground_mask
            num_bg_points = min(num_fg_points + num_bg_points, max_points)
            num_fg_points = 0

        bg_indices = torch.nonzero(bg_mask, as_tuple=False)
        num_bg_points = min(num_bg_points, bg_indices.shape[0])
        if num_bg_points > 0:
            bg_sampled = bg_indices[torch.randperm(bg_indices.shape[0])[:num_bg_points]]
            point_coords[b, num_fg_points:num_fg_points + num_bg_points] = bg_sampled
            point_labels[b, num_fg_points:num_fg_points + num_bg_points] = 0

    return point_coords, point_labels


def sample_eval_prompts(mask, k, bg_ratio=0.35, invert_labels=False):
    """Prompt sampling used for evaluation: exactly k clicks (when enough pixels exist).

    About bg_ratio of the clicks are background clicks taken from the instance's bounding box
    enlarged by 20% (at least one when k > 1), the rest are foreground clicks on the instance.
    With k = 1 a single foreground click is used. bg_ratio = 0 / 1 gives foreground-only /
    background-only prompts. invert_labels swaps the foreground/background labels of the clicks.

    mask: (H, W) binary instance mask. Returns a list of (row, col, label).
    """
    fg_mask = mask == 1
    fg_points = np.argwhere(fg_mask)
    if len(fg_points) == 0:
        return []

    y_min, y_max, x_min, x_max = expanded_bbox(fg_points, mask.shape)
    bg_mask = np.zeros_like(mask, dtype=bool)
    bg_mask[y_min:y_max + 1, x_min:x_max + 1] = True
    bg_mask[fg_mask] = False
    bg_points = np.argwhere(bg_mask)

    if bg_ratio >= 1:
        num_bg, num_fg = min(len(bg_points), k), 0
    elif k == 1 or bg_ratio <= 0:
        num_bg, num_fg = 0, min(len(fg_points), k)
    else:
        num_bg = min(len(bg_points), max(1, int(k * bg_ratio)))
        num_fg = min(len(fg_points), k - num_bg)

    fg_label, bg_label = (0, 1) if invert_labels else (1, 0)
    clicks = []
    for idx in np.random.choice(len(bg_points), num_bg, replace=False):
        clicks.append((*bg_points[idx], bg_label))
    for idx in np.random.choice(len(fg_points), num_fg, replace=False):
        clicks.append((*fg_points[idx], fg_label))
    return clicks
