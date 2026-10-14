'''
The MIT License
Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB, Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)
Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

References:
https://github.com/PRBonn/lidar-bonnetal
https://github.com/TiagoCortinhal/SalsaNext

Adapted to binary (foreground / background) masks for RangeSAM.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(kernel_size, kernel_size)

    return gaussian_kernel


class KNN(nn.Module):
    """KNN-based 3D refinement: transfers the 2D range-image mask back onto every 3D point.

    For each point, the k nearest neighbours (in range) are searched in a search x search
    window around the point's pixel, with distances weighted by an inverted Gaussian kernel
    of std sigma. The point is foreground if the average neighbour label exceeds 0.5.
    """

    def __init__(self, knn=1, search=3, sigma=0.5):
        super().__init__()
        self.knn = knn
        self.search = search
        self.sigma = sigma

    def forward(self, proj_range, unproj_range, proj_argmax, px, py):
        """
        proj_range: (H, W) range image (-1 for empty pixels).
        unproj_range: (P,) range of every 3D point.
        proj_argmax: (H, W) binary 2D prediction.
        px, py: (P,) pixel column / row of every 3D point.
        Returns a (P,) binary prediction.
        """
        device = proj_range.device
        H, W = proj_range.shape
        P = unproj_range.shape[0]

        pad = int((self.search - 1) / 2)

        proj_unfold_k_rang = F.unfold(proj_range[None, None, ...],
                                      kernel_size=(self.search, self.search),
                                      padding=(pad, pad))

        idx_list = py * W + px
        unproj_unfold_k_rang = proj_unfold_k_rang[:, :, idx_list]

        unproj_unfold_k_rang[unproj_unfold_k_rang < 0] = float('inf')

        center = int(((self.search * self.search) - 1) / 2)
        unproj_unfold_k_rang[:, center, :] = unproj_range

        k2_distances = torch.abs(unproj_unfold_k_rang - unproj_range)

        inv_gauss_k = (1 - get_gaussian_kernel(self.search, self.sigma, 1)).view(1, -1, 1)
        inv_gauss_k = inv_gauss_k.to(device).type(proj_range.type())

        k2_distances = k2_distances * inv_gauss_k

        _, knn_idx = k2_distances.topk(self.knn, dim=1, largest=False, sorted=False)

        proj_unfold_1_argmax = F.unfold(proj_argmax[None, None, ...].float(),
                                        kernel_size=(self.search, self.search),
                                        padding=(pad, pad)).long()
        unproj_unfold_1_argmax = proj_unfold_1_argmax[:, :, idx_list]

        knn_argmax = torch.gather(input=unproj_unfold_1_argmax, dim=1, index=knn_idx)

        # Average the neighbour labels and threshold
        knn_pred = knn_argmax.float().mean(dim=1)
        return (knn_pred > 0.5).long().view(P)
