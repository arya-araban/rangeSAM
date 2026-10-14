# Copyright 2022 - Valeo Comfort and Driving Assistance
# Modifications copyright 2024 - Arya Araban
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .preprocess import augmentor, projection


class RangeViewLoader(Dataset):
    """Projects (scan, instance) pairs to range images and crops them around the instance.

    Each item is a 5-channel range image crop (range, x, y, z, intensity), the binary
    instance mask of the crop and the valid-pixel mask.
    """

    def __init__(self, dataset, config, is_train=True):
        self.dataset = dataset
        self.config = config
        self.is_train = is_train

        # Point cloud augmentations (training only)
        self.augmentor = None
        if self.is_train:
            aug = self.config['augmentation']
            params = augmentor.AugmentParams()
            params.setFlipProb(p_flipx=aug['p_flipx'], p_flipy=aug['p_flipy'])
            params.setTranslationParams(
                p_transx=aug['p_transx'], trans_xmin=aug['trans_xmin'], trans_xmax=aug['trans_xmax'],
                p_transy=aug['p_transy'], trans_ymin=aug['trans_ymin'], trans_ymax=aug['trans_ymax'],
                p_transz=aug['p_transz'], trans_zmin=aug['trans_zmin'], trans_zmax=aug['trans_zmax'])
            params.setRotationParams(
                p_rot_roll=aug['p_rot_roll'], rot_rollmin=aug['rot_rollmin'], rot_rollmax=aug['rot_rollmax'],
                p_rot_pitch=aug['p_rot_pitch'], rot_pitchmin=aug['rot_pitchmin'], rot_pitchmax=aug['rot_pitchmax'],
                p_rot_yaw=aug['p_rot_yaw'], rot_yawmin=aug['rot_yawmin'], rot_yawmax=aug['rot_yawmax'])
            self.augmentor = augmentor.Augmentor(params)

        sensor = self.config['sensor']
        if sensor.get('scan_proj', False):
            self.projection = projection.ScanProjection(proj_h=sensor['proj_h'], proj_w=sensor['proj_w'])
        else:
            self.projection = projection.RangeProjection(
                fov_up=sensor['fov_up'], fov_down=sensor['fov_down'],
                fov_left=sensor['fov_left'], fov_right=sensor['fov_right'],
                proj_h=sensor['proj_h'], proj_w=sensor['proj_w'])
        self.proj_img_mean = torch.tensor(sensor['img_mean'], dtype=torch.float)
        self.proj_img_stds = torch.tensor(sensor['img_stds'], dtype=torch.float)

        self.crop_size = self.config['image_size']

    def _project(self, pointcloud, instance_mask):
        """Range-project a scan and its instance mask into a (7, H, W) tensor:
        5 normalized feature channels, the instance mask and the valid-pixel mask."""
        if self.augmentor is not None:
            augmented = self.augmentor.doAugmentation(pointcloud.copy())
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(
                augmented, ring_source=pointcloud)
        else:
            proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)

        proj_mask_tensor = torch.from_numpy(proj_mask)
        valid = proj_idx > 0
        proj_label = np.zeros(proj_mask.shape, dtype=np.float32)
        proj_label[valid] = instance_mask[proj_idx[valid]]
        proj_label_tensor = torch.from_numpy(proj_label) * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        proj_tensor = torch.cat(
            (proj_feature_tensor, proj_label_tensor.unsqueeze(0), proj_mask_tensor.float().unsqueeze(0)), dim=0)
        return proj_tensor, proj_range, proj_idx, proj_mask

    def __getitem__(self, index):
        pointcloud, sem_label, _ = self.dataset.loadDataByIndex(index)
        instance_mask = self.dataset.labelMapping(sem_label)

        proj_tensor, _, _, _ = self._project(pointcloud, instance_mask)
        proj_tensor, _ = crop_around_instance(proj_tensor, self.crop_size)
        return proj_tensor[:5], proj_tensor[5], proj_tensor[6]

    def __len__(self):
        return len(self.dataset)

    def get_instance_crop(self, pointcloud, instance_mask):
        """Crop used at evaluation time, plus the projection info needed to map
        2D predictions back onto the 3D points."""
        proj_tensor, proj_range, proj_idx, proj_mask = self._project(pointcloud, instance_mask)
        proj_tensor, crop_offset = crop_around_instance(proj_tensor, self.crop_size)

        crop_data = {
            'input2d': proj_tensor[:5],
            'label2d': proj_tensor[5],
            'mask2d': proj_tensor[6],
            'crop_offset': crop_offset,
        }
        projection_info = {
            'proj_range': proj_range,
            'proj_idx': proj_idx,
            'proj_mask': proj_mask,
            'uproj_x_idx': self.projection.cached_data['uproj_x_idx'],
            'uproj_y_idx': self.projection.cached_data['uproj_y_idx'],
            'uproj_depth': self.projection.cached_data['uproj_depth'],
        }
        return crop_data, projection_info


def find_foreground_bbox(mask):
    """Bounding box (min_y, max_y, min_x, max_x) of the foreground pixels, or None."""
    y, x = np.where(mask)
    if len(y) == 0:
        return None
    return np.min(y), np.max(y), np.min(x), np.max(x)


def crop_around_instance(proj_tensor, crop_size):
    """Crop the range image to crop_size, centred on the selected instance.

    The crop is shifted to stay inside the image when the instance is near a border;
    without a foreground instance it falls back to a centre crop.
    """
    _, h, w = proj_tensor.shape
    assert h == crop_size[0]

    bbox = find_foreground_bbox(proj_tensor[5].numpy() > 0)
    if bbox is None:
        offset_y = 0
        offset_x = (w - crop_size[1]) // 2
    else:
        min_y, max_y, min_x, max_x = bbox
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        offset_y = max(0, min(center_y - crop_size[0] // 2, h - crop_size[0]))
        offset_x = max(0, min(center_x - crop_size[1] // 2, w - crop_size[1]))

    proj_tensor = TF.crop(proj_tensor, offset_y, offset_x, crop_size[0], crop_size[1])
    return proj_tensor, [offset_y, offset_x]
