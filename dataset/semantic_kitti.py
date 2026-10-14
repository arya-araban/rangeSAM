import os

from .instance_dataset import InstanceDataset


class SemanticKitti(InstanceDataset):
    """SemanticKITTI: <root>/XX/{velodyne, instance_labels}, with <root> = .../dataset/sequences."""

    def scan_dirs(self, seq):
        seq_dir = os.path.join(self.root, f'{int(seq):02d}')
        return os.path.join(seq_dir, 'velodyne'), os.path.join(seq_dir, 'instance_labels')
