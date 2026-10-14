import os

from .instance_dataset import InstanceDataset


class KITTI360(InstanceDataset):
    """KITTI-360 raw scans: <root>/2013_05_28_drive_XXXX_sync/{velodyne_points/data, instance_labels}."""

    def scan_dirs(self, seq):
        seq_dir = os.path.join(self.root, f'2013_05_28_drive_{int(seq):04d}_sync')
        return os.path.join(seq_dir, 'velodyne_points', 'data'), os.path.join(seq_dir, 'instance_labels')
