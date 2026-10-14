import os

import numpy as np

from .instance_dataset import InstanceDataset


class ApolloScape(InstanceDataset):
    """ApolloScape detection set: <root>/{detection_train_bin, detection_instance_labels}/<sequence>."""

    def scan_dirs(self, seq):
        return (os.path.join(self.root, 'detection_train_bin', str(seq)),
                os.path.join(self.root, 'detection_instance_labels', str(seq)))

    @staticmethod
    def readPCD(path):
        pointcloud = np.fromfile(path, dtype=np.float32)
        if len(pointcloud) % 4 == 0:
            return pointcloud.reshape(-1, 4)
        if len(pointcloud) % 5 == 0:
            return pointcloud.reshape(-1, 5)[:, :4]
        raise ValueError(f'Unexpected point cloud size in {path}')
