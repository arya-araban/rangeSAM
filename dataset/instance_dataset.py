import os
import random

import numpy as np
from scipy import sparse


class InstanceDataset(object):
    """Base class for LiDAR datasets stored as raw scans plus per-instance binary masks.

    Expected layout (produced by the scripts in scripts/data_prep):
        <pointcloud_dir>/<frame>.bin
        <label_dir>/<frame>/<class>_<n>.npz   (one sparse binary mask per object instance)

    Every sample is a (scan, instance) pair: the selected instance is the foreground (1)
    and everything else is background (0).
    """

    mapped_cls_name = {0: 'background', 1: 'foreground'}
    # Approximate fraction of foreground/background points, only used for logging.
    cls_freq = np.array([0.95, 0.05], dtype=np.float32)

    def __init__(self, root, sequences, has_label=True):
        self.root = root
        self.sequences = list(sequences)
        self.has_label = has_label

        if not os.path.isdir(self.root):
            raise ValueError(f'Dataset not found: {self.root}')

        self.pointcloud_files = []
        self.label_dirs = {}
        self.class_instances = {}

        for seq in self.sequences:
            pointcloud_dir, label_dir = self.scan_dirs(seq)
            pc_files = sorted(f for f in os.listdir(pointcloud_dir) if f.endswith('.bin'))

            for pc_file in pc_files:
                pc_path = os.path.join(pointcloud_dir, pc_file)
                if not self.has_label:
                    self.pointcloud_files.append(pc_path)
                    continue

                frame_label_dir = os.path.join(label_dir, os.path.splitext(pc_file)[0])
                if not os.path.isdir(frame_label_dir):
                    continue
                label_files = sorted(f for f in os.listdir(frame_label_dir) if f.endswith('.npz'))
                if not label_files:
                    continue

                self.pointcloud_files.append(pc_path)
                self.label_dirs[pc_path] = frame_label_dir
                instances = {}
                for f in label_files:
                    instances.setdefault(self.get_class_name(f), []).append(f)
                self.class_instances[pc_path] = instances

        print(f'{type(self).__name__}: using {len(self.pointcloud_files)} scans from sequences {self.sequences}')

    def scan_dirs(self, seq):
        """Return (pointcloud_dir, label_dir) for a sequence."""
        raise NotImplementedError

    @staticmethod
    def readPCD(path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    @staticmethod
    def get_class_name(instance_file):
        return instance_file.rsplit('_', 1)[0]

    def labelMapping(self, label):
        return label

    def load_instance(self, index, instance_file):
        pc_path = self.pointcloud_files[index]
        mask = sparse.load_npz(os.path.join(self.label_dirs[pc_path], instance_file))
        label = mask.toarray().ravel().astype(np.int32)
        return label, label

    def loadLabelByIndex(self, index):
        # Sample a class uniformly, then an instance of that class (counters class imbalance).
        instances = self.class_instances[self.pointcloud_files[index]]
        selected_class = random.choice(list(instances.keys()))
        return self.load_instance(index, random.choice(instances[selected_class]))

    def loadDataByIndex(self, index):
        pointcloud = self.readPCD(self.pointcloud_files[index])
        if not self.has_label:
            empty = np.zeros(pointcloud.shape[0], dtype=np.int32)
            return pointcloud, empty, empty
        sem_label, inst_label = self.loadLabelByIndex(index)
        return pointcloud, sem_label, inst_label

    def get_all_instances(self, index):
        """List of (class_name, instance_file) for every instance in a scan."""
        instances = self.class_instances[self.pointcloud_files[index]]
        return [(class_name, f) for class_name, files in instances.items() for f in files]

    def __len__(self):
        return len(self.pointcloud_files)
