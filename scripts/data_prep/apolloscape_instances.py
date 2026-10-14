"""Create per-instance binary masks for the ApolloScape detection set.

ApolloScape annotates objects with 3D boxes; every point inside a box becomes part of that instance.

Input: <root>/detection_train_bin/<sequence>/<frame>.bin and
       <root>/detection_train_label/<sequence>/<frame>.txt
       (one object per line: type x y z length width height heading).

Output: <root>/detection_instance_labels/<sequence>/<frame>/<class>_<n>.npz, the layout
dataset/apolloscape.py expects.

    python scripts/data_prep/apolloscape_instances.py --root <ApolloScape>
"""

import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from tqdm import tqdm

OBJECT_TYPES = {1: 'small_vehicle', 2: 'big_vehicle', 3: 'pedestrian', 4: 'cyclist', 5: 'traffic_cone', 6: 'other'}


def load_pointcloud(path):
    pointcloud = np.fromfile(path, dtype=np.float32)
    if len(pointcloud) % 4 == 0:
        return pointcloud.reshape(-1, 4)
    if len(pointcloud) % 5 == 0:
        return pointcloud.reshape(-1, 5)[:, :4]
    raise ValueError(f'Unexpected point cloud size in {path}')


def points_in_box(points, x, y, z, length, width, height, heading):
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
    local = (points[:, :3] - [x, y, z]) @ rotation.T
    return ((np.abs(local[:, 0]) <= length / 2) &
            (np.abs(local[:, 1]) <= width / 2) &
            (np.abs(local[:, 2]) <= height / 2))


def process_frame(frame, scan_dir, label_dir, output_dir):
    label_path = os.path.join(label_dir, f'{frame}.txt')
    if not os.path.isfile(label_path):
        return
    points = load_pointcloud(os.path.join(scan_dir, f'{frame}.bin'))
    with open(label_path, 'r') as f:
        boxes = [list(map(float, line.split())) for line in f if line.strip()]

    frame_dir = os.path.join(output_dir, frame)
    count = defaultdict(int)
    for object_type, *box in boxes:
        mask = points_in_box(points, *box)
        if not mask.any():
            continue
        class_name = OBJECT_TYPES.get(int(object_type), 'unknown')
        count[class_name] += 1
        os.makedirs(frame_dir, exist_ok=True)
        sparse.save_npz(os.path.join(frame_dir, f'{class_name}_{count[class_name]}.npz'), sparse.csr_matrix(mask))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', required=True, help='ApolloScape directory with detection_train_bin/ and detection_train_label/')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    scan_root = os.path.join(args.root, 'detection_train_bin')
    for sequence in sorted(os.listdir(scan_root)):
        scan_dir = os.path.join(scan_root, sequence)
        if not os.path.isdir(scan_dir):
            continue
        label_dir = os.path.join(args.root, 'detection_train_label', sequence)
        output_dir = os.path.join(args.root, 'detection_instance_labels', sequence)
        frames = sorted(os.path.splitext(f)[0] for f in os.listdir(scan_dir) if f.endswith('.bin'))

        with Pool(args.num_workers) as pool:
            work = partial(process_frame, scan_dir=scan_dir, label_dir=label_dir, output_dir=output_dir)
            list(tqdm(pool.imap_unordered(work, frames), total=len(frames), desc=sequence))


if __name__ == '__main__':
    main()
