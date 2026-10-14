"""Create per-instance binary masks for the raw KITTI-360 scans.

Input: the per-scan labels recovered with third_party/recoverKITTI360label
(<labels_root>/<sequence>/labels/<frame>.bin holds the semantic id of every point,
<labels_root>/<sequence>/labels/semantics/<frame>.bin its instance id).

Output: <output_root>/<sequence>/instance_labels/<frame>/<class>_<n>.npz, one sparse binary mask
per object instance of the 21 KITTI-360 instance classes. By default the masks are written next
to velodyne_points/ in data_3d_raw, which is the layout dataset/kitti360.py expects.

    python scripts/data_prep/kitti360_instances.py --raw_root <KITTI-360>/data_3d_raw \
        --labels_root <recovered labels>
"""

import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from tqdm import tqdm

# KITTI-360 classes with instance annotations
INSTANCE_CLASSES = {
    11: 'building', 17: 'pole', 19: 'traffic light', 20: 'traffic sign', 24: 'person', 25: 'rider',
    26: 'car', 27: 'truck', 28: 'bus', 29: 'caravan', 30: 'trailer', 31: 'train', 32: 'motorcycle',
    33: 'bicycle', 34: 'garage', 36: 'stop', 37: 'smallpole', 38: 'lamp', 39: 'trash bin',
    40: 'vending machine', 41: 'box',
}

SEQUENCES = [0, 2, 3, 4, 5, 6, 7, 9, 10]


def process_frame(frame, label_dir, output_dir):
    semantic = np.fromfile(os.path.join(label_dir, f'{frame}.bin'), dtype=np.int16)
    instance = np.fromfile(os.path.join(label_dir, 'semantics', f'{frame}.bin'), dtype=np.uint16)

    is_thing = np.isin(semantic, list(INSTANCE_CLASSES)) & (instance != 0)
    instance_ids = np.unique(instance[is_thing])
    if len(instance_ids) == 0:
        return

    frame_dir = os.path.join(output_dir, frame)
    os.makedirs(frame_dir, exist_ok=True)
    count = defaultdict(int)
    for instance_id in instance_ids:
        mask = (instance == instance_id) & is_thing
        class_name = INSTANCE_CLASSES[int(np.bincount(semantic[mask]).argmax())]
        count[class_name] += 1
        sparse.save_npz(os.path.join(frame_dir, f'{class_name}_{count[class_name]}.npz'), sparse.csr_matrix(mask))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--raw_root', required=True, help='KITTI-360 data_3d_raw directory')
    parser.add_argument('--labels_root', required=True, help='output directory of recoverLabels.py')
    parser.add_argument('--output_root', default=None, help='defaults to --raw_root')
    parser.add_argument('--sequences', type=int, nargs='+', default=SEQUENCES)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    output_root = args.output_root or args.raw_root

    for seq in args.sequences:
        sequence = f'2013_05_28_drive_{seq:04d}_sync'
        label_dir = os.path.join(args.labels_root, sequence, 'labels')
        output_dir = os.path.join(output_root, sequence, 'instance_labels')
        frames = sorted(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.bin')
                        and os.path.isfile(os.path.join(label_dir, 'semantics', f)))

        with Pool(args.num_workers) as pool:
            work = partial(process_frame, label_dir=label_dir, output_dir=output_dir)
            list(tqdm(pool.imap_unordered(work, frames), total=len(frames), desc=sequence))


if __name__ == '__main__':
    main()
