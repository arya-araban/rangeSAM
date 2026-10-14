"""Create per-instance binary masks for SemanticKITTI.

Input: <root>/<sequence>/labels/<frame>.label (lower 16 bits: semantic id, upper 16 bits: instance id).

Output: <output_root>/<sequence>/instance_labels/<frame>/<class>_<n>.npz, one sparse binary mask
per object instance of the 8 SemanticKITTI instance classes. By default the masks are written
next to velodyne/, which is the layout dataset/semantic_kitti.py expects.

    python scripts/data_prep/semantic_kitti_instances.py --root <SemanticKITTI>/dataset/sequences
"""

import argparse
import os
from collections import defaultdict
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import sparse
from tqdm import tqdm

# Raw SemanticKITTI ids of the instance classes (moving objects merged with their static class)
INSTANCE_CLASSES = {
    10: 'car', 252: 'car',
    11: 'bicycle',
    15: 'motorcycle',
    18: 'truck', 258: 'truck',
    13: 'other-vehicle', 16: 'other-vehicle', 20: 'other-vehicle',
    256: 'other-vehicle', 257: 'other-vehicle', 259: 'other-vehicle',
    30: 'person', 254: 'person',
    31: 'bicyclist', 253: 'bicyclist',
    32: 'motorcyclist', 255: 'motorcyclist',
}

SEQUENCES = list(range(11))  # sequences with labels


def process_frame(frame, label_dir, output_dir):
    label = np.fromfile(os.path.join(label_dir, f'{frame}.label'), dtype=np.uint32)
    semantic = label & 0xFFFF
    instance = label >> 16

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
    parser.add_argument('--root', required=True, help='SemanticKITTI dataset/sequences directory')
    parser.add_argument('--output_root', default=None, help='defaults to --root')
    parser.add_argument('--sequences', type=int, nargs='+', default=SEQUENCES)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    output_root = args.output_root or args.root

    for seq in args.sequences:
        sequence = f'{seq:02d}'
        label_dir = os.path.join(args.root, sequence, 'labels')
        output_dir = os.path.join(output_root, sequence, 'instance_labels')
        frames = sorted(os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.label'))

        with Pool(args.num_workers) as pool:
            work = partial(process_frame, label_dir=label_dir, output_dir=output_dir)
            list(tqdm(pool.imap_unordered(work, frames), total=len(frames), desc=f'sequence {sequence}'))


if __name__ == '__main__':
    main()
