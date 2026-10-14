"""Evaluate a trained RangeSAM model with simulated user clicks.

Every object instance of every evaluated scan is segmented with K clicks sampled by
utils.point_sampling.sample_eval_prompts (about 65% foreground / 35% background clicks,
background clicks taken from the instance's bounding box enlarged by 20%). The 2D prediction
is mapped back onto the 3D points with the KNN refiner and compared to the ground truth.

Examples:
    # IoU@K on the KITTI-360 test split
    python evaluate.py configs/kitti360.yaml --data_root <KITTI-360>/data_3d_raw --checkpoint <ckpt> --k 1 3 5 10 20

    # Cross-dataset evaluation of the KITTI-360 model
    python evaluate.py configs/apolloscape.yaml --data_root <apolloscape> --checkpoint <ckpt> --model_config configs/kitti360.yaml

    # Precision / recall / F1 / AP / S_assoc at K = 20
    python evaluate.py configs/semantic_kitti.yaml --data_root <...>/dataset/sequences --checkpoint <ckpt> \
        --model_config configs/kitti360.yaml --k 20 --all_metrics
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
import yaml
from prettytable import PrettyTable
from tqdm import tqdm

from dataset import DATASETS, RangeViewLoader
from models import build_model
from utils.metrics.instance_metrics import calculate_ap, calculate_iou, calculate_precision_recall_f1, calculate_sassoc
from utils.point_sampling import sample_eval_prompts
from utils.postproc import KNN

# Fraction of background clicks for each prompt mode
PROMPT_MODES = {'mixed': 0.35, 'fg_only': 0.0, 'bg_only': 1.0, 'inverted': 0.35}


def is_dense(pointcloud, instance_mask, max_distance=15.0, min_points=50):
    """'Dense' objects: centre within max_distance metres of the sensor and at least min_points points."""
    points = pointcloud[instance_mask, :3]
    if len(points) < min_points:
        return False
    return np.linalg.norm(points.mean(axis=0)[:2]) <= max_distance


def uncrop(values, crop_offset, full_shape, fill):
    """Paste a crop back into a full-size range image filled with `fill`."""
    full = np.full(full_shape, fill, dtype=values.dtype)
    y, x = crop_offset
    full[y:y + values.shape[0], x:x + values.shape[1]] = values
    return full


def refine_3d(pred_2d, projection_info, crop_offset, knn, device):
    """Binary 3D mask from the 2D crop prediction using the KNN refiner.
    Only the range values inside the crop are visible to the refiner."""
    full_shape = projection_info['proj_mask'].shape
    pred = uncrop(pred_2d.astype(np.float32), crop_offset, full_shape, 0)
    crop_range = uncrop(np.ones(pred_2d.shape, dtype=bool), crop_offset, full_shape, False)
    proj_range = np.where(crop_range, projection_info['proj_range'], -1).astype(np.float32)

    pred_3d = knn(torch.from_numpy(proj_range).to(device),
                  torch.from_numpy(projection_info['uproj_depth'].astype(np.float32)).to(device),
                  torch.from_numpy(pred).to(device),
                  torch.from_numpy(projection_info['uproj_x_idx']).long().to(device),
                  torch.from_numpy(projection_info['uproj_y_idx']).long().to(device))
    return pred_3d.cpu().numpy() > 0


def scores_3d(scores_2d, projection_info, crop_offset):
    """Foreground probability of every 3D point, read from its range image pixel (0 outside the crop)."""
    full = uncrop(scores_2d, crop_offset, projection_info['proj_mask'].shape, 0)
    return full[projection_info['uproj_y_idx'], projection_info['uproj_x_idx']]


def predict(model, input2d, clicks, device):
    """Foreground probability of every pixel of the crop."""
    point_coords = torch.tensor([[[y, x] for y, x, _ in clicks]], dtype=torch.long, device=device)
    point_labels = torch.tensor([[label for _, _, label in clicks]], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(input2d.unsqueeze(0).to(device), point_coords, point_labels)
    return torch.sigmoid(output[0, 1]).cpu().numpy()


def evaluate(model, dataset, loader, knn, args, device):
    bg_ratio = PROMPT_MODES[args.prompts]
    invert = args.prompts == 'inverted'

    indices = list(range(len(dataset)))
    if args.num_scans is not None and args.num_scans < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.num_scans))

    # results[metric][k][class_name] -> list of per-instance values
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for index in tqdm(indices):
        pointcloud = dataset.readPCD(dataset.pointcloud_files[index])

        for class_name, instance_file in dataset.get_all_instances(index):
            target_3d = dataset.load_instance(index, instance_file)[0] > 0
            if args.dense and not is_dense(pointcloud, target_3d):
                continue

            crop_data, projection_info = loader.get_instance_crop(pointcloud, target_3d.astype(np.int32))
            target_2d = crop_data['label2d'].numpy() > 0
            if target_2d.sum() < args.min_pixels:
                continue  # instance (almost) not visible in the range image

            for k in args.k:
                clicks = sample_eval_prompts(target_2d.astype(np.uint8), k, bg_ratio=bg_ratio, invert_labels=invert)
                if not clicks:
                    continue

                scores_2d = predict(model, crop_data['input2d'], clicks, device)
                pred_2d = scores_2d > 0.5
                pred_3d = refine_3d(pred_2d, projection_info, crop_data['crop_offset'], knn, device)

                results['iou'][k][class_name].append(calculate_iou(pred_3d, target_3d))
                results['iou_2d'][k][class_name].append(calculate_iou(pred_2d, target_2d))

                if args.all_metrics:
                    precision, recall, f1 = calculate_precision_recall_f1(pred_3d, target_3d)
                    ap = calculate_ap(scores_3d(scores_2d, projection_info, crop_data['crop_offset']), target_3d)
                    values = {'precision': precision, 'recall': recall, 'f1': f1,
                              'ap@25': ap['ap@25'], 'ap@50': ap['ap@50'], 'ap': ap['ap'],
                              'sassoc': calculate_sassoc(pred_3d.astype(np.int32), target_3d.astype(np.int32))}
                    for metric, value in values.items():
                        results[metric][k][class_name].append(value)
    return results


def summarize(results, k_values):
    """Per K and metric: mean over classes (of the per-class means) and mean over all instances."""
    summary = {}
    for metric, per_k in results.items():
        summary[metric] = {}
        for k in k_values:
            per_class = {c: float(np.mean(v)) for c, v in sorted(per_k[k].items())}
            all_values = [x for v in per_k[k].values() for x in v]
            summary[metric][k] = {
                'class_mean': float(np.mean(list(per_class.values()))) if per_class else float('nan'),
                'instance_mean': float(np.mean(all_values)) if all_values else float('nan'),
                'num_instances': len(all_values),
                'per_class': per_class,
            }
    return summary


def print_summary(summary, k_values):
    overall = PrettyTable(['Metric'] + [f'K={k}' for k in k_values])
    for metric, per_k in summary.items():
        overall.add_row([f'{metric} (class mean)'] + [f"{100 * per_k[k]['class_mean']:.2f}" for k in k_values])
        overall.add_row([f'{metric} (instance mean)'] + [f"{100 * per_k[k]['instance_mean']:.2f}" for k in k_values])
    overall.add_row(['instances'] + [summary['iou'][k]['num_instances'] for k in k_values])
    print(overall)

    classes = sorted({c for k in k_values for c in summary['iou'][k]['per_class']})
    per_class = PrettyTable(['Class'] + [f'IoU@{k}' for k in k_values])
    for c in classes:
        per_class.add_row([c] + [f"{100 * summary['iou'][k]['per_class'].get(c, float('nan')):.2f}" for k in k_values])
    print(per_class)


def main():
    parser = argparse.ArgumentParser(description='Evaluate RangeSAM with simulated clicks')
    parser.add_argument('config', type=str, help='config of the evaluated dataset (configs/*.yaml)')
    parser.add_argument('--data_root', type=str, required=True, help='dataset root directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='trained RangeSAM checkpoint')
    parser.add_argument('--model_config', type=str, default=None,
                        help='config the model was trained with (defaults to config)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--sequences', type=str, nargs='+', default=None,
                        help='evaluate these sequences instead of the split from the config')
    parser.add_argument('--k', type=int, nargs='+', default=[1, 3, 5, 10, 20], help='numbers of clicks')
    parser.add_argument('--prompts', type=str, default='mixed', choices=list(PROMPT_MODES),
                        help='mixed (65/35 fg/bg), fg_only, bg_only or inverted (fg/bg labels swapped)')
    parser.add_argument('--dense', action='store_true',
                        help="only 'dense' objects: centre within 15 m and at least 50 points")
    parser.add_argument('--all_metrics', action='store_true',
                        help='also compute precision, recall, F1, AP@25, AP@50, AP and S_assoc')
    parser.add_argument('--num_scans', type=int, default=None, help='evaluate a random subset of scans')
    parser.add_argument('--min_pixels', type=int, default=5,
                        help='skip instances with fewer visible pixels in the range image crop')
    parser.add_argument('--seed', type=int, default=0, help='seed for the scan subset and the clicks')
    parser.add_argument('--output', type=str, default=None, help='save the results as JSON')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config, 'r') as f:
        data_config = yaml.safe_load(f)
    with open(args.model_config or args.config, 'r') as f:
        model_config = yaml.safe_load(f)

    model = build_model(model_config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint.get('model', checkpoint))
    model.to(device).eval()

    sequences = args.sequences or data_config['split'][args.split]
    dataset = DATASETS[data_config['dataset']](root=args.data_root, sequences=sequences)
    loader = RangeViewLoader(dataset, data_config, is_train=False)
    knn = KNN()  # k = 1 nearest neighbour in a 3 x 3 window

    results = evaluate(model, dataset, loader, knn, args, device)
    summary = summarize(results, args.k)
    print_summary(summary, args.k)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'args': vars(args), 'results': summary}, f, indent=2)
        print(f'Results saved to {args.output}')


if __name__ == '__main__':
    main()
