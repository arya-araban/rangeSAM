import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


def calculate_iou(pred, target):
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-6)


def calculate_precision_recall_f1(pred, target):
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


def calculate_ap(pred_scores, target, thresholds=(0.25, 0.5)):
    """Average precision of the per-point foreground scores: over the full precision-recall
    curve ('ap') and for the masks obtained at the given score thresholds ('ap@25', 'ap@50')."""
    scores, target = pred_scores.ravel(), target.ravel()
    precision, recall, _ = precision_recall_curve(target, scores)
    results = {'ap': auc(recall, precision)}
    for t in thresholds:
        results[f'ap@{int(t * 100)}'] = average_precision_score(target, (scores > t).astype(int))
    return results


def calculate_sassoc(pred, target):
    """Segmentation association quality S_assoc (Nunes et al., 3DUIS):
    mean over ground-truth instances t of TP(s, t) * IoU(s, t) / |t| for the best matching
    predicted instance s. Labels are instance ids, 0 = background."""
    target_ids = np.unique(target[target != 0])
    pred_ids = np.unique(pred[pred != 0])
    if len(target_ids) == 0:
        return 1.0 if len(pred_ids) == 0 else 0.0

    total = 0.0
    for t in target_ids:
        t_mask = target == t
        best_iou, best_tp = 0.0, 0
        for s in pred_ids:
            s_mask = pred == s
            tp = np.sum(t_mask & s_mask)
            if tp == 0:
                continue
            iou = tp / (np.sum(s_mask) + np.sum(t_mask) - tp + 1e-8)
            if iou > best_iou:
                best_iou, best_tp = iou, tp
        total += best_tp * best_iou / (np.sum(t_mask) + 1e-8)
    return total / len(target_ids)
