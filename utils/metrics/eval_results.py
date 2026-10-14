'''
Adapted from Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

from prettytable import PrettyTable


def eval_results(recorder, metrics_dict, class_names, mode):
    """Log per-class pixel-wise (2D) IoU / accuracy / recall and the label distribution."""
    class_acc = metrics_dict['class_acc']
    class_recall = metrics_dict['class_recall']
    class_iou = metrics_dict['class_iou']

    recorder.logger.info(f'============ {mode}: pixel-wise evaluation results ============')
    recorder.logger.info('Acc avg: {:.4f}, IOU avg: {:.4f}, Recall avg: {:.4f}'.format(
        metrics_dict['mean_acc'].item(), metrics_dict['mean_iou'].item(), metrics_dict['mean_recall'].item()))

    cls_eval_table = PrettyTable(['Class ID', 'Class Name', 'IOU', 'Acc', 'Recall'])
    for i, iou in enumerate(class_iou.cpu()):
        cls_eval_table.add_row([i, class_names[i], iou.item(), class_acc[i].cpu().item(), class_recall[i].cpu().item()])
    recorder.logger.info(cls_eval_table)

    distribution_table = PrettyTable(['Class Name', 'Number of Pixels', 'Percentage'])
    dist_data = metrics_dict['conf_matrix'].sum(0)
    for i in range(len(class_names)):
        distribution_table.add_row([class_names[i], dist_data[i].item(),
                                    '{:.4f}%'.format((dist_data[i] / dist_data.sum()).item() * 100)])
    recorder.logger.info(distribution_table)
