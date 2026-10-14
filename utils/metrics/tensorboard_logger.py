'''
Adapted from Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

def tensorboard_logger(epoch,
                       mode,
                       recorder,
                       metrics_dict,
                       loss_dict,
                       lr,
                       mapped_cls_name):

    # Metrics
    mean_acc, class_acc = metrics_dict['mean_acc'], metrics_dict['class_acc']
    mean_recall, class_recall = metrics_dict['mean_recall'], metrics_dict['class_recall']
    mean_iou, class_iou = metrics_dict['mean_iou'], metrics_dict['class_iou']

    # Losses
    loss_meter_avg = loss_dict['loss_meter_avg']
    loss_focal = loss_dict['loss_focal']
    loss_dice = loss_dict['loss_dice']
    loss_prompt = loss_dict['loss_prompt']

    recorder.tensorboard.add_scalar(
        tag='{}_Loss'.format(mode), scalar_value=loss_meter_avg, global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossFocal'.format(mode), scalar_value=loss_focal.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossDice'.format(mode), scalar_value=loss_dice.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_LossPrompt'.format(mode), scalar_value=loss_prompt.item(), global_step=epoch)

    recorder.tensorboard.add_scalar(
        tag='{}_meanAcc'.format(mode), scalar_value=mean_acc.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanIOU'.format(mode), scalar_value=mean_iou.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_meanRecall'.format(mode), scalar_value=mean_recall.item(), global_step=epoch)
    recorder.tensorboard.add_scalar(
        tag='{}_lr'.format(mode), scalar_value=lr, global_step=epoch)

    for i, (_, v) in enumerate(mapped_cls_name.items()):
        recorder.tensorboard.add_scalar(
            tag='{}_{:02d}_{}_Acc'.format(mode, i, v), scalar_value=class_acc[i].item(), global_step=epoch)
        recorder.tensorboard.add_scalar(
            tag='{}_{:02d}_{}_Recall'.format(mode, i, v), scalar_value=class_recall[i].item(),
            global_step=epoch)
        recorder.tensorboard.add_scalar(
            tag='{}_{:02d}_{}_IOU'.format(mode, i, v), scalar_value=class_iou[i].item(), global_step=epoch)
