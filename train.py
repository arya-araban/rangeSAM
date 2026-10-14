# Copyright 2023 - Valeo Comfort and Driving Assistance
# Modifications copyright 2024 - Arya Araban
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import time

import torch
import torch.nn as nn

import dataset
import utils.tools as tools
from option import Option
from utils.metrics import IOUEval, eval_results, tensorboard_logger
from utils.optim import FocalLoss, DiceLoss, PromptLoss, WarmupCosineLR
from utils.point_sampling import batch_sample_points


class Trainer(object):
    def __init__(self, settings: Option, model: nn.Module, recorder=None):
        self.settings = settings
        self.recorder = recorder
        self.model = model.cuda()
        self.remain_time = tools.RemainTime(self.settings.n_epochs)

        self.train_loader, self.val_loader, self.train_sampler = self._initDataloader()

        self.criterion = {
            'focal': FocalLoss(gamma=2, alpha=0.95).cuda(),  # alpha weights the (rare) foreground points
            'dice': DiceLoss().cuda(),
            'prompt': PromptLoss(kernel_size=3).cuda(),
        }

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                           lr=self.settings.lr,
                                           weight_decay=self.settings.weight_decay)

        if tools.is_dist_avail_and_initialized():
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda()
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.settings.gpu], find_unused_parameters=True)

        self.metrics = IOUEval(n_classes=self.settings.n_classes, device=torch.device('cpu'),
                               ignore=[], is_distributed=self.settings.distributed)

        # Linear warm-up followed by cosine decay, stepped every iteration
        self.scheduler = WarmupCosineLR(
            optimizer=self.optimizer,
            lr=self.settings.lr,
            warmup_steps=self.settings.warmup_epochs * len(self.train_loader),
            momentum=0.9,
            max_steps=len(self.train_loader) * (self.settings.n_epochs - self.settings.warmup_epochs))

        self.fp16_scaler = torch.cuda.amp.GradScaler() if self.settings.use_fp16 else None

    def _initDataloader(self):
        if self.settings.dataset not in dataset.DATASETS:
            raise ValueError('invalid dataset: {}'.format(self.settings.dataset))
        dataset_cls = dataset.DATASETS[self.settings.dataset]

        split = self.settings.split
        val_sequences = split['test'] if self.settings.test_split else split['valid']
        trainset = dataset_cls(root=self.settings.data_root, sequences=split['train'])
        valset = dataset_cls(root=self.settings.data_root, sequences=val_sequences) if val_sequences else None

        self.train_range_loader = dataset.RangeViewLoader(trainset, self.settings.config, is_train=True)
        self.val_range_loader = (dataset.RangeViewLoader(valset, self.settings.config, is_train=False)
                                 if valset is not None else None)

        train_sampler = None
        if tools.is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_range_loader)
        train_loader = torch.utils.data.DataLoader(
            self.train_range_loader,
            batch_size=self.settings.batch_size,
            num_workers=self.settings.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True)

        val_loader = None
        if self.val_range_loader is not None:
            val_loader = torch.utils.data.DataLoader(
                self.val_range_loader,
                batch_size=1,
                num_workers=self.settings.num_workers,
                shuffle=False,
                drop_last=False)

        return train_loader, val_loader, train_sampler

    def compute_losses(self, output, label, point_coords, point_labels):
        output = output[:, 1, ...].unsqueeze(1)  # foreground logit

        loss_focal = self.criterion['focal'](output, label)
        loss_dice = self.criterion['dice'](output, label)
        if self.settings.use_prompt_loss:
            loss_prompt = self.criterion['prompt'](output, label, point_coords, point_labels)
        else:
            loss_prompt = torch.zeros((), device=output.device)

        total_loss = (self.settings.focal_weight * loss_focal
                      + self.settings.dice_weight * loss_dice
                      + self.settings.prompt_weight * loss_prompt)
        return total_loss, loss_focal, loss_dice, loss_prompt

    def run(self, epoch, mode='Train'):
        if mode == 'Train':
            dataloader = self.train_loader
            self.model.train()
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
        elif mode == 'Validation':
            dataloader = self.val_loader
            self.model.eval()
        else:
            raise ValueError('invalid mode: {}'.format(mode))

        loss_meter = tools.AverageMeter()
        self.metrics.reset()
        lr = self.optimizer.param_groups[0]['lr']

        total_iter = len(dataloader)
        t_start = time.time()
        for i, (input_feature, input_label, _) in enumerate(dataloader):
            t_process_start = time.time()

            input_feature = input_feature.cuda()  # B x 5 x H x W: range, x, y, z, intensity
            input_label = input_label.cuda().long()  # B x H x W: 1 = selected instance, 0 = background

            if torch.all(input_label == 0):
                continue  # instance not visible in the range image

            # Simulated user clicks, sampled from the instance and an expanded box around it
            point_coords, point_labels = batch_sample_points(
                input_label, fg_range=self.settings.num_fg_points, bg_range=self.settings.num_bg_points)

            if mode == 'Train':
                with torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    output = self.model(input_feature, point_coords, point_labels)
                    total_loss, loss_focal, loss_dice, loss_prompt = self.compute_losses(
                        output, input_label, point_coords, point_labels)

                self.optimizer.zero_grad()
                if self.fp16_scaler is None:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.settings.grad_clipping)
                    self.optimizer.step()
                else:
                    self.fp16_scaler.scale(total_loss).backward()
                    self.fp16_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.settings.grad_clipping)
                    self.fp16_scaler.step(self.optimizer)
                    self.fp16_scaler.update()
                self.scheduler.step()
            else:
                with torch.no_grad(), torch.cuda.amp.autocast(self.fp16_scaler is not None):
                    output = self.model(input_feature, point_coords, point_labels)
                    total_loss, loss_focal, loss_dice, loss_prompt = self.compute_losses(
                        output, input_label, point_coords, point_labels)

            with torch.no_grad():
                self.metrics.addBatch(output.argmax(dim=1), input_label)
            loss_meter.update(total_loss.item(), input_feature.size(0))

            # Logging
            data_cost_time = t_process_start - t_start
            process_cost_time = time.time() - t_process_start
            self.remain_time.update(cost_time=(time.time() - t_start), mode=mode)
            remain_time = datetime.timedelta(seconds=self.remain_time.getRemainTime(
                epoch=epoch, iters=i, total_iter=total_iter, mode=mode))
            t_start = time.time()

            if (i % self.settings.log_frequency == 0 or i == total_iter - 1) and self.recorder is not None:
                lr = self.optimizer.param_groups[0]['lr']
                mean_iou, _, mean_acc, _ = self.metrics.getIoUnAcc()
                self.recorder.logger.info(
                    '>>> {} E[{:03d}|{:03d}] I[{:04d}|{:04d}] DT[{:.3f}] PT[{:.3f}] LR {} Loss {:0.4f} '
                    'Acc {:0.4f} IOU {:0.4f} RT {}'.format(
                        mode, self.settings.n_epochs, epoch + 1, total_iter, i + 1, data_cost_time,
                        process_cost_time, lr, total_loss.item(), mean_acc.item(), mean_iou.item(), remain_time))

        with torch.no_grad():
            mean_acc, class_acc = self.metrics.getAcc()
            mean_recall, class_recall = self.metrics.getRecall()
            mean_iou, class_iou = self.metrics.getIoU()
        metrics_dict = {
            'mean_acc': mean_acc, 'class_acc': class_acc,
            'mean_recall': mean_recall, 'class_recall': class_recall,
            'mean_iou': mean_iou, 'class_iou': class_iou,
            'conf_matrix': self.metrics.conf_matrix.clone().cpu(),
        }

        if self.recorder is not None:
            loss_dict = {
                'loss_meter_avg': loss_meter.avg,
                'loss_focal': loss_focal,
                'loss_dice': loss_dice,
                'loss_prompt': loss_prompt,
            }
            class_names = dataset.instance_dataset.InstanceDataset.mapped_cls_name
            eval_results(self.recorder, metrics_dict, class_names, mode)
            tensorboard_logger(epoch, mode, self.recorder, metrics_dict, loss_dict, lr, class_names)
            self.recorder.logger.info('>>> {} Loss {:0.4f} Acc {:0.4f} IOU {:0.4f} Recall {:0.4f}'.format(
                mode, loss_meter.avg, mean_acc.item(), mean_iou.item(), mean_recall.item()))

        return {'Acc': mean_acc.item(), 'IOU': mean_iou.item(), 'Recall': mean_recall.item()}
