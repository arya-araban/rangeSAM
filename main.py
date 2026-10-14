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

import argparse
import datetime
import os
import time

import numpy as np
import torch

from option import Option
from train import Trainer
from models import build_model
import utils.tools as tools


class Experiment(object):
    def __init__(self, settings: Option):
        self.settings = settings

        tools.init_distributed_mode(self.settings)
        if self.settings.distributed:
            torch.distributed.barrier()

        self.settings.check_path()

        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        np.random.seed(self.settings.seed)
        if self.settings.gpu is not None:
            torch.cuda.set_device(self.settings.gpu)
        torch.backends.cudnn.benchmark = True

        self.recorder = None
        if tools.is_main_process():
            self.recorder = tools.Recorder(self.settings, self.settings.save_path)

        self.epoch_start = 0
        self.model = build_model(self.settings.config, pretrained_path=self.settings.pretrained_model)
        if self.recorder is not None:
            self.recorder.logger.info(f'model = {self.model}')
            for key, val in self.model.counter_model_parameters().items():
                self.recorder.logger.info(f'==> {key}: {val}')

        self.trainer = Trainer(self.settings, self.model, self.recorder)
        self._loadCheckpoint()

    def _loadCheckpoint(self):
        if self.settings.checkpoint is None:
            return
        if not os.path.isfile(self.settings.checkpoint):
            raise FileNotFoundError('checkpoint file not found: {}'.format(self.settings.checkpoint))
        print(f'Loading checkpoint {self.settings.checkpoint}')

        checkpoint_data = torch.load(self.settings.checkpoint, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'])

        if not self.settings.val_only:
            self.trainer.optimizer.load_state_dict(checkpoint_data['optimizer'])
            self.epoch_start = checkpoint_data['epoch'] + 1
            if self.trainer.fp16_scaler is not None and checkpoint_data.get('fp16_scaler') is not None:
                self.trainer.fp16_scaler.load_state_dict(checkpoint_data['fp16_scaler'])

    def _save(self, path, epoch, extra=None):
        checkpoint_data = {
            'model': self.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'epoch': epoch,
        }
        if self.trainer.fp16_scaler is not None:
            checkpoint_data['fp16_scaler'] = self.trainer.fp16_scaler.state_dict()
        if extra:
            checkpoint_data.update(extra)
        torch.save(checkpoint_data, path)

    def run(self):
        t_start = time.time()

        if self.settings.val_only:
            self.trainer.run(self.epoch_start, mode='Validation')
        else:
            best_val_result = None
            for epoch in range(self.epoch_start, self.settings.n_epochs):
                self.trainer.run(epoch, mode='Train')

                is_val_epoch = (epoch % self.settings.val_frequency == 0 or
                                epoch == self.settings.n_epochs - 1)
                if self.trainer.val_loader is not None and is_val_epoch:
                    val_result = self.trainer.run(epoch, mode='Validation')

                    if self.recorder is not None:
                        if best_val_result is None:
                            best_val_result = dict(val_result)
                        for k, v in val_result.items():
                            if v >= best_val_result[k]:
                                best_val_result[k] = v
                                self.recorder.logger.info('New best {}: {}'.format(k, v))
                                self._save(os.path.join(self.recorder.checkpoint_path, f'best_{k}_model.pth'),
                                           epoch, extra={k: v})
                        self.recorder.logger.info('>>> Best result: ' + ' '.join(
                            f'{k}: {v:.4f}' for k, v in best_val_result.items()))

                if self.recorder is not None:
                    self._save(os.path.join(self.recorder.checkpoint_path, 'checkpoint.pth'), epoch)

        if self.recorder is not None:
            self.recorder.logger.info('=== Total time: {}'.format(
                datetime.timedelta(seconds=time.time() - t_start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or evaluate RangeSAM')
    parser.add_argument('config_path', type=str, help='path to a config file in configs/')
    parser.add_argument('--data_root', type=str, required=True, help='dataset root directory')
    parser.add_argument('--save_path', type=str, required=True, help='directory for logs and checkpoints')
    parser.add_argument('--id', type=str, help='name of the run (defaults to the id in the config)')
    parser.add_argument('--pretrained_model', type=str,
                        help='image-pretrained ViT weights used to initialise the encoder')
    parser.add_argument('--checkpoint', type=str, help='RangeSAM checkpoint to resume training or evaluate')
    parser.add_argument('--val_only', action='store_true', help='only run validation')
    parser.add_argument('--test_split', action='store_true', help='validate on the test split')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--log_frequency', type=int, default=100, help='logging frequency (iterations)')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--local-rank', type=int, default=0, help='set by torch.distributed launchers')
    args = parser.parse_args()

    settings = Option(args.config_path)
    settings.id = args.id or settings.id
    settings.data_root = args.data_root
    settings.val_only = args.val_only
    settings.test_split = args.test_split
    settings.num_workers = args.num_workers
    settings.log_frequency = args.log_frequency
    settings.seed = args.seed
    if args.pretrained_model is not None:
        settings.pretrained_model = args.pretrained_model
    if args.checkpoint is not None:
        settings.checkpoint = args.checkpoint
        settings.pretrained_model = None  # all weights come from the checkpoint

    # Never overwrite a previous run: log_<id>, log_<id>_1, log_<id>_2, ...
    run_name = f'eval_{settings.id}' if settings.val_only else f'log_{settings.id}'
    save_path = os.path.join(args.save_path, run_name)
    counter = 1
    while os.path.exists(save_path):
        save_path = os.path.join(args.save_path, f'{run_name}_{counter}')
        counter += 1
    settings.save_path = save_path
    print(f'Save path: {settings.save_path}')

    Experiment(settings).run()
