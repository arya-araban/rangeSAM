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

import os
import yaml

import utils.tools as tools


class Option(object):
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # General options (overridden from the command line in main.py)
        self.seed = 1
        self.gpu = None
        self.rank = 0
        self.world_size = 1
        self.distributed = False
        self.dist_backend = 'nccl'
        self.dist_url = 'env://'
        self.num_workers = self.config.get('num_workers', 4)
        self.log_frequency = 100
        self.data_root = None
        self.save_path = None
        self.val_only = False
        self.test_split = False

        # Data
        self.id = self.config['id']
        self.dataset = self.config['dataset']
        self.split = self.config['split']
        self.n_classes = self.config['n_classes']

        # Training
        self.n_epochs = self.config['n_epochs']
        self.warmup_epochs = self.config.get('warmup_epochs', 10)
        self.batch_size = self.config['batch_size']
        self.lr = self.config['lr']
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.grad_clipping = float(self.config['grad_clipping'])
        self.val_frequency = self.config.get('val_frequency', 1)
        self.use_fp16 = self.config.get('use_fp16', False)

        # Prompts sampled per training example: (min, max) foreground and background points
        self.num_fg_points = tuple(self.config.get('num_fg_points', [1, 12]))
        self.num_bg_points = tuple(self.config.get('num_bg_points', [0, 8]))

        # Loss
        self.use_prompt_loss = self.config.get('use_prompt_loss', True)
        self.focal_weight = self.config.get('focal_weight', 20.0)
        self.dice_weight = self.config.get('dice_weight', 1.0)
        self.prompt_weight = self.config.get('prompt_weight', 0.3)

        # Checkpoints / pre-trained ViT weights
        self.checkpoint = self.config.get('checkpoint', None)
        self.pretrained_model = self.config.get('pretrained_model', None)

        assert self.config['skip_filters'] in (0, self.config['D_h']), 'skip_filters has to be 0 or D_h'
        assert self.config['patch_size'] == self.config['patch_stride'], \
            'patch_size has to equal patch_stride when the convolutional stem is used'

    def check_path(self):
        if tools.is_main_process():
            if os.path.exists(self.save_path):
                print('WARNING: Directory exist: {}'.format(self.save_path))
            os.makedirs(self.save_path, exist_ok=True)
