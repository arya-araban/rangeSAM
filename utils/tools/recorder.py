'''
Adapted from Z. Zhuang et al.
https://github.com/ICEORY/PMF
'''

import os
import shutil
import logging
import sys
import tensorboardX


class Recorder(object):
    """Console/file logger, TensorBoard writer and checkpoint directory of a run."""

    def __init__(self, settings, save_path, use_tensorboard=True):
        self.save_path = save_path
        self.settings = settings
        self.log_path = os.path.join(self.save_path, 'log')
        self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.tensorboard = tensorboardX.SummaryWriter(logdir=self.save_path) if use_tensorboard else None
        self.logger = self._initLogger()
        self._saveConfig()

    def _initLogger(self):
        logger = logging.getLogger('console')
        logger.propagate = False
        file_handler = logging.FileHandler(os.path.join(self.log_path, 'console.log'))
        file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        return logger

    def _saveConfig(self):
        shutil.copyfile(self.settings.config_path, os.path.join(self.save_path, 'config.yaml'))
        with open(os.path.join(self.log_path, 'settings.log'), 'w') as f:
            for k, v in self.settings.__dict__.items():
                if k != 'config':
                    f.write('{}: {}\n'.format(k, v))
