#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import torch

from utils import register as R
from .abs_trainer import Trainer


@R.register('MEANTrainer')
class MEANTrainer(Trainer):

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss = self.model(**batch)

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)
        return loss