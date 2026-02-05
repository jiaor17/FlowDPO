#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
import torch

from utils import register as R
from .abs_trainer import Trainer


@R.register('DiffAbTrainer')
class DiffabTrainer(Trainer):

    # def train_step(self, batch, batch_idx):
    #     loss_dict = self.model(batch)
    #     loss = loss_dict['rot'] + loss_dict['pos'] + 5.0 * loss_dict['seq']

    #     val, log_type = False, 'Train'

    #     for key in loss_dict:
    #         self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)
    #     self.log(f'Loss/{log_type}', loss, batch_idx, val)

    #     lr = self.optimizer.state_dict()['param_groups'][0]['lr']
    #     self.log('lr', lr, batch_idx, val)
    #     return loss

    # def valid_step(self, batch, batch_idx):

    #     model = self.model.module if self.local_rank != -1 else self.model
    #     gen_S, gen_X = model.generate(batch, sample_opt={'pbar': False}, validation=True)

    #     mask = batch['mask'].unsqueeze(-1) & batch['atom_mask']
    #     gt_all_X = batch['X'].clone()
    #     gen_all_X = torch.where(mask.unsqueeze(-1), gen_X, gt_all_X)
    #     rmsd = ((gt_all_X - gen_all_X) ** 2).sum(-1)
    #     batch_size = rmsd.shape[0]
    #     rmsd = torch.sqrt(rmsd.view(batch_size, -1).sum(-1) / (mask.sum(-1).sum(-1))) # [bs]
    #     rmsd = torch.mean(rmsd)

    #     mask = batch['mask'] # [bs, L]
    #     aar = torch.mean(((batch['S'] == gen_S) & mask).sum(-1) / mask.sum(-1))

    #     self.log(f'Struct/bbRMSD/Validation', rmsd, batch_idx, val=True, batch_size=batch_size)
    #     self.log(f'Seq/AAR/Validation', aar, batch_idx, val=True, batch_size=batch_size)

    #     if model.train_structure:
    #         return rmsd
    #     else:
    #         return aar
    
    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        self.rand_rng_state = random.getstate() # data selection randomness
        random.seed(12)
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        random.setstate(self.rand_rng_state)
        return super()._valid_epoch_end(device)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss_dict = self.model(**batch)
        loss = loss_dict['pos'] + loss_dict['seq']

        log_type = 'Validation' if val else 'Train'

        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)
        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)
        return loss