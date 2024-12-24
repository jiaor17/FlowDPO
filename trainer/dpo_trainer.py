#!/usr/bin/python
# -*- coding:utf-8 -*-
import random
from copy import deepcopy
import torch
from torch_scatter import scatter_mean

from utils import register as R
from utils.gnn_utils import length_to_batch_id
from .abs_trainer import Trainer


@R.register('DPOTrainer')
class DPOTrainer(Trainer):

    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.dpo_beta = self.config.dpo_beta * self.config.timesteps
        self.dpo_mode = self.config.dpo_mode
        self.add_diffusion_loss = getattr(self.config, 'add_diffusion_loss', False)

        self.model_ref = deepcopy(self.model)
        for param in self.model_ref.parameters():
            param.requires_grad = False
        self.model_ref.eval()

        self.criteria = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _train_epoch(self, device):
        self.model_ref.to(device)
        return super()._train_epoch(device)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def _valid_epoch_begin(self, device):
        self.model_ref.to(device)
        self.model_ref.eval()
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
        batch.pop('use_dpo')
        true_X = batch['X'][batch['generate_mask']]
        gen_X, gen_S = self.model.sample(**batch, return_tensor=True)
        batch_ids = length_to_batch_id(batch['S'], batch['lengths'])[batch['generate_mask']]
        rmsd = ((true_X[:, 1] - gen_X[:, 1]) ** 2).sum(-1) # [N]
        rmsd = scatter_mean(rmsd, batch_ids)[0::2] # [bs], only calculate rmsd on positive data
        rmsd = torch.sqrt(rmsd)
        self.log('RMSD_CA/validation', rmsd.mean(), batch_idx, val=True, batch_size=rmsd.shape[0])
        return rmsd
        
        return self.share_step(batch, batch_idx, val=True)
        loss_dict = self.model(batch)
        loss = loss_dict['rot'] + loss_dict['pos'] + loss_dict['seq']

        log_type = 'Validation'

        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val=True)
        self.log(f'Loss/{log_type}', loss, batch_idx, val=True)
        return loss

    ########## Override end ##########

    def _reduction(self, loss, batch, loss_mask=None):
        if self.dpo_mode == 'node': # maybe this is not right
            mask_generate = batch['generate_mask'][0::2]
            loss = (loss * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        else:
            if loss_mask is not None:
                if not torch.any(loss_mask):
                    return 0
                loss = loss[loss_mask]
            loss = loss.mean()
        return loss

    def share_step(self, batch, batch_idx, val=False):
        use_dpo = batch.pop('use_dpo')

        # pass in same t for pos-neg pairs
        batch_size = batch['lengths'].shape[0] // 2
        t = torch.randint(0, self.config.timesteps, (batch_size,), dtype=torch.long, device=batch['S'].device)
        t = torch.stack([t, t], dim=-1).flatten()
        batch['t'] = t
        with torch.no_grad(): # reference model
            loss_dict_ref = self.model_ref(**batch, reduction=self.dpo_mode)
            loss_ref = loss_dict_ref['pos'] + loss_dict_ref['seq']
        
        loss_dict = self.model(**batch, reduction=self.dpo_mode)
        loss = loss_dict['pos'] + loss_dict['seq']
        pos_loss = loss[0::2]
        loss = loss - loss_ref
        loss = torch.clip(loss, -100.0, 100.0)

        # dpo loss
        pos_dist, neg_dist = loss[0::2], loss[1::2]
        pos_neg_dist = pos_dist - neg_dist
        logits = -self.dpo_beta * pos_neg_dist
        labels = torch.ones(logits.shape, device=logits.device)

        loss = self.criteria(logits, labels)
        loss = self._reduction(loss, batch, use_dpo)

        log_type = 'Validation' if val else 'Train'
        self.log(f'DPOLoss/{log_type}', loss, batch_idx, val)

        if self.add_diffusion_loss:
            loss = loss + self._reduction(pos_loss, batch)
        else:
            loss = loss + self._reduction(pos_loss, batch, ~use_dpo)


        self.log(f'Loss/{log_type}', loss, batch_idx, val)
        self.log(f'Positive Dist/{log_type}', self._reduction(pos_dist, batch, use_dpo), batch_idx, val)
        self.log(f'Negative Dist/{log_type}', self._reduction(neg_dist, batch, use_dpo), batch_idx, val)
        for key in loss_dict:
            self.log(f'{key}/{log_type}', self._reduction(loss_dict[key][0::2], batch), batch_idx, val)
        self.log(f'Positive Sample Loss/{log_type}', self._reduction(pos_loss, batch), batch_idx, val)
        self.log(f'Pos-Neg Dist/{log_type}', self._reduction(pos_neg_dist, batch, use_dpo), batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)
        
        # if val:
        #     return self._reduction(pos_neg_dist, batch)
        # else:
        #     return loss
        return loss