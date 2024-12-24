#!/usr/bin/python
# -*- coding:utf-8 -*-
import enum

import torch
import torch.nn as nn

import utils.register as R
from utils.oom_decorator import oom_decorator
from data.format import VOCAB
from utils.nn_utils import length_to_batch_id

from .diffusion.dpm_full import FullDPM


@R.register('DiffAb')
class DiffAb(nn.Module):

    def __init__(
            self,
            mode='fixseq',
            diffusion_opt={}):
        super().__init__()
        
        self.train_sequence, self.train_structure = True, True
        if mode == 'fixbb':
            self.train_structure = False
        elif mode == 'fixseq':
            self.train_sequence = False
        
        self.diffusion = FullDPM(
            num_classes=len(VOCAB),
            **diffusion_opt
        )

    @oom_decorator
    def forward(self, X, S, generate_mask, position_ids, segment_ids, lengths, atom_mask, t=None, reduction='all'):
        '''
        segment_ids: 0 for antigen, 1 for heavy chain, 2 for light chain
        '''
        loss_dict = self.diffusion.forward(
            S_0=S,
            X_0=X,
            position_ids=position_ids,
            mask_generate=generate_mask,
            lengths=lengths,
            segment_ids=segment_ids,
            atom_mask=atom_mask,
            t=t,
            reduction=reduction,
            denoise_structure=self.train_structure,
            denoise_sequence=self.train_sequence
        )

        # loss
        return loss_dict

    @torch.no_grad()
    def sample(
        self,
        X, S, generate_mask, position_ids, segment_ids, lengths, atom_mask,
        sample_opt={
            'pbar': False,
        },
        return_tensor=False
    ):
        sample_opt['sample_sequence'] = self.train_sequence
        sample_opt['sample_structure'] = self.train_structure
        
        traj = self.diffusion.sample(
            S=S,
            X=X,
            position_ids=position_ids,
            mask_generate=generate_mask,
            lengths=lengths,
            segment_ids=segment_ids,
            atom_mask=atom_mask,
            **sample_opt
        )
        X_0, S_0 = traj[0]

        if return_tensor:
            return X_0[generate_mask], S_0[generate_mask]

        batch_ids = length_to_batch_id(S, lengths)
        batch_X, batch_S = [], []
        for i in range(lengths.shape[0]):
            mask = (batch_ids == i) & generate_mask
            if self.train_structure:
                batch_X.append(X_0[mask])
            else:
                batch_X.append(None)
            if self.train_sequence:
                batch_S.append(''.join([VOCAB.idx_to_symbol(s) for s in S_0[mask]]))
            else:
                batch_S.append(None)

        return batch_X, batch_S