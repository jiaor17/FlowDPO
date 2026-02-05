#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from torch_scatter import scatter_sum

import utils.register as R
from data.format import VOCAB
from utils.nn_utils import knn_edges, variadic_meshgrid, length_to_batch_id
from utils.oom_decorator import oom_decorator

from .mc_egnn import MCAttEGNN


@R.register('MEANDock')
class MEANDock(nn.Module):
    def __init__(self, hidden_size, n_channel, n_layers, n_iter=3, k_neighbors=9, num_aa_type=len(VOCAB), max_position=300, dropout=0.1, dense=False):
        super().__init__()
        self.n_iter = n_iter
        self.n_channel = n_channel
        self.aa_embedding = nn.Embedding(num_aa_type, hidden_size)
        self.round_embedding = nn.Embedding(n_iter, hidden_size)
        self.position_embedding = nn.Embedding(max_position, hidden_size)
        self.gnn = MCAttEGNN(hidden_size, hidden_size, num_aa_type,
                             n_channel, 0, n_layers=n_layers,
                             residual=True, dropout=dropout, dense=dense)
        self.k_neighbors = k_neighbors

    def init_mask(self, X, generate_mask):
        '''
        set coordinates of masks following a unified distribution
        between the two ends
        '''
        X, cmask = X.clone(), torch.zeros_like(X, device=X.device).bool()
        n_channel, n_dim = X.shape[1:]
        l = generate_mask & (~F.pad(generate_mask[:-1], pad=(1, 0), value=False))
        r = generate_mask & (~F.pad(generate_mask[1:], pad=(0, 1), value=False))
        cdr_range = torch.stack([
            torch.nonzero(l).reshape(-1),
            torch.nonzero(r).reshape(-1)
        ], dim=-1)
        for start, end in cdr_range:
            l_coord, r_coord = X[start - 1], X[end + 1]  # [n_channel, 3]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)  # [n_mask, n_channel, 3]
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = True
        return X, cmask

    @torch.no_grad()
    def _get_edges(self, X, batch_ids, lengths, segment_ids, atom_mask):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        ) # (row, col)
        
        is_ctx = segment_ids[row] == segment_ids[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0) # [2, Ec]
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0) # [2, Ei]

        ctx_edges = knn_edges(ctx_edges, self.k_neighbors, X, atom_mask)
        inter_edges = knn_edges(inter_edges, self.k_neighbors, X, atom_mask)

        return ctx_edges, inter_edges

    @oom_decorator
    def forward(self, X, S, generate_mask, position_ids, segment_ids, lengths, atom_mask):
        '''
        :param X: [n_all_node, n_channel, 3]
        :param S: [n_all_node]
        :param L: list of cdr types
        :param offsets: [batch_size + 1]
        '''
        # save ground truth
        true_X = X.clone()

        # init mask
        X, cmask = self.init_mask(X, generate_mask)  # [n_all_node, n_channel, 3]
        aa_cnt = generate_mask.sum()

        H_0 = self.aa_embedding(S) + self.position_embedding(position_ids)  # [n_all_node, embed_size]

        loss = 0
        batch_ids = length_to_batch_id(S, lengths)

        for r in range(self.n_iter):
            ctx_edges, inter_edges = self._get_edges(X, batch_ids, lengths, segment_ids, atom_mask)

            r_embedding = self.round_embedding(torch.ones_like(generate_mask).long() * r)
            H, Z = self.gnn(H_0 + r_embedding, X, ctx_edges, inter_edges)

            # refine
            X = X.clone()
            X[generate_mask] = Z[generate_mask]
            
        loss = F.mse_loss(Z[cmask], true_X[cmask], reduction='sum') / (aa_cnt * self.n_channel)

        return loss

    
    @torch.no_grad()
    def sample(
        self,
        X, S, generate_mask, position_ids, segment_ids, lengths, atom_mask,
        sample_opt={
            'pbar': False,
        },
        return_tensor=False
    ):
        # init mask
        X, cmask = self.init_mask(X, generate_mask)  # [n_all_node, n_channel, 3]
        aa_cnt = generate_mask.sum()

        H_0 = self.aa_embedding(S) + self.position_embedding(position_ids)  # [n_all_node, embed_size]

        loss = 0
        batch_ids = length_to_batch_id(lengths)

        for r in range(self.n_iter):
            ctx_edges, inter_edges = self._get_edges(X, batch_ids, lengths, segment_ids, atom_mask)

            r_embedding = self.round_embedding(torch.ones_like(generate_mask).long() * r)
            H, Z = self.gnn(H_0 + r_embedding, X, ctx_edges, inter_edges)

            # refine
            X = X.clone()
            X[generate_mask] = Z[generate_mask]
        
        if return_tensor:
            return X[generate_mask], S[generate_mask]
        batch_X, batch_S = [], []
        for i in range(lengths.shape[0]):
            mask = (batch_ids == i) & generate_mask
            batch_X.append(X_0[mask])
            batch_S.append(None)

        return batch_X, batch_S