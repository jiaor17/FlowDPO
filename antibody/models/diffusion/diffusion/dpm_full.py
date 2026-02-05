import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
import numpy as np

from torch.autograd import grad
from torch_scatter import scatter_mean

from utils.nn_utils import variadic_meshgrid, length_to_batch_id, knn_edges

from .transition import PositionTransition, AminoacidCategoricalTransition, FlowMatchingTransition

from .mc_egnn import MC_E_GCL


class EpsilonNet(nn.Module):

    def __init__(
            self,
            hidden_size,
            n_channel,
            num_classes=20,
            max_position=300,
            n_layers=3,
            dropout=0.1,
        ):
        super().__init__()
        
        self.n_layers = n_layers

        self.seq_embedding = nn.Embedding(num_classes, hidden_size)
        self.position_embedding = nn.Embedding(max_position, hidden_size)
        edge_size = hidden_size // 4
        self.edge_embedding = nn.Embedding(2, edge_size) # context or interaction

        input_size = hidden_size + 3 # beta encoding
        self.input_linear = nn.Linear(input_size, hidden_size)
        for i in range(n_layers):
            self.add_module(f'gcl_{i}', MC_E_GCL(
                hidden_size, hidden_size, hidden_size, n_channel,
                edges_in_d=edge_size, dropout=dropout
            ))

        self.eps_seq_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, num_classes), nn.Softmax(dim=-1) 
        )

    def forward(
            self, p_noisy, s_noisy, position_ids, ctx_edges, inter_edges, beta
        ):
        """
        Args:
            s_noisy: (N)
            X_noisy: (N, 14, 3)
            mask_generate: (N)
            batch_ids: (N)
            beta: (N)
        Returns:
            eps_H: (N, hidden_size)
            eps_X: (N, 14, 3)
        """

        # prepare inputs
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)
        edge_embed = torch.cat([
            torch.zeros_like(ctx_edges[0]), torch.ones_like(inter_edges[0]) # [E]
        ], dim=-1)
        edge_embed = self.edge_embedding(edge_embed) # [E, embed size]
        seq_embed = self.seq_embedding(s_noisy)
        position_embed = self.position_embedding(position_ids)
        h = torch.cat([seq_embed + position_embed, t_embed], dim=-1)
        x = p_noisy

        h = self.input_linear(h)
        for i in range(self.n_layers):
            h, x = self._modules[f'gcl_{i}'](h, edges, x, edge_attr=edge_embed)

        eps_p = x - p_noisy
        s_denoised = self.eps_seq_net(h)
        return eps_p, s_denoised


class FullDPM(nn.Module):

    def __init__(
        self, 
        hidden_size,
        n_channel,
        num_steps, 
        n_layers=3,
        num_classes=20,
        max_position=300,
        dropout=0.1,
        trans_pos_opt={}, 
        trans_seq_opt={},
        k_neighbors=9,
        flow_matching=False,
        std=10.0
    ):
        super().__init__()
        self.eps_net = EpsilonNet(
            hidden_size, n_channel, num_classes, max_position,
            n_layers, dropout
        )
        self.num_steps = num_steps
        if flow_matching:
            self.trans_pos = FlowMatchingTransition(num_steps, **trans_pos_opt)
        else:
            self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, num_classes, **trans_seq_opt)
        self.k_neighbors = k_neighbors
        self.std = std

        self.register_buffer('_dummy', torch.empty([0, ]))
    
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

    def _normalize_position(self, X, batch_ids, mask_generate, atom_mask):
        ctx_mask = mask_generate[:-1] != mask_generate[1:] # from 0 to 1 or from 1 to 0
        ctx_mask = F.pad(ctx_mask, pad=(0, 1), value=0)
        ctx_mask = (ctx_mask[:, None].expand_as(atom_mask)) & atom_mask # [N, 4]
        centers = scatter_mean(X[ctx_mask], batch_ids[:, None].expand_as(ctx_mask)[ctx_mask], dim=0) # [bs, 3]
        centers = centers[batch_ids].unsqueeze(1) # [N, 1, 3]
        X = (X - centers) / self.std
        return X, centers
    
    def _unnormalize_position(self, X_norm, centers):
        X = X_norm * self.std + centers
        return X
    
    def forward(self, S_0, X_0, position_ids, mask_generate, lengths, segment_ids, atom_mask, denoise_structure, denoise_sequence, t=None, reduction='all'):
        N = lengths.shape[0]
        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)
        batch_ids = length_to_batch_id(S_0, lengths)
        X_0, centers = self._normalize_position(X_0, batch_ids, mask_generate, atom_mask)

        if denoise_structure:
            # Add noise to positions
            p_noisy, eps_p = self.trans_pos.add_noise(X_0, mask_generate, t[batch_ids])
        else:
            p_noisy = X_0.clone()
            eps_p = torch.zeros_like(p_noisy)

        if denoise_sequence:
            # Add noise to sequence
            _, s_noisy = self.trans_seq.add_noise(S_0, mask_generate, t[batch_ids])
        else:
            s_noisy = S_0.clone()

        beta = self.trans_pos.var_sched.betas[t][batch_ids]
        ctx_edges, inter_edges = self._get_edges(p_noisy, batch_ids, lengths, segment_ids, atom_mask)
        eps_p_pred, c_denoised = self.eps_net(
            p_noisy, s_noisy, position_ids, ctx_edges, inter_edges, beta
        )   # (N, 4, 3), (N, 20)

        loss_dict = {}

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1).mean(dim=-1)  # (N)
        if reduction == 'all':
            loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8) # [1]
        elif reduction == 'graph':
            loss_pos = scatter_mean(loss_pos[mask_generate], batch_ids[mask_generate], dim=0) # [batch size]
        elif reduction == 'node':
            pass
        else:
            raise NotImplementedError(f'Reduction mode {reduction} not implemented')
        loss_dict['pos'] = loss_pos * (1.0 if denoise_structure else 0.0)

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, S_0, t[batch_ids])
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t[batch_ids]) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N)
        if reduction == 'all':
            loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8) # [1]
        elif reduction == 'graph':
            loss_seq = scatter_mean(kldiv[mask_generate], batch_ids[mask_generate], dim=0) # [batch size]
        elif reduction == 'node':
            loss_seq = kldiv
        else:
            raise NotImplementedError(f'Reduction mode {reduction} not implemented')
        loss_dict['seq'] = loss_seq * (1.0 if denoise_sequence else 0.0)

        return loss_dict

    @torch.no_grad()
    def sample(self,
        S, X, position_ids, mask_generate, lengths, segment_ids, atom_mask, 
        sample_structure=True, sample_sequence=True, pbar=False
    ):
        """
        Args:
            H: contextual hidden states, (N, latent_size)
            X: contextual atomic coordinates, (N, 14, 3)
            L: cholesky decomposition of the covariance matrix \Sigma=LL^T, (bs, 3, 3)
            energy_func: guide diffusion towards lower energy landscape
        """
        batch_ids = length_to_batch_id(S, lengths)
        X, centers = self._normalize_position(X, batch_ids, mask_generate, atom_mask)

        # Set the orientation and position of residues to be predicted to random values
        if sample_structure:
            X_rand = torch.randn_like(X) # [N, 14, 3]
            X_init = torch.where(mask_generate[:, None, None].expand_as(X), X_rand, X)
        else:
            X_init = X

        if sample_sequence:
            S_rand = torch.zeros_like(S)
            S_init = torch.where(mask_generate, S_rand, S)
        else:
            S_init = S

        # traj = {self.num_steps: (self._unnormalize_position(X_init, centers, batch_ids, L), H_init)}
        traj = {self.num_steps: (X_init, S_init)}
        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x
        for t in pbar(range(self.num_steps, 0, -1)):
            X_t, S_t = traj[t]
            # X_t, _ = self._normalize_position(X_t, batch_ids, mask_generate, atom_mask, L)
            # print(t, 'input', X_t[0, 0] * 1000)
            
            # beta = self.trans_x.var_sched.betas[t].view(1).repeat(X_t.shape[0])
            beta = self.trans_pos.var_sched.betas[t].expand([X_t.shape[0], ])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)
            
            ctx_edges, inter_edges = self._get_edges(X_t, batch_ids, lengths, segment_ids, atom_mask)
            eps_p_pred, c_denoised = self.eps_net(
                X_t, S_t, position_ids, ctx_edges, inter_edges, beta
            )   # (N, 4, 3), (N, 20)

            _, S_next = self.trans_seq.denoise(S_t, c_denoised, mask_generate, t_tensor)
            X_next = self.trans_pos.denoise(X_t, eps_p_pred, mask_generate, t_tensor)

            if not sample_structure:
                X_next = X_t
            if not sample_sequence:
                S_next = S_t

            # traj[t-1] = (self._unnormalize_position(X_next, centers, batch_ids, L), H_next)
            traj[t-1] = (X_next, S_next)
            traj[t] = (self._unnormalize_position(traj[t][0], centers).cpu(), traj[t][1].cpu())
            # traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.
        traj[0] = (self._unnormalize_position(traj[0][0], centers), traj[0][1])
        return traj