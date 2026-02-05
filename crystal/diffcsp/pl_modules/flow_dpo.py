import math, copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from typing import Any, Dict

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": self.hparams.optim.monitor}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPFlow(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hparams.latent_dim = 0
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        self.decoder_ref = hydra.utils.instantiate(self.hparams.decoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)

        origin_model = torch.load(self.hparams.ori_ckpt)['state_dict']
        decoder_state_dict = self.get_decoder_state_dict(origin_model)
        self.timesteps = self.hparams.timesteps
        self.decoder.load_state_dict(decoder_state_dict, strict=False)
        self.decoder_ref.load_state_dict(decoder_state_dict, strict=False)

        for param in self.decoder_ref.parameters():
            param.requires_grad = False

        self.decoder_ref.eval()

        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.dpo_beta = self.hparams.dpo_beta * self.hparams.timesteps

        self.criteria = nn.BCEWithLogitsLoss()




    def clip_loss(self, loss):
        
        if torch.isinf(loss) or torch.isnan(loss):
            return torch.zeros_like(loss)
        return loss


    def get_kl_dist(self, model_new, model_ref, batch, time_consts):

        time_emb, c0, c1 = time_consts

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords
        rand_l, rand_x = torch.randn_like(lattices), torch.rand_like(frac_coords)
        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        optimal_rand_x = self.lap(rand_x, frac_coords, batch.atom_types, batch.num_atoms, batch.batch)
        optimal_delta_f = self.find_opt_f1_minus_f2(optimal_rand_x, frac_coords)
        optimal_delta_f = self.de_translation(optimal_delta_f, batch.batch)

        c1_per_atom = c1.repeat_interleave(batch.num_atoms)[:, None]

        input_frac_coords = (frac_coords + c1_per_atom * optimal_delta_f) % 1.  # xt = exp_x1 (t log_x1 (x0))
        
        tar_x = -optimal_delta_f         

        pred_l_new, pred_x_new = model_new(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        pred_l_ref, pred_x_ref = model_ref(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        pred_x_new = self.de_translation(pred_x_new, batch.batch)
        pred_x_ref = self.de_translation(pred_x_ref, batch.batch)

        loss_new = self.hparams.cost_lattice * F.mse_loss(pred_l_new, rand_l, reduction='none').mean(dim=(-1, -2)) + self.hparams.cost_coord * scatter(F.mse_loss(pred_x_new, tar_x, reduction='none').mean(dim=-1), batch.batch, reduce='mean')
        loss_ref = self.hparams.cost_lattice * F.mse_loss(pred_l_ref, rand_l, reduction='none').mean(dim=(-1, -2)) + self.hparams.cost_coord * scatter(F.mse_loss(pred_x_ref, tar_x, reduction='none').mean(dim=-1), batch.batch, reduce='mean')

        return loss_new - loss_ref

    def uniform_sample_t(self, batch_size, device):
        ts = np.random.choice(np.arange(1, self.timesteps+1), batch_size)
        return torch.from_numpy(ts).to(device)

    def find_opt_f1_minus_f2(self, f1, f2):
        p = 2 * math.pi
        return torch.atan2(torch.sin((f1 - f2)*p), torch.cos((f1 - f2)*p)) / p

    def forward(self, batch):


        self.decoder_ref.eval()

        batch_pos, batch_neg = batch

        batch_size = batch_pos.num_graphs
        times = self.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)


        c1 = times / self.timesteps
        c0 = 1 - c1


        time_consts = (time_emb, c0, c1)

        pos_dist = self.get_kl_dist(self.decoder, self.decoder_ref, batch_pos, time_consts)

        neg_dist = self.get_kl_dist(self.decoder, self.decoder_ref, batch_neg, time_consts)

        logits = -self.dpo_beta * (pos_dist - neg_dist)

        labels = torch.ones(batch_size).to(self.device)

        loss = self.criteria(logits, labels)

        return {
            'loss' : loss,
            'pos_dist' : pos_dist.mean(),
            'neg_dist' : neg_dist.mean()
        }


    def flow_forward(self, batch):


        batch_size = batch.num_graphs
        times = self.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        c1 = times / self.timesteps
        c0 = 1 - c1


        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords # x1

        rand_l, rand_x = torch.randn_like(lattices), torch.rand_like(frac_coords) # x0

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l

        optimal_rand_x = self.lap(rand_x, frac_coords, batch.atom_types, batch.num_atoms, batch.batch)
        
        optimal_delta_f = self.find_opt_f1_minus_f2(optimal_rand_x, frac_coords)

        optimal_delta_f = self.de_translation(optimal_delta_f, batch.batch)

        c1_per_atom = c1.repeat_interleave(batch.num_atoms)[:, None]

        input_frac_coords = (frac_coords + c1_per_atom * optimal_delta_f) % 1.  # xt = exp_x1 (t log_x1 (x0))


        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)

        pred_x = self.de_translation(pred_x, batch.batch)

        tar_x = -optimal_delta_f    # d_xt


        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss_lattice = self.clip_loss(loss_lattice)
        loss_coord = self.clip_loss(loss_coord)


        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord)

        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord
        }

    def de_translation(self, coord_shift, batch_idx):

        graph_shift = scatter(coord_shift, batch_idx, reduce='mean', dim=0)
        return coord_shift - graph_shift[batch_idx]


    @torch.no_grad()
    def sample(self, batch, diff_ratio = 1.0, step_lr = 1e-5):


        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        time_start = self.timesteps - 1

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']



            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)


            pred_x = self.de_translation(pred_x, batch.batch)

            step_size = 1. / self.timesteps

            x_t_minus_1 = x_t + step_size * pred_x # * (1 + step_lr * (1 - t / self.timesteps))

            l_t_minus_1 = l_t - step_size * (pred_l - l_t) / (1. - t * step_size)




            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        tar = traj[0]


        return tar, traj_stack


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        loss = output_dict['loss']
        pos_dist = output_dict['pos_dist']
        neg_dist = output_dict['neg_dist']


        self.log_dict(
            {'train_loss': loss,
            'pos_dist': pos_dist,
            'neg_dist': neg_dist},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, *args) -> torch.Tensor:

        batch = args[0]


        output_dict = self.get_val_output_dict(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, *args) -> torch.Tensor:

        batch = args[0]

        output_dict = self.get_val_output_dict(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def get_val_output_dict(self, batch):

        try:
            batch_pos, batch_neg = batch
            output_dict = self(batch)
        except:
            output_dict = self.flow_forward(batch)

        return output_dict

    def compute_stats(self, output_dict, prefix):

        log_dict = {f'{prefix}_{k}':v for k,v in output_dict.items()}
        loss = output_dict['loss']

        return log_dict, loss