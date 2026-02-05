import hydra
import omegaconf
import torch
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
import os
from torch_geometric.data import Data
import pickle
import numpy as np

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)

class ConfidenceDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, path_prop: ValueNode, use_max_batch: ValueNode, threshold: ValueNode,
                 num_ref_pairs: ValueNode, num_gen_pairs: ValueNode):
        self.name = name
        self.path = path
        self.path_prop = path_prop
        self.samples = torch.load(self.path)
        self.num_ref_pairs = num_ref_pairs
        self.num_gen_pairs = num_gen_pairs
        self.num_pairs = self.num_ref_pairs + self.num_gen_pairs
        self.threshold = threshold
        with open(self.path_prop, 'rb') as f:
            self.props = pickle.load(f)
        self.num_comps = self.props.shape[1]
        self.num_samples = min(self.props.shape[0], use_max_batch)
        self.num_atoms = self.samples['num_atoms'][0]
        self.end_idx = torch.cumsum(self.num_atoms, dim=0)
        self.start_idx = self.end_idx - self.num_atoms
        self.valid = np.logical_and(self.props == self.props, self.props < self.threshold)
        self.min_props = np.nanmin(self.props[:self.num_samples], axis=0)
        self.min_valid = np.logical_and(self.min_props == self.min_props, self.min_props < self.threshold)
    
    def __len__(self) -> int:
        return self.num_comps * self.num_pairs

    def __getitem__(self, index):
        
        pair_idx, comp_idx = index // self.num_comps, index % self.num_comps
        start, end = self.start_idx[comp_idx], self.end_idx[comp_idx]

        if pair_idx < self.num_ref_pairs or not self.min_valid[comp_idx]: # ref, gen



            data1 = Data(
                frac_coords=self.samples['input_data_batch']['frac_coords'][start : end].reshape(-1,3),
                atom_types=self.samples['input_data_batch']['atom_types'][start : end].reshape(-1),
                lengths=self.samples['input_data_batch']['lengths'][comp_idx].view(1, -1),
                angles=self.samples['input_data_batch']['angles'][comp_idx].view(1, -1),
                num_atoms=self.num_atoms[comp_idx],
                num_nodes=self.num_atoms[comp_idx] 
            )                       # data1 is the reference

            sample_idx = np.random.choice(self.num_samples, 1)

            data2 = Data(
                frac_coords=self.samples['frac_coords'][sample_idx, start : end].reshape(-1,3),
                atom_types=self.samples['atom_types'][sample_idx, start : end].reshape(-1),
                lengths=self.samples['lengths'][sample_idx, comp_idx].view(1, -1),
                angles=self.samples['angles'][sample_idx, comp_idx].view(1, -1),
                num_atoms=self.num_atoms[comp_idx],
                num_nodes=self.num_atoms[comp_idx]  
            )

            return data1, data2


        else:   # gen1, gen2

            valid_idxs = np.where(self.valid[:self.num_samples, comp_idx])[0]
            sample_idx1 = np.random.choice(valid_idxs, 1)
            other_idxs = np.where(np.arange(self.num_samples) != sample_idx1)[0]
            sample_idx2 = np.random.choice(other_idxs, 1)

            if self.valid[sample_idx2, comp_idx] and self.props[sample_idx2, comp_idx] < self.props[sample_idx1, comp_idx]:
                sample_idx2, sample_idx1 = sample_idx1, sample_idx2

            data1 = Data(
                frac_coords=self.samples['frac_coords'][sample_idx1, start : end].reshape(-1,3),
                atom_types=self.samples['atom_types'][sample_idx1, start : end].reshape(-1),
                lengths=self.samples['lengths'][sample_idx1, comp_idx].view(1, -1),
                angles=self.samples['angles'][sample_idx1, comp_idx].view(1, -1),
                num_atoms=self.num_atoms[comp_idx],
                num_nodes=self.num_atoms[comp_idx]  
            )   

            data2 = Data(
                frac_coords=self.samples['frac_coords'][sample_idx2, start : end].reshape(-1,3),
                atom_types=self.samples['atom_types'][sample_idx2, start : end].reshape(-1),
                lengths=self.samples['lengths'][sample_idx2, comp_idx].view(1, -1),
                angles=self.samples['angles'][sample_idx2, comp_idx].view(1, -1),
                num_atoms=self.num_atoms[comp_idx],
                num_nodes=self.num_atoms[comp_idx]  
            )            

            return data1, data2


    def __repr__(self) -> str:
        return f"PreferenceDataset(len: {self.num_comps * self.num_pairs})" 