import argparse
import json
import os
import copy
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
from time import time

import yaml
import easydict
import torch
from torch.utils.data import DataLoader

from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.random_seed import setup_seed
from data import create_dataloader, create_dataset
from data.format import VOCAB


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    # ckpts = sorted(ckpts, key=lambda x:x[0])
    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def overwrite_blocks(blocks, seq=None, X=None):
    if seq is not None:
        assert len(blocks) == len(seq), f'{len(blocks)} {len(seq)}'
    new_blocks = []
    for i, block in enumerate(blocks):
        block = deepcopy(block)
        if seq is None:
            abrv = block.abrv
        else:
            abrv = VOCAB.symbol_to_abrv(seq[i])
            if block.abrv != abrv:
                if X is None:
                    block.units = [atom for atom in block.units if atom.name in VOCAB.backbone_atoms]
        if X is not None:
            coords = X[i]
            atoms = VOCAB.backbone_atoms + sidechain_atoms[VOCAB.abrv_to_symbol(abrv)]
            block.units = [
                Atom(atom_name, clamp_coord(coord), atom_name[0]) for atom_name, coord in zip(atoms, coords)
            ]
        block.abrv = abrv
        new_blocks.append(block)
    return new_blocks


def create_data_variants(config, structure_factory):
    structure = structure_factory()
    structure_id = structure['id']

    data_variants = []
    if config.mode == 'single_cdr':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            data_variants.append({
                'data': data_var,
                'name': f'{structure_id}-{cdr_name}',
                'tag': f'{cdr_name}',
                'cdr': cdr_name,
                'residue_first': residue_first,
                'residue_last': residue_last,
            })
    elif config.mode == 'multiple_cdrs':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        transform = Compose([
            MaskMultipleCDRs(selection=cdrs, augmentation=False),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-MultipleCDRs',
            'tag': 'MultipleCDRs',
            'cdrs': cdrs,
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'full':
        transform = Compose([
            MaskAntibody(),
            MergeChains(),
        ])
        data_var = transform(structure_factory())
        data_variants.append({
            'data': data_var,
            'name': f'{structure_id}-Full',
            'tag': 'Full',
            'residue_first': None,
            'residue_last': None,
        })
    elif config.mode == 'abopt':
        cdrs = sorted(list(set(find_cdrs(structure)).intersection(config.sampling.cdrs)))
        for cdr_name in cdrs:
            transform = Compose([
                MaskSingleCDR(cdr_name, augmentation=False),
                MergeChains(),
            ])
            data_var = transform(structure_factory())
            residue_first, residue_last = get_residue_first_last(data_var)
            for opt_step in config.sampling.optimize_steps:
                data_variants.append({
                    'data': data_var,
                    'name': f'{structure_id}-{cdr_name}-O{opt_step}',
                    'tag': f'{cdr_name}-O{opt_step}',
                    'cdr': cdr_name,
                    'opt_step': opt_step,
                    'residue_first': residue_first,
                    'residue_last': residue_last,
                })
    else:
        raise ValueError(f'Unknown mode: {config.mode}.')
    return data_variants


def main(args, opt_args):
    # load test config
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    
    # load model
    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
    print_log(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    test_loader = create_dataloader(test_set, config['dataloader'])
    
    # save path
    if args.save_dir is None:
        save_dir = os.path.join(ckpt_dir, 'results')
    else:
        save_dir = args.save_dir

    # generation
    config = easydict.EasyDict(config)

    # save reference
    for idx, item in enumerate(tqdm(test_set)):
        summary = test_set.get_summary(idx)
        out_dir = os.path.join(save_dir, summary['id'], config.dataset.test.cdr_type)
        os.makedirs(out_dir, exist_ok=True)
        summary['num_samples'] = config.sampling.num_samples
        summary['cdr_type'] = config.dataset.test.cdr_type
        summary['seq'] = ''.join([VOCAB.idx_to_symbol(aa) for aa in item['S'][item['generate_mask']]])
        summary['backbone'] = item['X'][item['generate_mask']].tolist()
        json.dump(summary, open(os.path.join(out_dir, 'metadata.json'), 'w'), indent=2)

    # generation
    for n in range(config.sampling.num_samples):
        print_log(f'Generation for {n}-th candidates')
        idx = 0
        for batch in tqdm(test_loader):
            batch = to_device(batch, device)
            batch_X, batch_S = model.sample(**batch)
            for X, S in zip(batch_X, batch_S):
                summary = test_set.get_summary(idx)
                out_dir = os.path.join(save_dir, summary['id'], config.dataset.test.cdr_type)
                X = X.cpu().tolist()
                json.dump(
                    {'id': summary['id'], 'cdr_type': config.dataset.test.cdr_type, 'seq': S, 'backbone': X}, 
                    open(os.path.join(out_dir, '%04d.json' % (n, )), 'w'),
                    indent=2
                )
                idx += 1

    return


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated peptides')

    parser.add_argument('--part_index', type=int, default=-1, help='Index of current process for parallel generation (e.g. 0)')
    parser.add_argument('--num_parts', type=int, default=-1, help='Total number of paralleling processes (e.g. 8)')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)