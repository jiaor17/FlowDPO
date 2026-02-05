import math
import json
from typing import Optional
from copy import deepcopy

import torch

import utils.register as R

from .format import VOCAB
from .mmap_dataset import MMAPDataset
from .constants import CDR


def block_to_X(block): # backbone only
    x, atom_mask = [], []
    for atom_name in VOCAB.backbone_atoms:
        atom_mask.append([])
        if block.has_unit(atom_name):
            x.append(block.get_unit_by_name(atom_name).get_coord())
            atom_mask[-1].append(1)
        else:
            x.append((0, 0, 0))
            atom_mask[-1].append(0)
    return x, atom_mask


def item_to_data(item, cdr_type_enum):
    ag_blocks = []
    if item['antigen'] is not None:
        for i, (_, blocks) in enumerate(item['antigen']):
            epitope_idxs = item['epitope_resid'][i][1]
            for j in epitope_idxs:
                ag_blocks.append(blocks[j])
    h_blocks = [] if item['heavy'] is None else item['heavy'][1]
    l_blocks = [] if item['light'] is None else item['light'][1]
    h_cdrmap = [] if item['heavy_cdrmap'] is None else item['heavy_cdrmap']
    l_cdrmap = [] if item['light_cdrmap'] is None else item['light_cdrmap']

    all_blocks = ag_blocks + h_blocks + l_blocks
    X = [block_to_X(block)[0] for block in all_blocks]
    S = [VOCAB.abrv_to_idx(block.abrv) for block in all_blocks]
    position_ids = [0 for _ in ag_blocks] + [block.id[0] for block in h_blocks + l_blocks]
    atom_mask = [block_to_X(block)[1] for block in all_blocks]
    lengths = len(S)
    
    generate_mask = [0 for _ in ag_blocks] + \
                    [1 if flag == cdr_type_enum else 0 for flag in h_cdrmap] + \
                    [1 if flag == cdr_type_enum else 0 for flag in l_cdrmap]
    segment_ids = [0 for _ in ag_blocks] + [1 for _ in h_blocks] + [2 for _ in l_blocks]

    data = {
        'X': torch.tensor(X, dtype=torch.float), # [N, 4, 3],
        'S': torch.tensor(S, dtype=torch.long), # [N]
        'generate_mask': torch.tensor(generate_mask, dtype=torch.bool), # [N]
        'position_ids': torch.tensor(position_ids, dtype=torch.long), # [N]
        'segment_ids': torch.tensor(segment_ids, dtype=torch.long), # [N]
        'atom_mask': torch.tensor(atom_mask, dtype=torch.bool).squeeze(-1), # [N, 4]
        'lengths': lengths # [1]
    }
    return data


@R.register('SAbDab')
class SAbDab(MMAPDataset):

    def __init__(self, mmap_dir: str, cdr_type: str, specify_data: Optional[str] = None, specify_index: Optional[str] = None) -> None:
        super().__init__(mmap_dir, specify_data, specify_index)
        '''
        cdr_type: H_CDR1, ..., L_CDR1, ...
        '''
        self.cdr_type = cdr_type
        self.cdr_type_enum = CDR.cdr_str_to_enum(cdr_type)
        assert self.cdr_type_enum is not None
        self.summaries = [json.loads(p[0]) for p in self._properties]

    def get_len(self, idx):
        return self.summaries[idx]['length']
    
    def get_summary(self, idx):
        return deepcopy(self.summaries[idx])

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        return item_to_data(item, self.cdr_type_enum)
    
    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if values[0] is None:
                results[key] = None
                continue
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            else:
                results[key] = torch.cat(values, dim=0)
        return results
    

if __name__ == '__main__':
    import sys
    dataset = SAbDab(sys.argv[1], 'H_CDR3')
    print(len(dataset))
    print(dataset[0])