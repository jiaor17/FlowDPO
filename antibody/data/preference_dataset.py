import json
import random
from copy import deepcopy
from typing import Dict, List, Optional

import torch
from easydict import EasyDict
from Bio.PDB import Polypeptide

import utils.register as R

from .sabdab import SAbDab
from .mmap_dataset import MMAPDataset


def default_minus(a, b):
    a_met = 0.0 if a is None else a.RMSD
    b_met = 0.0 if b is None else b.RMSD
    return a_met - b_met


@R.register('PreferenceDataset')
class PreferenceDataset(SAbDab):
    def __init__(self, mmap_dir: str, cdr_type: str, dpo_mmap_dir: str, specify_data: Optional[str] = None, specify_index: Optional[str] = None,
                 p_ref: float=0.4, threshold: float=1.0, negative_threshold: float=2.0, minus: callable=default_minus) -> None:
        super().__init__(mmap_dir, cdr_type, specify_data, specify_index)
        
        self.cdr_type = cdr_type
        self.dpo_dataset = MMAPDataset(dpo_mmap_dir)
        self.p_ref = p_ref
        self.minus = minus
        self.threshold = threshold
        self.neg_threshold = negative_threshold

        # load metrics / id to indexes
        self.all_metrics, self.id2indexes = [], {}
        for i, (index, props) in enumerate(zip(self.dpo_dataset._indexes, self.dpo_dataset._properties)):
            _id = index[0] # pdbcode_hchain_lchain_agchains-CDR-index
            metrics = EasyDict(json.loads(props[-1]))

            self.all_metrics.append(metrics)
            self.id2indexes[_id] = i
        
        # classify candidates
        self.ref2cands = {}
        for i, index in enumerate(self.dpo_dataset._indexes):
            _id = index[0]
            item_id, cdr_type, num = _id.split('-')
            ref_id = item_id
            if ref_id not in self.ref2cands:
                self.ref2cands[ref_id] = {}
            if cdr_type not in self.ref2cands[ref_id]:
                self.ref2cands[ref_id][cdr_type] = []
            self.ref2cands[ref_id][cdr_type].append(i)
        self.references = []
        for index in self._indexes:
            ref_id = index[0]
            if ref_id in self.ref2cands:
                self.references.append(ref_id)
        
    def __len__(self):
        return len(self.references)
    
    def _overwrite(self, full_data, cdr_data):
        seq, x = cdr_data['seq'], cdr_data['backbone']
        mask = full_data['generate_mask']
        if seq is not None:
            full_data['S'][mask] = seq
        if x is not None:
            full_data['X'][mask] = torch.tensor(x)
        return full_data

    def __getitem__(self, idx: int):
        ref_id = self.references[idx]
        cands_i = self.ref2cands[ref_id][self.cdr_type]
        qualified = [self.minus(self.all_metrics[i], None) < self.threshold for i in cands_i]
        neg_qualified = [self.minus(self.all_metrics[i], None) > self.neg_threshold for i in cands_i]
        neg_cands_i = [i for n, i in enumerate(cands_i) if neg_qualified[n]]
        if len(neg_cands_i) == 0:
            neg_cands_i = cands_i

        ref_data = super().__getitem__(idx)
        # if len(neg_cands_i) == 0:
        #     return (ref_data, ref_data), False

        if random.random() < self.p_ref or sum(qualified) == 0 or len(cands_i) < 2:
            # use reference as positive sample
            pos_data = ref_data
            neg_data = self._overwrite(deepcopy(ref_data), self.dpo_dataset[random.choice(neg_cands_i)])
        else:
            # select one from qualified candidates as positive sample
            pos_i = random.choice([i for n, i in enumerate(cands_i) if qualified[n]])
            neg_i = random.choice([i for i in neg_cands_i if i != pos_i])
            if self.minus(self.all_metrics[pos_i], self.all_metrics[neg_i]) > 0:
                pos_i, neg_i = neg_i, pos_i
            pos_data = self._overwrite(deepcopy(ref_data), self.dpo_dataset[pos_i])
            neg_data = self._overwrite(deepcopy(ref_data), self.dpo_dataset[neg_i])
        
        return (pos_data, neg_data), True

    def collate_fn(self, batch):
        flat_batch = []
        use_dpo = []
        for (pos, neg), dpo in batch:
            flat_batch.append(pos)
            flat_batch.append(neg)
            use_dpo.append(dpo)
        results = super().collate_fn(flat_batch)
        results['use_dpo'] = torch.tensor(use_dpo, dtype=torch.bool)
        return results
    

if __name__ == '__main__':
    import sys
    dataset = PreferenceDataset(sys.argv[1], sys.argv[2], sys.argv[3], specify_index=sys.argv[4])
    print(len(dataset))
    random.seed(0)
    print(dataset[0])
