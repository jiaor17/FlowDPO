#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import argparse
import statistics
from easydict import EasyDict

import ray
import numpy as np
from Bio import PDB

from utils.logger import print_log
from evaluation.rmsd import compute_rmsd

from .base import TaskScanner, Task



def get_cdr(bio_struct, residue_first, residue_last):
    assert residue_first[0] == residue_last[0], "Start residue and End residue not in the same chain!"
    residue_first, residue_last = tuple(residue_first), tuple(residue_last)
    chain = residue_first[0]
    data, seq_map = parsers.parse_biopython_structure(bio_struct[chain])
    start, end = seq_map[residue_first], (seq_map[residue_last] + 1)
    for key in data:
        data[key] = data[key][start:end]
    return data


@ray.remote(num_cpus=1)
def cal_metrics(task: Task):
    
    model = json.load(open(task.current_path, 'r'))
    ref = json.load(open(task.ref_path, 'r'))

    model_backbone, ref_backbone = np.array(model['backbone']), np.array(ref['backbone'])

    # CA RMSD
    rmsd = compute_rmsd(model_backbone[:, 1], ref_backbone[:, 1])
    task.set_metric('RMSD', rmsd)

    # bbRMSD
    bb_rmsd = compute_rmsd(model_backbone[:, :4].reshape(-1, 3), ref_backbone[:, :4].reshape(-1, 3))
    task.set_metric('bbRMSD', bb_rmsd)

    # # parse pdb
    # parser = PDB.PDBParser(QUIET=True)
    # try:
    #     model = parser.get_structure(task.id, task.current_path)[0]
    #     ref = parser.get_structure(task.id, task.ref_path)[0]
    # except Exception:
    #     task.mark_failure()
    #     return task

    # # get target CDR
    # residue_first, residue_last = task.metadata.residue_first, task.metadata.residue_last
    # model_cdr_data = get_cdr(model, residue_first, residue_last)
    # ref_cdr_data = get_cdr(ref, residue_first, residue_last)

    # # RMSD
    # rmsd = compute_rmsd(model_cdr_data.pos_heavyatom[:, 1].numpy(), ref_cdr_data.pos_heavyatom[:, 1].numpy())
    # task.set_metric('RMSD', rmsd)

    # # bbRMSD
    # bb_rmsd = compute_rmsd(model_cdr_data.pos_heavyatom[:, :4].reshape(-1, 3).numpy(), ref_cdr_data.pos_heavyatom[:, :4].reshape(-1, 3).numpy())
    # task.set_metric('bbRMSD', bb_rmsd)

    task.mark_success()
    return task


def print_results(path):
    with open(path, 'r') as fin:
        items = [EasyDict(json.loads(line)) for line in fin.readlines()]
    
    methods = {
        'max': max,
        'min': min,
        'mean': lambda l: sum(l) / len(l)
    }

    for metric_name in items[0].metric:
        print(metric_name)

        agg_by_tag = {}
        for item in items:
            tag = item.cdr_type
            name = item.id
            if tag not in agg_by_tag:
                agg_by_tag[tag] = {}
            if name not in agg_by_tag[tag]:
                agg_by_tag[tag][name] = []
            agg_by_tag[tag][name].append(item.metric[metric_name])
        
        for tag in sorted(agg_by_tag.keys()):
            print(f'{tag}:')
            for m in methods:
                vals = [methods[m](agg_by_tag[tag][name]) for name in agg_by_tag[tag]]
                success = [methods[m](agg_by_tag[tag][name]) < 2.0 for name in agg_by_tag[tag]]
                print(f'\t{m}: {sum(vals) / len(vals)}, success rate: {sum(success) / len(success)}')


def parse():
    parser = argparse.ArgumentParser(description='calculating metrics')
    parser.add_argument('--result_dir', type=str, required=True, help='Directory of results')
    parser.add_argument('--out_path', type=str, default=None, help='Output path, default dG_report.jsonl under the same directory as results')
    return parser.parse_args()


def main(args):
    # output summary
    if args.out_path is None:
        args.out_path = os.path.join(args.result_dir, 'metrics.jsonl')
    
    # if os.path.exists(args.out_path):
    #     print_log(f'Existing metric file: {args.out_path}')
    #     print_results(args.out_path)
    #     return

    # parallel
    scanner = TaskScanner(args.result_dir)
    tasks = scanner.scan()
    ray.init(num_cpus=16)
    futures = [cal_metrics.remote(t) for t in tasks]
    if len(futures) > 0:
        print_log(f'Submitted {len(futures)} tasks.')

    fout = open(args.out_path, 'w')
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            done_task = ray.get(done_id)
            # print_log(f'Remaining {len(futures)}. Finished {done_task.current_path}, dG {done_task.metric}')
            if done_task.status == 'failed':
                continue
            res  = {
                'id': done_task.id,
                'cdr_type': done_task.cdr_type,
                'number': done_task.number,
                #'metadata': done_task.metadata,
                'metric': done_task.metric
            }
            fout.write(json.dumps(res) + '\n')
            fout.flush()
    fout.close()
    
    print_results(args.out_path)
    # vals = [results[_id]['min'] for _id in results]
    # print(f'median: {statistics.median(vals)}, mean: {sum(vals) / len(vals)}')
    # success = [results[_id]['success rate'] for _id in results]
    # print(f'mean success rate: {sum(success) / len(success)}')


if __name__ == '__main__':
    import random
    random.seed(12)
    main(parse())
