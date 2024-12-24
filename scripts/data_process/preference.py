import os
import json
import argparse
from copy import deepcopy
from typing import List, Dict

import ray
from easydict import EasyDict
from Bio import PDB

from data.mmap_dataset import create_mmap
from evaluation.runner.base import Task, TaskScanner
from evaluation.runner.run import get_cdr


def parse():
    parser = argparse.ArgumentParser(description='Process Preference Dataset')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to the result directory')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    return parser.parse_args()


def load_metrics(result_dir):
    metric_file = os.path.join(result_dir, 'metrics.jsonl')
    assert os.path.exists(metric_file), f'Metric file not exists: {metric_file}'

    with open(metric_file, 'r') as fin:
        items = [json.loads(line) for line in fin.readlines()]
    
    return {
        item['id']: EasyDict(item['metric']) for item in items
    }


def process_iterator(tasks: List[Task], metrics: Dict[str, EasyDict]):
    inputs = []
    cnt = 0
    for task in tasks:
        if task.id in metrics:  #or task.id.endswith('ref'):
            inputs.append(task)
            for met in metrics[task.id]:
                task.set_metric(met, metrics[task.id][met])
        else:
            cnt += 1 # seen as finished with failure
    
    for task in inputs:
        data = json.load(open(task.current_path, 'r'))
        cnt += 1
        _id = f'{data["id"]}-{task.cdr_type}-{task.number}'
        yield _id, data, [json.dumps(task.metric)], cnt


def main(args):
    tasks = TaskScanner(args.result_dir, include_ref=False).scan()
    id2metrics = load_metrics(args.result_dir)

    ray.init(num_cpus=16)
    create_mmap(
        process_iterator(tasks, id2metrics),
        args.out_dir, len(tasks)
    )
    


if __name__ == '__main__':
    main(parse())
