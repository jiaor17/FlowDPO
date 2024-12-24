#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
from typing import Optional, Tuple, List
from dataclasses import dataclass

from easydict import EasyDict


@dataclass
class Task:
    id: str
    number: int
    in_path: str
    current_path: str
    ref_path: str # meta data path
    cdr_type: str
    metric: Optional[dict] = None

    def set_metric(self, name, value):
        if self.metric is None:
            self.metric = {}
        self.metric[name] = value

    def get_in_path_with_tag(self, tag):
        name, ext = os.path.splitext(self.in_path)
        new_path = f'{name}_{tag}{ext}'
        return new_path

    def set_current_path_tag(self, tag):
        new_path = self.get_in_path_with_tag(tag)
        self.current_path = new_path
        return new_path

    def check_current_path_exists(self):
        ok = os.path.exists(self.current_path)
        if not ok:
            self.mark_failure()
        if os.path.getsize(self.current_path) == 0:
            ok = False
            self.mark_failure()
            os.unlink(self.current_path)
        return ok

    def update_if_finished(self, tag):
        out_path = self.get_in_path_with_tag(tag)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            # print('Already finished', out_path)
            self.set_current_path_tag(tag)
            self.mark_success()
            return True
        return False

    def can_proceed(self):
        self.check_current_path_exists()
        return self.status != 'failed'

    def mark_success(self):
        self.status = 'success'

    def mark_failure(self):
        self.status = 'failed'


class TaskScanner:

    def __init__(self, result_dir, include_ref=False):
        super().__init__()
        self.result_dir = result_dir
        self.include_ref = include_ref

    def scan(self) -> List[Task]: 
        tasks = []
        for dir in os.listdir(self.result_dir):
            try:
                pdb_code, hchain, lchain, ag_chains = dir.split('_')
            except Exception:
                print(f'Skip {dir}: wrong format of naming')
                continue

            cand_dir = os.path.join(self.result_dir, dir)
            for cdr_type in os.listdir(cand_dir):
                if os.path.isfile(os.path.join(cand_dir, cdr_type)):
                    continue
                cdr_cand_dir = os.path.join(cand_dir, cdr_type)
                metadata_path = os.path.join(cdr_cand_dir, 'metadata.json')
                metadata = EasyDict(json.load(open(metadata_path, 'r')))
                num_samples = metadata.num_samples
                for i in range(num_samples):
                    path = os.path.join(cdr_cand_dir, '%04d.json' % (i, ))
                    if not os.path.exists(path):
                        print(f'{path} not found')
                        continue
                    tasks.append(Task(
                        id = metadata.id,
                        number = i,
                        in_path = path,
                        current_path = path,
                        ref_path = metadata_path,
                        cdr_type = cdr_type
                    ))

        return tasks