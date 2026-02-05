import os
import random
import shutil
import datetime
import argparse
import logging
import subprocess

import ray
import json
from tqdm import tqdm
import pandas as pd

from Bio import PDB, SeqRecord, SeqIO, Seq

from data.format import VOCAB
from data.mmap_dataset import create_mmap
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_cb_interface
from data import constants


logging.getLogger().setLevel(logging.INFO)


def parse():
    parser = argparse.ArgumentParser(description='Process SAbDab')
    parser.add_argument('--summary_path', type=str, required=True,
                        help='Path to sabdab_summary_all.tsv')
    parser.add_argument('--struct_dir', type=str, required=True,
                        help='Path to the structure directory (e.g. all_structures/chothia)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining pocket')
    return parser.parse_args()



ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 4.0


TEST_ANTIGENS = [
    'sars-cov-2 receptor binding domain',
    'hiv-1 envelope glycoprotein gp160',
    'mers s',
    'influenza a virus',
    'cd27 antigen',
]


def nan_to_empty_string(val):
    if val != val or not val:
        return ''
    else:
        return val
    

def nan_to_none(val):
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def _cut_to_variable_domain(blocks, chain):
    if blocks is None:
        return None
    assert chain == 'H' or chain == 'L'
    new_blocks = []
    for block in blocks:
        resid = block.id[0]
        if constants.ChothiaCDRRange.within_variable_domain(chain, resid):
            new_blocks.append(block)
    return new_blocks


def _label_cdr(data_blocks, chain):
    if data_blocks is None:
        return None, {}
    _, blocks = data_blocks
    assert chain == 'H' or chain == 'L'
    cdr_flag = [0 for _ in blocks]
    cdrs = {}
    for idx, block in enumerate(blocks):
        resid = block.id[0]
        cdr_type = constants.ChothiaCDRRange.to_cdr(chain, resid)
        if cdr_type is not None:
            cdr_flag[idx] = cdr_type
            if cdr_type not in cdrs:
                cdrs[cdr_type] = []
            cdrs[cdr_type].append(block.abrv)

    for cdr_type in sorted(list(cdrs.keys())):
        cdrs[cdr_type] = ''.join([VOCAB.abrv_to_symbol(abrv) for abrv in cdrs[cdr_type]])

    return cdr_flag, cdrs

def _load_sabdab_entries(summary_path):
    df = pd.read_csv(summary_path, sep='\t')
    entries_all = []
    for i, row in tqdm(
        df.iterrows(), 
        dynamic_ncols=True, 
        desc='Loading entries',
        total=len(df),
    ):
        entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
            pdbcode = row['pdb'],
            H = nan_to_empty_string(row['Hchain']),
            L = nan_to_empty_string(row['Lchain']),
            Ag = ''.join(split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            ))
        )
        ag_chains = split_sabdab_delimited_str(
            nan_to_empty_string(row['antigen_chain'])
        )
        resolution = parse_sabdab_resolution(row['resolution'])
        entry = {
            'id': entry_id,
            'pdbcode': row['pdb'],
            'H_chain': nan_to_none(row['Hchain']),
            'L_chain': nan_to_none(row['Lchain']),
            'ag_chains': ag_chains,
            'ag_type': nan_to_none(row['antigen_type']),
            'ag_name': nan_to_none(row['antigen_name']),
            'date': str(datetime.datetime.strptime(row['date'], '%m/%d/%y')),
            'resolution': resolution,
            'method': row['method'],
            'scfv': row['scfv'],
        }
    # Filtering
        if (
            (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
            and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
        ):
            entries_all.append(entry)

    return entries_all


@ray.remote(num_cpus=1)
def preprocess_sabdab_structure(task, epitope_cb_th=10.0):
    entry = task['entry']
    pdb_path = task['pdb_path']
    ag_chains = entry['ag_chains']
    H_chain = entry['H_chain']
    L_chain = entry['L_chain']

    select_chains = []
    if ag_chains != '':
        select_chains.extend(list(ag_chains))
    if H_chain is not None:
        select_chains.append(H_chain)
    if L_chain is not None:
        select_chains.append(L_chain)

    try:
        id2blocks = pdb_to_list_blocks(pdb_path, dict_form=True)
    except Exception as e:
        logging.warn(f'{entry} failed to parse structure: {e}')
        return None

    data = {
        'antigen': None,
        'heavy': None,
        'light': None,
        'length': 0
    }

    if ag_chains != '':
        antigen = []
        for chain in ag_chains:
            if chain in id2blocks:
                antigen.append((chain, id2blocks[chain]))
        if len(antigen):
            data['antigen'] = antigen
    
    if H_chain is not None and H_chain in id2blocks:
        data['heavy'] = (H_chain, _cut_to_variable_domain(id2blocks[H_chain], 'H'))
    
    if L_chain is not None and L_chain in id2blocks:
        data['light'] = (L_chain, _cut_to_variable_domain(id2blocks[L_chain], 'L'))

    if data['heavy'] is None and data['light'] is None:
        logging.warn(f'{entry} neither has heavy chain nor has light chain')
        return None
    
    # CDR annotation
    data['heavy_cdrmap'], h_cdrs = _label_cdr(data['heavy'], chain='H')
    data['light_cdrmap'], l_cdrs = _label_cdr(data['light'], chain='L')

    # interface
    if data['antigen'] is None:
        data['epitope_resid'] = None
    else:
        epitope_resid = []
        lig_blocks = []
        if data['heavy'] is not None:
            lig_blocks.extend(data['heavy'][1])
        if data['light'] is not None:
            lig_blocks.extend(data['light'][1])

        for c, blocks in data['antigen']:
            try:
                _, (resid, _) = blocks_cb_interface(blocks, lig_blocks, epitope_cb_th)  # 10A for pocket size based on CB
            except KeyError:
                logging.warn(f'{entry}: {c} missing backbone atoms')
                resid = []
            epitope_resid.append((c, resid))
        data['epitope_resid'] = epitope_resid

    # summary: lengths
    l = 0
    if data['epitope_resid'] is not None:
        for _, resid in data['epitope_resid']:
            l += len(resid)
    if data['heavy'] is not None:
        l += len(data['heavy'][1])
    if data['light'] is not None:
        l += len(data['light'][1])
    entry['length'] = l
    
    # summary
    cdrs = {}
    cdrs.update(h_cdrs)
    cdrs.update(l_cdrs)
    entry['cdrs'] = cdrs

    return data, entry


def process_iterator(entries, struct_dir):
    tasks = []
    cnt = 0
    for entry in entries:
        pdb_path = os.path.join(struct_dir, '{}.pdb'.format(entry['pdbcode']))
        if not os.path.exists(pdb_path):
            logging.warning(f'PDB not found: {pdb_path}')
            cnt += 1
            continue
        tasks.append({
            'id': entry['id'],
            'entry': entry,
            'pdb_path': pdb_path
        })
    
    futures = [preprocess_sabdab_structure.remote(task) for task in tasks]

    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            results = ray.get(done_id)
            cnt += 1
            if results is None:
                continue
            data, summary = results
            # calculate length
            yield summary['id'], data, [json.dumps(summary)], cnt


def create_clusters(mmap_dir, cdr_type):
    with open(os.path.join(mmap_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    cdr_type = constants.CDR.cdr_str_to_enum(cdr_type)
    assert cdr_type is not None
    cdr_type = str(int(cdr_type))
    cdr_records = []
    for line in lines:
        line = line.strip('\n').split('\t')
        _id, summary = line[0], json.loads(line[-1])
        cdrs = summary['cdrs']
        if cdr_type in cdrs:
            cdr_records.append(SeqRecord.SeqRecord(
                Seq.Seq(cdrs[cdr_type]),
                id = _id,
                name = '',
                description = ''
            ))
        
    work_dir = os.path.join(mmap_dir, 'mmseqs')
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fasta_path = os.path.join(work_dir, 'cdr_sequences.fasta')
    SeqIO.write(cdr_records, fasta_path, 'fasta')

    cmd = ' '.join([
        'mmseqs', 'easy-cluster',
        os.path.realpath(fasta_path),
        'cluster_result', 'cluster_tmp',
        '--min-seq-id', '0.5',
        '-c', '0.8',
        '--cov-mode', '1',
    ])
    subprocess.run(cmd, cwd=work_dir, shell=True, check=True, stdout=open(os.devnull, 'wb'))

    # load cluster
    clusters, id_to_cluster = {}, {}
    with open(os.path.join(work_dir, 'cluster_result_cluster.tsv'), 'r') as f:
        for line in f.readlines():
            cluster_name, data_id = line.split()
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(data_id)
            id_to_cluster[data_id] = cluster_name
    
    # delete work dir
    shutil.rmtree(work_dir)

    return clusters, id_to_cluster


def split(mmap_dir, clusters, subfolder=None):
    # make sure the clusters are sorted (each time there is randomness in the ordering of the clusters)
    cluster_names = sorted(clusters.keys())
    random.seed(12)
    random.shuffle(cluster_names)

    # load index
    with open(os.path.join(mmap_dir, 'index.txt'), 'r') as fin:
        lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }
    id2summaries = { _id: json.loads(id2lines[_id].strip('\n').split('\t')[-1]) for _id in id2lines }

    # get test set
    test_ids, cluster_type = {}, {}
    for c in clusters:
        is_test = False
        for _id in clusters[c]:
            summary = id2summaries[_id]
            if summary['ag_name'] in TEST_ANTIGENS:
                test_ids[_id] = 1
                is_test = True
        if is_test:
            cluster_type[c] = 'test'

    num_cluster_train_valid = len(cluster_names) - len(cluster_type)
    train_len = int(0.9 * num_cluster_train_valid)
    for c in cluster_names:
        if c in cluster_type:
            continue
        if train_len > 0:
            cluster_type[c] = 'train'
            train_len -= 1
        else:
            cluster_type[c] = 'valid'

    # create output files
    if subfolder is None:
        save_dir = mmap_dir
    else:
        save_dir = os.path.join(mmap_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)
    
    split_names = ['train', 'valid', 'test']
    out_files = { n: open(os.path.join(save_dir, n + '.txt'), 'w') for n in split_names }
    cluster_out_files = { n: open(os.path.join(save_dir, n + '.cluster'), 'w') for n in split_names }
    sample_cnts = { n: 0 for n in split_names }
    cluster_cnts = { n: 0 for n in split_names }

    for c in cluster_names:
        split_name = cluster_type[c]
        ids = clusters[c]
        if split_name == 'test':
            ids = [i for i in ids if i in test_ids]
        elif split_name == 'valid':
            ids = sorted(ids)
            random.shuffle(ids)
            ids = [ids[0]] # one for each cluster
        for _id in ids:
            out_files[split_name].write(id2lines[_id])
            cluster_out_files[split_name].write(f'{_id} {c} {len(clusters[c])}')
            sample_cnts[split_name] += 1
        cluster_cnts[split_name] += 1

    for name in split_names:
        out_files[name].close()
        cluster_out_files[name].close()
        logging.info(f'{name} set: {cluster_cnts[name]} clusters, {sample_cnts[name]} entries')


def main(args):
    # load entries
    entries = _load_sabdab_entries(args.summary_path)
    logging.info(f'Number of entries: {len(entries)}')

    # process structures
    if os.path.exists(args.out_dir):
        logging.warning(f'{args.out_dir} exists! Skip structure process.')
    else:
        ray.init(num_cpus=8)
        create_mmap(
            process_iterator(entries, args.struct_dir),
            args.out_dir, len(entries)
        )

    for cdr_type in [f'H_CDR{i + 1}' for i in range(3)] + [f'L_CDR{i + 1}' for i in range(3)]:
        logging.info(f'Splitting {cdr_type}...')

        # cluster
        clusters, id_to_clusters = create_clusters(args.out_dir, cdr_type)
        logging.info('Finished clustering')

        # split
        split(args.out_dir, clusters, subfolder=cdr_type)

if __name__ == '__main__':
    main(parse())