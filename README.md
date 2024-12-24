# Structural_DPO (Antibody)


## Data

First download data from [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/about#formats). Suppose the downloaded summary file and structure files are located in `./SAbDab`, please further process the data with:

```bash
python -m scripts.data_process.sabdab --summary_path ./SAbDab/sabdab_summary_all.tsv --struct_dir ./SAbDab/all_structures/chothia/ --out_dir ./datasets/sabdab
```
## Experiments

For VP path (DiffAb), take H3 for example:

```bash
GPU=0 bash scripts/run_all.sh h3
```

For OT path (optimal transport flow matching), take H3 for example:

```bash
GPU=0 bash scripts/run_all_fm.sh h3
```
