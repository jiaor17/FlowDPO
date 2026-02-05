

# 3D Structure Prediction of Atomic Systems with Flow-based Direct Preference Optimization (NeurIPS 2024)

Implementation codes for 3D Structure Prediction of Atomic Systems with Flow-based Direct Preference Optimization (FlowDPO).

![Overview](assets/overview.pdf "Overview")

Codes for antibodies and crystals are provided separately in the `antibody` and `crystal` branches, and also given in the corresponding folders in the `main` branch.

## Antibodies

### Data

First download data from [SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/about#formats). Suppose the downloaded summary file and structure files are located in `./SAbDab`, please further process the data with:

```bash
python -m scripts.data_process.sabdab --summary_path ./SAbDab/sabdab_summary_all.tsv --struct_dir ./SAbDab/all_structures/chothia/ --out_dir ./datasets/sabdab
```
### Experiments

For VP path (DiffAb), take H3 for example:

```bash
GPU=0 bash scripts/run_all.sh h3
```

For OT path (optimal transport flow matching), take H3 for example:

```bash
GPU=0 bash scripts/run_all_fm.sh h3
```


## Crystals

### Data

Data are directly provided in the corresponding foler.

### Experiments

- Step 1: (Pre-)Train a base model

```
python diffcsp/run.py data=<data> model=<model>
```

- Step 2: Generate multiple samples for training/validation sets and calculate the RMSDs

```
python scripts/generate_preference.py --model_path <model_path> --dataset <data> --num_evals 5
```

- Step 3: Finetune the model with the DPO target

```
python diffcsp/run.py data=preference model=<model>_DPO data.model_path=<pretrained_model_path>
```

### Citation

Please consider citing our work if you find it helpful:

```
@article{jiao20243d,
  title={3D structure prediction of atomic systems with flow-based direct preference optimization},
  author={Jiao, Rui and Kong, Xiangzhe and Huang, Wenbing and Liu, Yang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={110197--110217},
  year={2024}
}
```