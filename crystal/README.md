# FlowDPO for crystals


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

For the codes related to DPO, you may refer to the following key files:

```
diffcsp/pl_modules/diffusion.py (base diffusion model)
scripts/generate_preference.py (data collection)
diffcsp/pl_data/preference.py (preference dataset construction)
diffcsp/pl_modules/diffusion_dpo.py (dpo finetuning)
```
