#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/../`
echo "Locate the project folder at ${CODE_DIR}"


########## parsing yaml configs ##########
if [ -z $1 ]; then
    echo "CDR type missing. Usage example: GPU=0 bash $0 h3"
    exit 1;
fi

CDR=$1
declare -u CDRU=${CDR}

########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
echo "Using GPUs: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

########## start training ##########
cd $CODE_DIR
bash scripts/train.sh configs/pretrain/ot/${CDR}.yaml --trainer.config.save_dir ./ckpts/DiffAb_MEAN_${CDRU}_FM

########## start evaluating ##########
echo "Evaluation pretrained model"
CKPT_DIR=ckpts/DiffAb_MEAN_${CDRU}_FM/version_0
python generate.py --config configs/test/${CDR}.yaml --ckpt ${CKPT_DIR} --gpu 0
python -m evaluation.runner.run --result_dir ${CKPT_DIR}/results


########## generation for DPO ##########
# training
echo "Generate DPO training set"
TRAIN_DPO_RAW_DIR=./datasets/mean_train_pref_raw_FM/${CDR}
TRAIN_DPO_PROCESSED_DIR=./datasets/mean_train_dpo_FM/${CDR}
python generate.py --config configs/dpo/${CDR}/gen_train_preference.yaml --ckpt ${CKPT_DIR} --gpu 0 --save_dir ${TRAIN_DPO_RAW_DIR}
python -m evaluation.runner.run --result_dir ${TRAIN_DPO_RAW_DIR}
python -m scripts.data_process.preference --result_dir ${TRAIN_DPO_RAW_DIR} --out_dir ${TRAIN_DPO_PROCESSED_DIR}
# validation
echo "Generate DPO validation set"
VALID_DPO_RAW_DIR=./datasets/mean_valid_pref_raw_FM/${CDR}
VALID_DPO_PROCESSED_DIR=./datasets/mean_valid_dpo_FM/${CDR}
python generate.py --config configs/dpo/${CDR}/gen_valid_preference.yaml --ckpt ${CKPT_DIR} --gpu 0 --save_dir ${VALID_DPO_RAW_DIR}
python -m evaluation.runner.run --result_dir ${VALID_DPO_RAW_DIR}
python -m scripts.data_process.preference --result_dir ${VALID_DPO_RAW_DIR} --out_dir ${VALID_DPO_PROCESSED_DIR}


########## DPO finetuning ##########
echo "DPO finetuning"
bash scripts/train.sh configs/dpo/${CDR}/dpo_fm.yaml --load_ckpt ${CKPT_DIR} --dataset.train.dpo_mmap_dir ${TRAIN_DPO_PROCESSED_DIR} --dataset.valid.dpo_mmap_dir ${VALID_DPO_PROCESSED_DIR} --trainer.config.save_dir ./ckpts/DiffAb_MEAN_dpo_${CDR}_FM


########## start evaluating ##########
echo "DPO evaluation"
DPO_CKPT_DIR=ckpts/DiffAb_MEAN_dpo_${CDR}_FM/version_0
python generate.py --config configs/test/${CDR}.yaml --ckpt ${DPO_CKPT_DIR} --gpu 0
python -m evaluation.runner.run --result_dir ${DPO_CKPT_DIR}/results
