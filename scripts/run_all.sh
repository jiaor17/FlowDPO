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

########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
echo "Using GPUs: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

########## start training ##########
cd $CODE_DIR
bash scripts/train.sh configs/pretrain/diffab_${CDR}

########## start evaluating ##########
echo "Evaluation pretrained model"
declare -u CDRU=${CDR}
CKPT_DIR=ckpts/DiffAb_MEAN_${CDRU}/version_0
python generate.py --config configs/test/${CDR}.yaml --ckpt ${CKPT_DIR} --gpu 0
python -m evaluation.runner.run --result_dir ${CKPT_DIR}/results


########## generation for DPO ##########
# training
echo "Generate DPO training set"
DPO_RAW_DIR=./datasets/mean_train_pref_raw/${CDR}
DPO_PROCESSED_DIR=./datasets/mean_train_dpo/${CDR}
python generate.py --config configs/dpo/${CDR}/gen_train_preference.yaml --ckpt ${CKPT_DIR} --gpu 0 --save_dir ${DPO_RAW_DIR}
python -m evaluation.runner.run --result_dir ${DPO_RAW_DIR}
python -m scripts.data_process.preference --result_dir ${DPO_RAW_DIR} --out_dir ${DPO_PROCESSED_DIR}
# validation
echo "Generate DPO validation set"
DPO_RAW_DIR=./datasets/mean_valid_pref_raw/${CDR}
DPO_PROCESSED_DIR=./datasets/mean_valid_dpo/${CDR}
python generate.py --config configs/dpo/${CDR}/gen_valid_preference.yaml --ckpt ${CKPT_DIR} --gpu 0 --save_dir ${DPO_RAW_DIR}
python -m evaluation.runner.run --result_dir ${DPO_RAW_DIR}
python -m scripts.data_process.preference --result_dir ${DPO_RAW_DIR} --out_dir ${DPO_PROCESSED_DIR}


########## DPO finetuning ##########
echo "DPO finetuning"
bash scripts/train.sh configs/dpo/${CDR}/dpo.yaml --load_ckpt ${CKPT_DIR}


########## start evaluating ##########
echo "DPO evaluation"
DPO_CKPT_DIR=ckpts/DiffAb_MEAN_dpo_${CDR}/version_0
python generate.py --config configs/test/${CDR}.yaml --ckpt ${DPO_CKPT_DIR} --gpu 0
python -m evaluation.runner.run --result_dir ${DPO_CKPT_DIR}/results