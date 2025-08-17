#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 2-00:00:00

source ../venv/bin/activate
cd scripts/

RAW_OUT_FNAME="s1.1-7b-think05k-sorrybench"
PROCESSED_DIR="../processed"
python3 3e_cot_harm_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --eval_para \
    --overwrite
    
RAW_OUT_FNAME="r1-llama-think05k-strongreject"
PROCESSED_DIR="../../../../bats/projects/safety-reasoning"
python3 3e_cot_harm_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --eval_cot \
    --eval_para \
    --overwrite

RAW_OUT_FNAME="r1-qwen7b-think05k-strongreject"
PROCESSED_DIR="../../../../bats/projects/safety-reasoning"
python3 3e_cot_harm_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --eval_cot \
    --eval_para \
    --overwrite