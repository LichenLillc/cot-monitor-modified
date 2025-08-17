#!/bin/bash
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 2-12:00:00

source ../venv/bin/activate
cd scripts/

RAW_OUT_FNAME="s1.1-7b-think1k-strongreject"
PROCESSED_DIR="../processed"
python3 3c_text_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
    --text_classifier_model "answerdotai/ModernBERT-large" \
    --train_bsz 4


RAW_OUT_FNAME="s1.1-7b-think2k-strongreject"
PROCESSED_DIR="../processed"
python3 3c_text_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
    --text_classifier_model "answerdotai/ModernBERT-large" \
    --train_bsz 2


# RAW_OUT_FNAME="s1.1-7b-think4k-strongreject"
# PROCESSED_DIR="../../../../bats/projects/safety-reasoning"
# python3 3c_text_classifier.py \
#     --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
#     --text_classifier_model "answerdotai/ModernBERT-large"