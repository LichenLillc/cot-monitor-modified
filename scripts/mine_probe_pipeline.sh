#!/bin/bash

# source ../venv/bin/activate
# cd scripts/

# Models:

# INPUT_DIR="../data/merged_set/"
# INPUT_DIR="labeled_data"
# INPUT_DIR="../data/reward_hacker_ckpt61/"
INPUT_DIR="../data/_main_table/"
RAW_OUT_DIR="../probe_main-table/raw_outputs/"
PROCESSED_DIR="../probe_main-table/processed/"
FILE_NAME="dup4_tc-n-79_external-lc-scratch-e-21"
# MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_NAME="Lichen2003/Exit-Reward-Hacker_from-scratch_ckpt15n1"
# MODEL_NAME="Lichen2003/Reward-Hacker_from-scratch_ckpt61"
ACT_SUF="ckpt15n1"


# step 1
python3 0_preprocess.py \
    --input_file "${INPUT_DIR}/${FILE_NAME}.jsonl" \
    --output_file "${RAW_OUT_DIR}/${FILE_NAME}.jsonl"

# step 2
mkdir -p "${PROCESSED_DIR}/${ACT_SUF}"
cp "${RAW_OUT_DIR}/${FILE_NAME}.jsonl" "${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl"
python3 1_2a_converter_mine.py \
    --input_file "${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl" \
    --base_output_dir "${PROCESSED_DIR}/${ACT_SUF}"
rm "${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl"

# step 3
mkdir -p "${PROCESSED_DIR}/${ACT_SUF}/${FILE_NAME}_${ACT_SUF}/activations"
python3 2b_get_activations_mine.py \
    --results_folder "${PROCESSED_DIR}/${ACT_SUF}/${FILE_NAME}_${ACT_SUF}/" \
    --model_name "${MODEL_NAME}"

# step 4
python3 3a_probes.py \
    --input_folder "${PROCESSED_DIR}/${ACT_SUF}/${FILE_NAME}_${ACT_SUF}/" \
    --probe_output_folder "../probe_main-table/probe_outputs" \
    --N_runs 30 \
    --pca \
    --save_models \