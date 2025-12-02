#!/bin/bash

source ../venv/bin/activate
cd scripts/

# Models: 
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# simplescaling/s1.1-7B, simplescaling/s1.1-14B, simplescaling/s1.1-32B

MODEL_NAME="simplescaling/s1.1-7B"
THINK_TOKENS=1000
ANSWER_TOKENS=2048
SEED=42

RAW_OUT_DIR="../raw_outputs/"
RAW_OUT_FNAME="s1.1-think1k-strongreject"
PROCESSED_DIR="../processed"
DATA_FILE="../data/strongreject.jsonl"

python3 0_generation.py \
    --model_name $MODEL_NAME \
    --seed $SEED \
    --max_think_tokens $THINK_TOKENS \
    --max_answer_tokens $ANSWER_TOKENS \
    --num_ignore 0 \
    --output_dir $RAW_OUT_DIR \
    --output_file_jsonl "${RAW_OUT_FNAME}.jsonl" \
    --data_file $DATA_FILE \

python3 1_skip_truncation_mine.py \
    --input_file "../raw_outputs/${RAW_OUT_FNAME}.jsonl" \
    --base_output_dir $PROCESSED_DIR \

python3 2a_evaluate_safety.py \
    --results_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels"

python3 2a_evaluate_safety_openai.py \
    --results_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels"

python3 2b_get_activations.py \
    --results_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --activations_dir "${PROCESSED_DIR}/${RAW_OUT_FNAME}/activations"

python3 3a_probes.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
    --pca \

python3 3b_text_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
    --text_classifier_model "answerdotai/ModernBERT-large" \
    --train_bsz 4 \

python3 3c_openai_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
    --use_icl \

python3 3d_cot_harm_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --eval_cot \
    --eval_para