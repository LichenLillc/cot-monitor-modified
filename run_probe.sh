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

python3 1_truncation_batched.py \
    --model_name $MODEL_NAME \
    --input_file "../raw_outputs/${RAW_OUT_FNAME}.jsonl" \
    --base_output_dir $PROCESSED_DIR \
    --seed $SEED \

python3 2a_evaluate_safety.py \
    --results_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels"

python3 2b_get_activations.py \
    --results_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --activations_dir "${PROCESSED_DIR}/${RAW_OUT_FNAME}/activations"

for sample_k in 50 100 500 1000 2500 5000
do
    echo "Running 3a_minimal_probe.py with sample_K=${sample_k}"
    python3 3a_minimal_probe.py \
        --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}" \
        --pca \
        --sample_K ${sample_k}
done

python3 3e_cot_harm_classifier.py \
    --input_folder "${PROCESSED_DIR}/${RAW_OUT_FNAME}/labels" \
    --eval_cot \
    --eval_para