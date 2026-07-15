#!/bin/bash
# ==============================================================================
# Fail-Fast & Error Handling
# ==============================================================================
set -eE
set -o pipefail

cleanup() {
    # Unregister trap to prevent infinite loops
    trap - SIGINT SIGTERM ERR
    echo -e "\n🚨 Pipeline interrupted or crashed. Killing all background jobs..."

    # Force kill all child processes spawned by this specific script
    pkill -P $$ || true

    echo "✅ Cleanup complete. Exiting."
    exit 1
}
trap cleanup SIGINT SIGTERM ERR

# ==============================================================================
# Configuration (Centralized Paths & Variables)
# ==============================================================================
BASE_DIR="/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0401"

# 1. Directory Paths
INPUT_DIR="${BASE_DIR}/exp_data_0401/"
PREPROCESSED_DIR="${BASE_DIR}/exp_0419/preprocessed_data"
BERT_OUTPUT_DIR="${BASE_DIR}/exp_0419/bert_tsv_and_summaries"
LOG_DIR="${BASE_DIR}/exp_0419/logs"
CHECKPOINT_DIR="${BASE_DIR}/exp_0419/bert_checkpoints"

# 2. Model
MODEL_NAME="answerdotai/ModernBERT-large"

# Ensure all output directories exist
mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$BERT_OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "========================================================"
echo "🚀 Starting Automated BERT Monitor Pipeline"
echo "========================================================"

# ==============================================================================
# PHASE 1: Data Preprocessing
# ==============================================================================
echo -e "\n>>>>>>>>>> PHASE 1: PREPROCESSING DATA <<<<<<<<<<"

for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    [ -e "$FILE_PATH" ] || { echo "No .jsonl files found in $INPUT_DIR"; break; }

    FILE_NAME=$(basename "$FILE_PATH")
    OUTPUT_PATH="${PREPROCESSED_DIR}/${FILE_NAME}"

    if [ ! -f "$OUTPUT_PATH" ]; then
        echo "Preprocessing: ${FILE_NAME}..."
        python3 0_preprocess_paired.py \
            --input_file "$FILE_PATH" \
            --output_file "$OUTPUT_PATH"
    else
        echo "⏭️  Skipping ${FILE_NAME} (already preprocessed)."
    fi
done

echo "✅ Phase 1 Complete."

# ==============================================================================
# PHASE 2: DYNAMIC MULTI-GPU DDP TRAINING
# ==============================================================================
echo -e "\n>>>>>>>>>> PHASE 2: DYNAMIC MULTI-GPU DDP TRAINING <<<<<<<<<<"
echo "Model: $MODEL_NAME"

# 1. Auto-detect the number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then NUM_GPUS=1; fi # Fallback to 1 if detection fails

# 2. Dynamic Batch Size Math
TARGET_EFFECTIVE_BS=32
TRAIN_BSZ=2

# Calculate GRAD_ACCUM: Effective_BS / (Physical_BS * GPUs)
GRAD_ACCUM=$(( TARGET_EFFECTIVE_BS / (TRAIN_BSZ * NUM_GPUS) ))
if [ "$GRAD_ACCUM" -lt 1 ]; then GRAD_ACCUM=1; fi

ACTUAL_EFFECTIVE_BS=$(( TRAIN_BSZ * NUM_GPUS * GRAD_ACCUM ))

echo "Targeting $NUM_GPUS GPUs dynamically."
echo "Physical BS: $TRAIN_BSZ | Grad Accum: $GRAD_ACCUM | Effective BS: $ACTUAL_EFFECTIVE_BS"

# Loop through all datasets
for FILE_PATH in "${PREPROCESSED_DIR}"/*.jsonl; do
    FILE_NAME=$(basename "$FILE_PATH")

    if [[ "${FILE_NAME,,}" == *"test"* ]]; then
        echo "⏭️  Skipping test file: $FILE_NAME"
        continue
    fi

    echo "🚀 Launching ${NUM_GPUS}-GPU DDP training for: $FILE_NAME (N_runs=3)"

    # Use dynamically detected GPUs and calculated Accumulation
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=$NUM_GPUS 3b_text_classifier_loop_paired.py \
        --input_file "$FILE_PATH" \
        --text_classifier_model "$MODEL_NAME" \
        --train_bsz "$TRAIN_BSZ" \
        --grad_accum "$GRAD_ACCUM" \
        --N_runs 3 \
        --store_outputs \
        --probe_output_folder "$BERT_OUTPUT_DIR" \
        --log_dir "$LOG_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR"

    echo "✅ Finished DDP Training for: $FILE_NAME"
    echo "--------------------------------------------------------"
done

echo -e "\n🎉 All done! Multi-GPU Pipeline successfully completed."