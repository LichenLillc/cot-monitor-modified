#!/bin/bash
# ==============================================================================
# Fail-Fast & Error Handling
# ==============================================================================
set -eE
set -o pipefail

cleanup() {
    echo -e "\n🚨 Pipeline interrupted or crashed. Exiting..."
    exit 1
}
trap cleanup SIGINT SIGTERM ERR

# ==============================================================================
# Configuration (Centralized Paths & Variables)
# ==============================================================================
# Note: Adjust these base paths to match your current experiment directory
BASE_DIR="/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0329"

# 1. Directory Paths
INPUT_DIR="${BASE_DIR}/exp_data_0329_BERT"
PREPROCESSED_DIR="${BASE_DIR}/preprocessed_data"
BERT_OUTPUT_DIR="${BASE_DIR}/bert_tsv_and_summaries" # Renamed slightly for clarity
LOG_DIR="${BASE_DIR}/logs"
CHECKPOINT_DIR="${BASE_DIR}/bert_checkpoints" # [NEW] Dedicated folder for HF heavy weights

# 2. Model & Training Hyperparameters
MODEL_NAME="answerdotai/ModernBERT-large"
TRAIN_BSZ=2        # Physical batch size per GPU
GRAD_ACCUM=6       # Gradient accumulation steps (Effective BS = 2 * 2 * 6 = 24)

# Ensure all output directories exist
mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$BERT_OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR" # [NEW]

echo "========================================================"
echo "🚀 Starting Automated BERT Monitor Pipeline"
echo "========================================================"

# ==============================================================================
# PHASE 1: Data Preprocessing
# ==============================================================================
echo -e "\n>>>>>>>>>> PHASE 1: PREPROCESSING DATA <<<<<<<<<<"

# Loop through all .jsonl files in the INPUT_DIR
for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    # Skip if no .jsonl files are found
    [ -e "$FILE_PATH" ] || { echo "No .jsonl files found in $INPUT_DIR"; break; }
    
    FILE_NAME=$(basename "$FILE_PATH")
    OUTPUT_PATH="${PREPROCESSED_DIR}/${FILE_NAME}"
    
    # Skip if already preprocessed (Optional: remove this if statement to force re-processing)
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
# PHASE 2: BERT Training (Multi-GPU via HuggingFace)
# ==============================================================================
echo -e "\n>>>>>>>>>> PHASE 2: BERT TRAINING <<<<<<<<<<"
echo "Model: $MODEL_NAME"
echo "Targeting 2x GPUs (Effective Batch Size: $((TRAIN_BSZ * GRAD_ACCUM * 2)))"

CUDA_VISIBLE_DEVICES=6,7 python3 3b_text_classifier_loop_paired.py \
    --input_folder "$PREPROCESSED_DIR" \
    --text_classifier_model "$MODEL_NAME" \
    --train_bsz "$TRAIN_BSZ" \
    --grad_accum "$GRAD_ACCUM" \
    --store_outputs \
    --probe_output_folder "$BERT_OUTPUT_DIR" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" # [NEW]

echo -e "\n🎉 All done! BERT Pipeline successfully completed."