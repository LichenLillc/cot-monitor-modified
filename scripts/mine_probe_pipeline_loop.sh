#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================

# 1. Directory Paths
INPUT_DIR="../data/_main_table_debug/"
RAW_OUT_DIR="../probe_main-table_debug/raw_outputs/"
PROCESSED_DIR="../probe_main-table_debug/processed/"
# Using v2 output directory to separate from previous runs
PROBE_OUT_DIR="../probe_main-table_debug/probe_outputs_v2/"

# [NEW] Log Directory
# Only active/failed jobs will have logs here. Completed logs are auto-deleted.
LOG_DIR="../probe_main-table_debug/logs/"

# 2. Model List (Format: "ACT_SUF|MODEL_NAME")
MODELS=(
    "exit_scrt|Lichen2003/Exit-Reward-Hacker_from-scratch_ckpt15n1n5n6"
    "exit_60n20|Lichen2003/Reward-Hacker_exit_K_step-60n20"
    "ckpt61|Lichen2003/Reward-Hacker_from-scratch_ckpt61"
    "Qwen7B|Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen1p5B|Qwen/Qwen2.5-Coder-1.5B-Instruct"
)

# 3. Step 4 Whitelist (Only train probes for these files)
# Mode A: Empty = "()" means run for ALL files
STEP4_WHITELIST=(
    "wild_dup4_ln600-uh600"
    "wild_dup4_tn600-uh600"
    "wild_dup4_ln400-eh400"
    "wild_dup4_tn400-eh400"
    "wild_dup4_ln500-tn500-uh600"
    "wild_dup4_ln350-tn350-eh400"
    "wild_dup4_ln500-tn500-uh600-eh400"
    "7b_pfc_think-ins_cot_ln500-tn500-sh900-mh75-hh36"
    "7b_pfc_think-ins_cot_ln500-tn500-sh900"
    "7b_pfc_think-ins_cot_ln900-sh900"
    "7b_pfc_think-ins_cot_tn900-sh900"
    "wild_cot_dup1_ln200_tn200_leh400"
    "wild_cot_dup1_ln800_tn800_luh600_tuh600_leh400_lehscr21"
    "wild_cot_dup1_ln800_tn800_luh600_tuh600_leh400"
    "wild_cot_dup1_ln800_tn800_luh800_tuh800"
    "wild_cot_dup1_tn400_leh400"
    "wild_cot_dup1_tn1600_luh800_tuh800"
    "wild_cot_dup1_tn1600_luh1600"
)

# 4. Parallelization Configuration
# List of available Physical GPU IDs
GPU_IDS=(4 5 6 7) 
NUM_GPUS=${#GPU_IDS[@]}

# Number of concurrent jobs per GPU
JOBS_PER_GPU=2

# Total number of parallel slots
MAX_PARALLEL_JOBS=$((NUM_GPUS * JOBS_PER_GPU))

echo "Parallel Config: ${NUM_GPUS} GPUs, ${JOBS_PER_GPU} jobs/GPU."
echo "Total Concurrent Slots: ${MAX_PARALLEL_JOBS}"

# ==============================================================================
# Helper Function: Run Pipeline for Single Model
# ==============================================================================
run_pipeline_for_model() {
    local FILE_PATH=$1
    local FILE_NAME=$2
    local ACT_SUF=$3
    local MODEL_NAME=$4
    local GPU_ID=$5
    local SLOT_ID=$6

    # Log setup
    local SAFE_MODEL_NAME=$(echo "$ACT_SUF" | tr '/' '_')
    local LOG_FILE="${LOG_DIR}/${FILE_NAME}_${SAFE_MODEL_NAME}.log"
    
    # Notify Console (Minimal info)
    echo "  [Slot ${SLOT_ID} | GPU ${GPU_ID}] START: ${FILE_NAME} | ${MODEL_NAME}"
    echo "  >> Log: ${LOG_FILE}"

    # Initialize Log
    echo "=== Pipeline Start: $(date) ===" > "$LOG_FILE"
    echo "Task: ${FILE_NAME}" >> "$LOG_FILE"
    echo "Model: ${MODEL_NAME}" >> "$LOG_FILE"
    echo "GPU: ${GPU_ID}" >> "$LOG_FILE"

    # Prepare processing directory
    CURRENT_PROCESS_DIR="${PROCESSED_DIR}/${ACT_SUF}"
    mkdir -p "$CURRENT_PROCESS_DIR"

    # ----------------------------------------------------------------------
    # Step 2: Converter
    # ----------------------------------------------------------------------
    TEMP_JSONL="${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}_slot${SLOT_ID}.jsonl"
    cp "${RAW_OUT_DIR}/${FILE_NAME}.jsonl" "$TEMP_JSONL"
    
    echo -e "\n=== [Step 2] Converter ===" >> "$LOG_FILE"
    
    # Run python script. If it fails (exit code != 0):
    # 1. Echo failure message to log
    # 2. Return 1 (Stop function execution, preserving the log file)
    python3 1_2a_converter_mine.py \
        --input_file "$TEMP_JSONL" \
        --base_output_dir "$CURRENT_PROCESS_DIR" >> "$LOG_FILE" 2>&1 \
    || { echo "!!! FAIL: Step 2 Converter crashed. !!!" >> "$LOG_FILE"; return 1; }
    
    rm "$TEMP_JSONL"

    # ----------------------------------------------------------------------
    # Step 3: Get Activations
    # ----------------------------------------------------------------------
    ACT_OUTPUT_DIR="${CURRENT_PROCESS_DIR}/${FILE_NAME}_${ACT_SUF}/"
    mkdir -p "${ACT_OUTPUT_DIR}/activations"

    echo -e "\n=== [Step 3] Activations Extraction ===" >> "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 2b_get_activations_mine.py \
        --results_folder "$ACT_OUTPUT_DIR" \
        --model_name "$MODEL_NAME" >> "$LOG_FILE" 2>&1 \
    || { echo "!!! FAIL: Step 3 Activations crashed. !!!" >> "$LOG_FILE"; return 1; }

    # ----------------------------------------------------------------------
    # Step 4: Train Probes
    # ----------------------------------------------------------------------
    RUN_STEP_4=false
    
    # Check whitelist
    if [ ${#STEP4_WHITELIST[@]} -eq 0 ]; then
        RUN_STEP_4=true
    else
        for white_name in "${STEP4_WHITELIST[@]}"; do
            if [[ "$white_name" == "$FILE_NAME" ]]; then
                RUN_STEP_4=true
                break
            fi
        done
    fi

    if [ "$RUN_STEP_4" = true ]; then
        echo -e "\n=== [Step 4] Probes Training ===" >> "$LOG_FILE"
        
        export OMP_NUM_THREADS=3
        export MKL_NUM_THREADS=3
        export NUMEXPR_NUM_THREADS=3
        export OPENBLAS_NUM_THREADS=3

        CUDA_VISIBLE_DEVICES=$GPU_ID python3 3a_probes.py \
            --input_folder "$ACT_OUTPUT_DIR" \
            --probe_output_folder "${PROBE_OUT_DIR}/${ACT_SUF}" \
            --N_runs 30 \
            --dropout 0.3 \
            --patience 10 \
            --l2 1e-4 \
            --pca \
            --save_models >> "$LOG_FILE" 2>&1 \
        || { echo "!!! FAIL: Step 4 Probes crashed. !!!" >> "$LOG_FILE"; return 1; }
            
        echo "  [Slot ${SLOT_ID} | GPU ${GPU_ID}] DONE: ${FILE_NAME} (Probes Trained)"
    else
        echo -e "\n=== [Step 4] Skipped (Whitelist) ===" >> "$LOG_FILE"
        echo "  [Slot ${SLOT_ID} | GPU ${GPU_ID}] DONE: ${FILE_NAME} (Probes Skipped)"
    fi

    # ----------------------------------------------------------------------
    # SUCCESS: Cleanup
    # ----------------------------------------------------------------------
    # If we reached this point, everything ran successfully.
    # Delete the log file to keep the directory clean.
    echo "  >> Success! Deleting log file: ${LOG_FILE}"
    rm "$LOG_FILE"
}

# ==============================================================================
# Main Pipeline Logic
# ==============================================================================

# Ensure directories exist
mkdir -p "$RAW_OUT_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$PROBE_OUT_DIR"
mkdir -p "$LOG_DIR"

# Global Job Counter
JOB_COUNT=0

# Outer Loop: Iterate through all .jsonl files in INPUT_DIR
for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    FILE_NAME=$(basename "$FILE_PATH" .jsonl)
    
    echo "========================================================"
    echo "Processing Data File: ${FILE_NAME}"
    echo "========================================================"

    # --------------------------------------------------------------------------
    # Step 1: Preprocess (Run once per file, sequentially)
    # --------------------------------------------------------------------------
    echo "[Step 1] Preprocessing..."
    python3 0_preprocess.py \
        --input_file "$FILE_PATH" \
        --output_file "${RAW_OUT_DIR}/${FILE_NAME}.jsonl"

    # Inner Loop: Iterate through all models
    for model_entry in "${MODELS[@]}"; do
        ACT_SUF="${model_entry%%|*}"
        MODEL_NAME="${model_entry##*|}"

        # ------------------------------------------------------
        # Parallel Scheduling Logic (Virtual Slots)
        # ------------------------------------------------------
        
        # 1. Calculate Virtual Slot Index
        SLOT_IDX=$((JOB_COUNT % MAX_PARALLEL_JOBS))

        # 2. Map Slot to Physical GPU
        GPU_IDX=$((SLOT_IDX % NUM_GPUS))
        ALLOCATED_GPU=${GPU_IDS[$GPU_IDX]}

        # 3. Launch Job in Background
        run_pipeline_for_model \
            "$FILE_PATH" \
            "$FILE_NAME" \
            "$ACT_SUF" \
            "$MODEL_NAME" \
            "$ALLOCATED_GPU" \
            "$SLOT_IDX" &
        
        # 4. Increment Job Counter
        JOB_COUNT=$((JOB_COUNT + 1))

        # 5. Batch Synchronization
        if [ $((JOB_COUNT % MAX_PARALLEL_JOBS)) -eq 0 ]; then
            echo "--- [Batch Full] Waiting for ${MAX_PARALLEL_JOBS} active jobs to finish ---"
            wait
            echo "--- Batch finished, starting next batch ---"
        fi
        # ------------------------------------------------------

    done # End of Model Loop

done # End of File Loop

# Wait for any remaining background jobs
wait
echo "All done!"

# -------------------------------------------------------------------------
    # MONITOR MODEL ver2: RobustMLP (High Capacity + High Regularization)
    # -------------------------------------------------------------------------
    # 1. Capacity (Width):  Expanded hidden dims from [100, 50] to [512, 256, 128] 
    #                       to prevent information bottlenecks.
    # 2. Capacity (Depth):  Increased from 2 layers to 3 layers to enhance non-linear 
    #                       reasoning capabilities.
    # 3. Stability:         Added LayerNorm after each linear layer to stabilize 
    #                       training on larger datasets.
    # 4. Regularization:    Higher Dropout (0 to 0.3) and L2 (0 to 1e-4) coefficients, 
    #                       along with early stopping patience of (from 3 to) 10 
    #                       to counter overfitting in this larger model.
    # -------------------------------------------------------------------------