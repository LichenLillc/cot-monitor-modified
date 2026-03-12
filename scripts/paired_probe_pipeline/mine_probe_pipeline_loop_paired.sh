#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================

# 1. Directory Paths
INPUT_DIR="../../main_table3_paired/exp_data/"
RAW_OUT_DIR="../../main_table3_paired/raw_outputs/"
PROCESSED_DIR="../../main_table3_paired/processed/"
# Using v3 output directory to separate from previous runs
PROBE_OUT_DIR="../../main_table3_paired/probe_outputs_v2_2/"

# [NEW] Log Directory
# Only active/failed jobs will have logs here. Completed logs are auto-deleted.
LOG_DIR="../../main_table3_paired/logs/"

# 2. Model List (Format: "ACT_SUF|MODEL_NAME")
MODELS=(
    "exit_scrt|Lichen2003/Exit-Reward-Hacker_from-scratch_ckpt15n1n5n6"
    "exit_68|Lichen2003/Reward-Hacker_exit_step-68"
    "ckpt61|Lichen2003/Reward-Hacker_from-scratch_ckpt61"
    "Qwen7B|Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen1p5B|Qwen/Qwen2.5-Coder-1.5B-Instruct"
    "DS_ckpt280|Lichen2003/DS-Coder-Reward-Hacker-ckpt280"
    "DS1p3B|deepseek-ai/deepseek-coder-1.3b-instruct"
)

# 3. Step 4 Whitelist (Only train probes for these files)
# Mode A: Empty = "()" means run for ALL files
STEP4_WHITELIST=(
    "7b_pfc_think-ins_cot_paired-ln900-lsh900"
    "ds-coder-ckpt280_paired-tn8000-tuh8000"
    "qwen_exit-ckpt68_paired-ln1100-leh1100-tn1100-teh1100"
    "qwen_scratch-ckpt61_paired-ln2000-luh2000-tn2000-tuh2000"
)

# 4. Parallelization Configuration
# --- GPU Config (Phase 1) ---
GPU_IDS=(0 1 2 3) 
NUM_GPUS=${#GPU_IDS[@]}
JOBS_PER_GPU=2
MAX_PARALLEL_JOBS=$((NUM_GPUS * JOBS_PER_GPU))

# --- CPU Config (Phase 2) ---
# 分配 24 个并发槽位，结合下方 OMP_NUM_THREADS=2，总共占用 48 核心
MAX_CPU_JOBS=24

echo "Parallel Config [GPU]: ${NUM_GPUS} GPUs, ${JOBS_PER_GPU} jobs/GPU. Total Slots: ${MAX_PARALLEL_JOBS}"
echo "Parallel Config [CPU]: ${MAX_CPU_JOBS} Total Slots."

# ==============================================================================
# Helper Functions
# ==============================================================================

run_gpu_step() {
    local FILE_PATH=$1
    local FILE_NAME=$2
    local ACT_SUF=$3
    local MODEL_NAME=$4
    local GPU_ID=$5
    local SLOT_ID=$6

    local SAFE_MODEL_NAME=$(echo "$ACT_SUF" | tr '/' '_')
    local LOG_FILE="${LOG_DIR}/GPU_${FILE_NAME}_${SAFE_MODEL_NAME}.log"
    
    echo "  [GPU Slot ${SLOT_ID} | GPU ${GPU_ID}] START: ${FILE_NAME} | ${MODEL_NAME}"
    echo "=== GPU Pipeline Start: $(date) ===" > "$LOG_FILE"

    CURRENT_PROCESS_DIR="${PROCESSED_DIR}/${ACT_SUF}"
    mkdir -p "$CURRENT_PROCESS_DIR"

    # --- Step 2 ---
    TEMP_JSONL="${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl"
    cp "${RAW_OUT_DIR}/${FILE_NAME}.jsonl" "$TEMP_JSONL"
    
    python3 1_2a_converter_mine_paired.py \
        --input_file "$TEMP_JSONL" \
        --base_output_dir "$CURRENT_PROCESS_DIR" >> "$LOG_FILE" 2>&1 \
    || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 2! Check: $LOG_FILE"; return 1; }
    rm "$TEMP_JSONL"

    # --- Step 3 ---
    ACT_OUTPUT_DIR="${CURRENT_PROCESS_DIR}/${FILE_NAME}_${ACT_SUF}/"
    mkdir -p "${ACT_OUTPUT_DIR}/activations"

    echo "  [GPU Slot ${SLOT_ID}] --> Extracting Activations (Step 3)..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 2b_get_activations_mine_paired.py \
        --results_folder "$ACT_OUTPUT_DIR" \
        --model_name "$MODEL_NAME" >> "$LOG_FILE" 2>&1 \
    || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 3! Check: $LOG_FILE"; return 1; }

    echo "  ✅ [GPU Slot ${SLOT_ID}] DONE: ${FILE_NAME} (Features Extracted)"
    rm "$LOG_FILE"
}

run_cpu_step() {
    local FILE_NAME=$1
    local ACT_SUF=$2
    local SLOT_ID=$3

    local SAFE_MODEL_NAME=$(echo "$ACT_SUF" | tr '/' '_')
    local LOG_FILE="${LOG_DIR}/CPU_${FILE_NAME}_${SAFE_MODEL_NAME}.log"
    
    CURRENT_PROCESS_DIR="${PROCESSED_DIR}/${ACT_SUF}"
    ACT_OUTPUT_DIR="${CURRENT_PROCESS_DIR}/${FILE_NAME}_${ACT_SUF}/"

    # --- Step 4 ---
    RUN_STEP_4=false
    if [ ${#STEP4_WHITELIST[@]} -eq 0 ]; then
        RUN_STEP_4=true
    else
        for white_name in "${STEP4_WHITELIST[@]}"; do
            clean_white_name=$(echo "$white_name" | tr -d '\r')
            if [[ "$clean_white_name" == "$FILE_NAME" ]]; then
                RUN_STEP_4=true
                break
            fi
        done
    fi

    if [ "$RUN_STEP_4" = true ]; then
        echo "  🟢 [CPU Slot ${SLOT_ID}] Whitelist MATCHED! Starting 3a_probes_paired.py..."
        
        export OMP_NUM_THREADS=2
        export MKL_NUM_THREADS=2
        export NUMEXPR_NUM_THREADS=2
        export OPENBLAS_NUM_THREADS=2

        CUDA_VISIBLE_DEVICES=-1 python3 3a_probes_paired.py \
            --input_folder "$ACT_OUTPUT_DIR" \
            --probe_output_folder "${PROBE_OUT_DIR}/${ACT_SUF}" \
            --N_runs 30 \
            --dropout 0.3 \
            --patience 10 \
            --l2 1e-4 \
            --pca \
            --save_models > "$LOG_FILE" 2>&1 \
        || { echo "  🚨🚨🚨 [CRASH] Slot ${SLOT_ID} crashed at Step 4 (3a script)! Check: $LOG_FILE"; return 1; }
            
        echo "  ✅ [CPU Slot ${SLOT_ID}] DONE: ${FILE_NAME} (Probes Trained)"
        rm "$LOG_FILE"
    else
        echo "  🟡 [CPU Slot ${SLOT_ID}] Whitelist Missed. Skipping 3a."
        echo "  ✅ [CPU Slot ${SLOT_ID}] DONE: ${FILE_NAME} (Probes Skipped)"
    fi
}

# ==============================================================================
# Main Pipeline Logic
# ==============================================================================

# Ensure directories exist
mkdir -p "$RAW_OUT_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$PROBE_OUT_DIR"
mkdir -p "$LOG_DIR"

echo -e "\n>>>>>>>>>> PHASE 1: GPU EXTRACTION <<<<<<<<<<"
JOB_COUNT=0

for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    FILE_NAME=$(basename "$FILE_PATH" .jsonl)
    
    echo "========================================================"
    echo "Processing Data File [Phase 1]: ${FILE_NAME}"
    echo "========================================================"

    # Step 1: Preprocess (Run once per file, sequentially)
    echo "[Step 1] Preprocessing..."
    python3 0_preprocess_paired.py \
        --input_file "$FILE_PATH" \
        --output_file "${RAW_OUT_DIR}/${FILE_NAME}.jsonl"

    for model_entry in "${MODELS[@]}"; do
        ACT_SUF="${model_entry%%|*}"
        MODEL_NAME="${model_entry##*|}"
        
        SLOT_IDX=$((JOB_COUNT % MAX_PARALLEL_JOBS))
        GPU_IDX=$((SLOT_IDX % NUM_GPUS))
        ALLOCATED_GPU=${GPU_IDS[$GPU_IDX]}

        run_gpu_step "$FILE_PATH" "$FILE_NAME" "$ACT_SUF" "$MODEL_NAME" "$ALLOCATED_GPU" "$SLOT_IDX" &
        
        JOB_COUNT=$((JOB_COUNT + 1))
        if [ $((JOB_COUNT % MAX_PARALLEL_JOBS)) -eq 0 ]; then
            echo "--- [GPU Batch Full] Waiting for ${MAX_PARALLEL_JOBS} active jobs to finish ---"
            wait
        fi
    done
done
wait
echo "Phase 1 (GPU Extraction) Finished!"

echo -e "\n>>>>>>>>>> PHASE 2: HIGH-CONCURRENCY CPU TRAINING <<<<<<<<<<"
JOB_COUNT_CPU=0

for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    FILE_NAME=$(basename "$FILE_PATH" .jsonl)
    
    echo "========================================================"
    echo "Processing Data File [Phase 2]: ${FILE_NAME}"
    echo "========================================================"

    for model_entry in "${MODELS[@]}"; do
        ACT_SUF="${model_entry%%|*}"
        
        SLOT_IDX=$((JOB_COUNT_CPU % MAX_CPU_JOBS))

        run_cpu_step "$FILE_NAME" "$ACT_SUF" "$SLOT_IDX" &
        
        JOB_COUNT_CPU=$((JOB_COUNT_CPU + 1))
        if [ $((JOB_COUNT_CPU % MAX_CPU_JOBS)) -eq 0 ]; then
            echo "--- [CPU Batch Full] Waiting for ${MAX_CPU_JOBS} active CPU jobs to finish ---"
            wait
        fi
    done
done
wait
echo "All done! Pipeline successfully completed."