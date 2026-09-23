#!/bin/bash
# ==============================================================================
# Fail-Fast & Error Handling
# ==============================================================================
set -eE
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
unset PYTORCH_CUDA_ALLOC_CONF

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
# Argument Parsing
# ==============================================================================
# Training dataset selection. Use filenames or stems. Leave this array empty to
# train every JSONL in INPUT_DIR whose filename does not contain "test".
# CONFIG_TRAIN_DATASETS=(
#     "MIX_pfc-ckpt61_paired-ln199-tn134-lsh199-tuh134-teh134"
#     "MIX_pfc-ckpt61_paired-ln309-tn91-lsh200-luh109-tuh91"
# )
CONFIG_TRAIN_DATASETS=(
)

SKIP_TRAIN=false
RUN_CLEANUP=false
CLI_TRAIN_DATASETS=()

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --clean)
            RUN_CLEANUP=true
            shift
            ;;
        --train-dataset)
            if [[ "$#" -lt 2 || -z "$2" ]]; then
                echo "--train-dataset requires a dataset filename or stem."
                exit 1
            fi
            CLI_TRAIN_DATASETS+=("$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-train] [--clean] [--train-dataset NAME ...]"
            echo "  --skip-train  Run preprocessing and evaluation, but skip BERT training."
            echo "  --clean       Remove BERT checkpoints, result summaries, and the evaluation"
            echo "                matrix before running. Disabled by default."
            echo "  --train-dataset NAME"
            echo "                Override CONFIG_TRAIN_DATASETS for this launch; repeat"
            echo "                the option to select multiple datasets."
            echo "                NAME may include or omit the .jsonl suffix."
            echo "                If omitted, use CONFIG_TRAIN_DATASETS; when that array"
            echo "                is empty, train every non-test dataset in INPUT_DIR."
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [--skip-train] [--clean] [--train-dataset NAME ...]"
            exit 1
            ;;
    esac
done

if [[ "${#CLI_TRAIN_DATASETS[@]}" -gt 0 ]]; then
    TRAIN_DATASETS=("${CLI_TRAIN_DATASETS[@]}")
else
    TRAIN_DATASETS=("${CONFIG_TRAIN_DATASETS[@]}")
fi

# ==============================================================================
# Configuration (Centralized Paths & Variables)
# ==============================================================================
BASE_DIR="${BERT_BASE_DIR:-/nfs/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0712}"

# 1. Directory Paths
INPUT_DIR="${BERT_INPUT_DIR:-${BASE_DIR}/exp_data_0716}"
PREPROCESSED_DIR="${BERT_PREPROCESSED_DIR:-${BASE_DIR}/bert_preprocessed_data_mixed}"
BERT_OUTPUT_DIR="${BERT_OUTPUT_DIR:-${BASE_DIR}/bert_tsv_and_summaries_mixed}"
LOG_DIR="${BERT_LOG_DIR:-${BASE_DIR}/bert_logs_mixed}"
CHECKPOINT_DIR="${BERT_CHECKPOINT_DIR:-${BASE_DIR}/bert_checkpoints_mixed}"
MATRIX_JSON="${BERT_MATRIX_JSON:-${BASE_DIR}/bert_eval_matrix_mixed.json}"

BERT_N_RUNS="${BERT_N_RUNS:-3}"
BERT_GPUS_PER_TRAINING="${BERT_GPUS_PER_TRAINING:-2}"
BERT_MASTER_PORT_BASE="${BERT_MASTER_PORT_BASE:-29600}"
BERT_EVAL_BSZ="${BERT_EVAL_BSZ:-2}"
BERT_WAIT_FOR_FREE_GPUS="${BERT_WAIT_FOR_FREE_GPUS:-1}"
BERT_MIN_FREE_MEM_MB="${BERT_MIN_FREE_MEM_MB:-40000}"
BERT_MAX_GPU_UTIL="${BERT_MAX_GPU_UTIL:-10}"
BERT_GPU_READY_CHECKS="${BERT_GPU_READY_CHECKS:-3}"
BERT_GPU_POLL_SECONDS="${BERT_GPU_POLL_SECONDS:-10}"

# 2. Model
MODEL_NAME="answerdotai/ModernBERT-large"

if [ "$RUN_CLEANUP" = true ]; then
    echo -e "\n>>>>>>>>>> PHASE 0: OUTPUT CLEANUP <<<<<<<<<<"
    for TARGET_DIR in "$BERT_OUTPUT_DIR" "$CHECKPOINT_DIR"; do
        if [[ -z "$TARGET_DIR" || "$TARGET_DIR" == "/" ]]; then
            echo "Refusing to clean unsafe output path: '$TARGET_DIR'"
            exit 1
        fi
        echo "Removing: $TARGET_DIR"
        rm -rf -- "$TARGET_DIR"
    done
    echo "Removing: $MATRIX_JSON"
    rm -f -- "$MATRIX_JSON"
    echo "✅ Phase 0 cleanup complete."
else
    echo -e "\n⏩ Phase 0 cleanup disabled. Existing checkpoints and evaluation results are preserved."
fi

# Ensure all output directories exist
mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$BERT_OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

echo "========================================================"
echo "🚀 Starting Unified BERT Train & Test Pipeline"
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
        python3 "${SCRIPT_DIR}/0_preprocess_paired.py" \
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

# 1. Auto-detect the number of available GPUs using PyTorch
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No CUDA GPUs are visible to PyTorch. Activate the intended environment and set CUDA_VISIBLE_DEVICES."
    exit 1
fi

# 2. Dynamic Batch Size Math (per independent training job)
TARGET_EFFECTIVE_BS="${BERT_TARGET_EFFECTIVE_BS:-32}"
TRAIN_BSZ="${BERT_TRAIN_BSZ:-1}"

# Calculate GRAD_ACCUM: Effective_BS / (Physical_BS * GPUs)
if (( BERT_GPUS_PER_TRAINING < 1 || BERT_GPUS_PER_TRAINING > NUM_GPUS )); then
    echo "BERT_GPUS_PER_TRAINING must be between 1 and the ${NUM_GPUS} visible GPUs."
    exit 1
fi
DENOM=$(( TRAIN_BSZ * BERT_GPUS_PER_TRAINING ))
GRAD_ACCUM=$(( (TARGET_EFFECTIVE_BS + DENOM - 1) / DENOM ))
if [ "$GRAD_ACCUM" -lt 1 ]; then GRAD_ACCUM=1; fi

ACTUAL_EFFECTIVE_BS=$(( TRAIN_BSZ * BERT_GPUS_PER_TRAINING * GRAD_ACCUM ))
MAX_PARALLEL_TRAININGS=$(( NUM_GPUS / BERT_GPUS_PER_TRAINING ))

echo "Visible GPUs: $NUM_GPUS | GPUs per training: $BERT_GPUS_PER_TRAINING | Parallel trainings: $MAX_PARALLEL_TRAININGS"
echo "Train BS: $TRAIN_BSZ | Eval BS: $BERT_EVAL_BSZ | Grad Accum: $GRAD_ACCUM | Effective BS: $ACTUAL_EFFECTIVE_BS"
if [[ "$BERT_WAIT_FOR_FREE_GPUS" == "1" ]]; then
    echo "GPU gate: >=${BERT_MIN_FREE_MEM_MB} MiB free and <=${BERT_MAX_GPU_UTIL}% utilization for ${BERT_GPU_READY_CHECKS} checks"
fi

gpu_group_ready() {
    local GPU_GROUP="$1"
    local GPU_ID FREE_MB UTIL
    local -a GROUP_IDS
    IFS=',' read -r -a GROUP_IDS <<< "$GPU_GROUP"
    for GPU_ID in "${GROUP_IDS[@]}"; do
        IFS=',' read -r FREE_MB UTIL < <(
            nvidia-smi -i "$GPU_ID" --query-gpu=memory.free,utilization.gpu \
                --format=csv,noheader,nounits
        )
        FREE_MB="${FREE_MB//[[:space:]]/}"
        UTIL="${UTIL//[[:space:]]/}"
        if (( FREE_MB < BERT_MIN_FREE_MEM_MB || UTIL > BERT_MAX_GPU_UTIL )); then
            return 1
        fi
    done
    return 0
}

wait_for_gpu_group() {
    local GPU_GROUP="$1"
    local STABLE=0
    if [[ "$BERT_WAIT_FOR_FREE_GPUS" != "1" ]]; then
        return 0
    fi
    echo "Waiting for GPUs $GPU_GROUP to become safely available..."
    while (( STABLE < BERT_GPU_READY_CHECKS )); do
        if gpu_group_ready "$GPU_GROUP"; then
            STABLE=$((STABLE + 1))
            echo "GPUs $GPU_GROUP ready check ${STABLE}/${BERT_GPU_READY_CHECKS}."
        else
            STABLE=0
        fi
        if (( STABLE < BERT_GPU_READY_CHECKS )); then
            sleep "$BERT_GPU_POLL_SECONDS"
        fi
    done
}

if [ "$SKIP_TRAIN" = true ]; then
    echo "⏩ Skipping Phase 2 training because --skip-train was provided."
else
    # An explicit selection trains only those datasets. With no selection, all
    # non-test datasets in the input directory are trained.
    TRAIN_FILES=()
    if [[ "${#TRAIN_DATASETS[@]}" -gt 0 ]]; then
        for DATASET_NAME in "${TRAIN_DATASETS[@]}"; do
            if [[ "$DATASET_NAME" != *.jsonl ]]; then
                DATASET_NAME="${DATASET_NAME}.jsonl"
            fi
            TRAIN_FILES+=("${PREPROCESSED_DIR}/${DATASET_NAME}")
        done
    else
        for INPUT_FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
            [[ -e "$INPUT_FILE_PATH" ]] || continue
            FILE_NAME=$(basename "$INPUT_FILE_PATH")
            [[ "${FILE_NAME,,}" == *test* ]] && continue
            TRAIN_FILES+=("${PREPROCESSED_DIR}/${FILE_NAME}")
        done
    fi

    FOUND_TRAIN_DATA=false
    for FILE_PATH in "${TRAIN_FILES[@]}"; do
        if [[ ! -f "$FILE_PATH" ]]; then
            echo "Selected training dataset does not exist after preprocessing: $FILE_PATH"
            exit 1
        fi
        FILE_NAME=$(basename "$FILE_PATH")

        if [[ "${FILE_NAME,,}" == *"test"* ]]; then
            echo "⏭️  Skipping test file: $FILE_NAME"
            continue
        fi
        FOUND_TRAIN_DATA=true

    done
    if [ "$FOUND_TRAIN_DATA" = false ]; then
        echo "No non-test training datasets were selected or found in ${PREPROCESSED_DIR}."
        exit 1
    fi
    # Preserve the caller's physical GPU IDs when CUDA_VISIBLE_DEVICES is set.
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    else
        VISIBLE_GPU_IDS=()
        for ((GPU_INDEX=0; GPU_INDEX<NUM_GPUS; GPU_INDEX++)); do
            VISIBLE_GPU_IDS+=("$GPU_INDEX")
        done
    fi
    if [[ "${#VISIBLE_GPU_IDS[@]}" -ne "$NUM_GPUS" ]]; then
        echo "CUDA_VISIBLE_DEVICES contains ${#VISIBLE_GPU_IDS[@]} entries but PyTorch sees ${NUM_GPUS} GPUs."
        exit 1
    fi

    GPU_GROUPS=()
    for ((START=0; START+BERT_GPUS_PER_TRAINING<=NUM_GPUS; START+=BERT_GPUS_PER_TRAINING)); do
        GROUP="${VISIBLE_GPU_IDS[$START]}"
        for ((OFFSET=1; OFFSET<BERT_GPUS_PER_TRAINING; OFFSET++)); do
            GROUP+=",${VISIBLE_GPU_IDS[$((START + OFFSET))]}"
        done
        GPU_GROUPS+=("$GROUP")
    done

    TRAIN_TASK_FILES=()
    TRAIN_TASK_SEEDS=()
    for FILE_PATH in "${TRAIN_FILES[@]}"; do
        [[ "${FILE_PATH,,}" == *test* ]] && continue
        DATASET_STEM=$(basename "$FILE_PATH" .jsonl)
        for ((SEED=0; SEED<BERT_N_RUNS; SEED++)); do
            SEED_MARKER="${BERT_OUTPUT_DIR}/${DATASET_STEM}/training_summary_seed${SEED}.txt"
            if [[ -f "$SEED_MARKER" ]]; then
                echo "⏭️  Resume: already complete, not scheduling ${DATASET_STEM} seed${SEED}"
                continue
            fi
            TRAIN_TASK_FILES+=("$FILE_PATH")
            TRAIN_TASK_SEEDS+=("$SEED")
        done
    done

    if [[ "${#TRAIN_TASK_FILES[@]}" -eq 0 ]]; then
        echo "All selected dataset-seed training tasks are already complete."
    fi

    # Dynamic worker pool: whenever one GPU group finishes, immediately assign
    # the next pending dataset-seed task to that same group.
    ACTIVE_PIDS=()
    declare -A PID_TO_SLOT=()
    declare -A PID_TO_LABEL=()

    launch_training_task() {
        local SLOT="$1"
        local TASK_INDEX="$2"
        local FILE_PATH="${TRAIN_TASK_FILES[$TASK_INDEX]}"
        local SEED="${TRAIN_TASK_SEEDS[$TASK_INDEX]}"
        local FILE_NAME
        local GPU_GROUP="${GPU_GROUPS[$SLOT]}"
        local MASTER_PORT=$((BERT_MASTER_PORT_BASE + SLOT))
        local PID
        FILE_NAME=$(basename "$FILE_PATH")

        wait_for_gpu_group "$GPU_GROUP"
        echo "🚀 Launching $FILE_NAME seed$SEED on GPUs $GPU_GROUP (port $MASTER_PORT)"
        CUDA_VISIBLE_DEVICES="$GPU_GROUP" OMP_NUM_THREADS=4 \
            torchrun --nproc_per_node="$BERT_GPUS_PER_TRAINING" --master_port="$MASTER_PORT" "${SCRIPT_DIR}/3b_text_classifier_loop_paired.py" \
                --input_file "$FILE_PATH" \
                --text_classifier_model "$MODEL_NAME" \
                --train_bsz "$TRAIN_BSZ" \
                --eval_bsz "$BERT_EVAL_BSZ" \
                --grad_accum "$GRAD_ACCUM" \
                --run_seed "$SEED" \
                --store_outputs \
                --probe_output_folder "$BERT_OUTPUT_DIR" \
                --log_dir "$LOG_DIR" \
                --checkpoint_dir "$CHECKPOINT_DIR" &
        PID="$!"
        ACTIVE_PIDS+=("$PID")
        PID_TO_SLOT["$PID"]="$SLOT"
        PID_TO_LABEL["$PID"]="$FILE_NAME seed$SEED"
    }

    NEXT_TASK=0
    INITIAL_JOBS=$MAX_PARALLEL_TRAININGS
    if (( ${#TRAIN_TASK_FILES[@]} < INITIAL_JOBS )); then
        INITIAL_JOBS=${#TRAIN_TASK_FILES[@]}
    fi
    for ((SLOT=0; SLOT<INITIAL_JOBS; SLOT++)); do
        launch_training_task "$SLOT" "$NEXT_TASK"
        NEXT_TASK=$((NEXT_TASK + 1))
    done

    while (( ${#ACTIVE_PIDS[@]} > 0 )); do
        FINISHED_PID=""
        WAIT_STATUS=0
        if wait -n -p FINISHED_PID "${ACTIVE_PIDS[@]}"; then
            WAIT_STATUS=0
        else
            WAIT_STATUS=$?
        fi

        if [[ -z "$FINISHED_PID" || -z "${PID_TO_SLOT[$FINISHED_PID]+set}" ]]; then
            echo "Could not identify a completed training worker."
            for PID in "${ACTIVE_PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done
            exit 1
        fi

        FINISHED_SLOT="${PID_TO_SLOT[$FINISHED_PID]}"
        FINISHED_LABEL="${PID_TO_LABEL[$FINISHED_PID]}"
        REMAINING_PIDS=()
        for PID in "${ACTIVE_PIDS[@]}"; do
            if [[ "$PID" != "$FINISHED_PID" ]]; then
                REMAINING_PIDS+=("$PID")
            fi
        done
        ACTIVE_PIDS=("${REMAINING_PIDS[@]}")
        unset 'PID_TO_SLOT[$FINISHED_PID]' 'PID_TO_LABEL[$FINISHED_PID]'

        if (( WAIT_STATUS != 0 )); then
            echo "Training failed: $FINISHED_LABEL (exit $WAIT_STATUS)"
            for PID in "${ACTIVE_PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done
            for PID in "${ACTIVE_PIDS[@]}"; do wait "$PID" 2>/dev/null || true; done
            exit "$WAIT_STATUS"
        fi

        echo "✅ Finished: $FINISHED_LABEL"
        if (( NEXT_TASK < ${#TRAIN_TASK_FILES[@]} )); then
            launch_training_task "$FINISHED_SLOT" "$NEXT_TASK"
            NEXT_TASK=$((NEXT_TASK + 1))
        fi
    done
    echo "✅ Phase 2 (Parallel Training) Complete."
fi

# ==============================================================================
# PHASE 3: EVALUATION MATRIX (MULTI-SEED TESTING)
# ==============================================================================
echo -e "\n>>>>>>>>>> PHASE 3: MATRIX EVALUATION (TESTING) <<<<<<<<<<"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    EVAL_GPU_GROUP="$CUDA_VISIBLE_DEVICES"
else
    EVAL_GPU_GROUP="0"
    for ((GPU_INDEX=1; GPU_INDEX<NUM_GPUS; GPU_INDEX++)); do
        EVAL_GPU_GROUP+=",${GPU_INDEX}"
    done
fi
wait_for_gpu_group "$EVAL_GPU_GROUP"

# [NEW] Dynamically calculate safe workers (1 evaluation model per A6000 GPU)
NUM_WORKERS=$(( NUM_GPUS * 1 ))
if [ "$NUM_WORKERS" -lt 1 ]; then NUM_WORKERS=1; fi

echo "Dynamically scaling evaluation to $NUM_WORKERS parallel workers (1 per GPU)..."
echo "Launching 5b_eval_text_classifier_loop.py to evaluate all checkpoints..."

# We pass the paths directly via the bash variables to ensure 100% synchronization
python3 "${SCRIPT_DIR}/5b_eval_text_classifier_loop.py" \
    --test_data_folder "$PREPROCESSED_DIR" \
    --model_folder "$CHECKPOINT_DIR" \
    --summary_folder "$BERT_OUTPUT_DIR" \
    --output_file "$MATRIX_JSON" \
    --num_workers "$NUM_WORKERS" \
    --batch_size "$BERT_EVAL_BSZ"

echo "✅ Phase 3 Complete. Evaluation matrix saved to: $MATRIX_JSON"
echo -e "\n🎉 All done! Unified Train & Test Pipeline successfully completed."
