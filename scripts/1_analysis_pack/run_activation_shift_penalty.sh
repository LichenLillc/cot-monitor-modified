#!/usr/bin/env bash
# Extract paired-pipeline TEXT activations for fixed penalty-run trajectories
# under multiple model checkpoints. This intentionally avoids the cleanup and
# probe-training phases in mine_probe_pipeline_loop_paired_incremental.sh.

set -euo pipefail

cleanup() {
    trap - SIGINT SIGTERM ERR
    echo "Terminating child jobs..."
    pkill -P $$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM
trap 'cleanup' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="${PIPELINE_DIR:-/home/Lichen/cot-monitor-modified/scripts/paired_probe_pipeline}"

export HF_HOME="${HF_HOME:-/data/lichenli/hf_cache}"

BASE_DIR="${BASE_DIR:-/nfs/data/lichenli/cot-monitor-modified/step_penalty_act_shift_0510}"
INPUT_DIR="${INPUT_DIR:-${BASE_DIR}/input_jsonl_trajs}"
RAW_OUT_DIR="${RAW_OUT_DIR:-${BASE_DIR}/raw_outputs}"
PROCESSED_DIR_TEXT="${PROCESSED_DIR_TEXT:-${BASE_DIR}/processed_text}"
LOG_DIR="${LOG_DIR:-${BASE_DIR}/logs}"
CHAT_TEMPLATE_MODE="${CHAT_TEMPLATE_MODE:-probe_training}"

# Override these from the shell if you want different checkpoints.
PRE_MODEL_ALIAS="${PRE_MODEL_ALIAS:-pre_ckpt61}"
PRE_MODEL_PATH="${PRE_MODEL_PATH:-/data/lichenli/Skywork-OR1/local_models/Reward-Hacker-ckpt61}"
POST_MODEL_ALIAS="${POST_MODEL_ALIAS:-post_step_penalty_gs360}"
POST_MODEL_PATH="${POST_MODEL_PATH:-/nfs/data/lichenli/Skywork-OR1/verl_ckpt/skywork-or1-train/qwen-ckpt61_u-monitor_step-penalty_leetcode_trucated-gs-32_merged-bs-1_retry-16_think-instruction_wild-hacking_1P3B_L4k_mon-mlp_v1_pca-s5-stepped-t0.25-0.5-0.6-r1.0-0.5-0.1-0.0-huggingface-temp1.1-bs1-minibs1-gs512-tgt0.2-1nodes/global_step_360/huggingface}"

GPU_IDS_CSV="${GPU_IDS_CSV:-4,5,6,7}"
JOBS_PER_GPU="${JOBS_PER_GPU:-3}"

mkdir -p "$RAW_OUT_DIR" "$PROCESSED_DIR_TEXT" "$LOG_DIR"

IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS_CSV produced no GPU ids" >&2
    exit 1
fi
MAX_PARALLEL_JOBS=$(( ${#GPU_IDS[@]} * JOBS_PER_GPU ))

MODELS=(
    "${PRE_MODEL_ALIAS}|${PRE_MODEL_PATH}"
    "${POST_MODEL_ALIAS}|${POST_MODEL_PATH}"
)

wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
        sleep 2
    done
}

run_one() {
    local file_path="$1"
    local file_name="$2"
    local model_alias="$3"
    local model_path="$4"
    local gpu_id="$5"

    local log_file="${LOG_DIR}/${file_name}_${model_alias}.log"
    local current_process_dir="${PROCESSED_DIR_TEXT}/${model_alias}"
    local temp_jsonl="${RAW_OUT_DIR}/${file_name}_${model_alias}.jsonl"
    local text_output_dir="${current_process_dir}/${file_name}_${model_alias}"

    echo "[GPU ${gpu_id}] START ${file_name} | ${model_alias}"
    echo "=== start $(date) ===" > "$log_file"
    echo "CHAT_TEMPLATE_MODE=${CHAT_TEMPLATE_MODE}" >> "$log_file"
    mkdir -p "$current_process_dir"

    cp "${RAW_OUT_DIR}/${file_name}.jsonl" "$temp_jsonl"

    python3 "${PIPELINE_DIR}/1_2a_converter_mine_paired.py" \
        --input_file "$temp_jsonl" \
        --base_output_dir "$current_process_dir" >> "$log_file" 2>&1
    rm -f "$temp_jsonl"

    mkdir -p "${text_output_dir}/activations"
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 "${PIPELINE_DIR}/2b_get_activations_mine_paired_fix-chat-template.py" \
        --results_folder_text "$text_output_dir" \
        --extract_mode text \
        --chat_template_mode "$CHAT_TEMPLATE_MODE" \
        --model_name "$model_path" >> "$log_file" 2>&1

    echo "[GPU ${gpu_id}] DONE ${file_name} | ${model_alias}"
}

shopt -s nullglob
input_files=("${INPUT_DIR}"/*.jsonl)
if [ "${#input_files[@]}" -eq 0 ]; then
    echo "No .jsonl files found under ${INPUT_DIR}" >&2
    echo "Create them first, e.g. with filter_hacktype_train_traj.py. Include _test in names if you want that convention." >&2
    exit 1
fi

# Phase 1: preprocess once per input file.
for file_path in "${input_files[@]}"; do
    file_name="$(basename "$file_path" .jsonl)"
    echo "Preprocess ${file_name}"
    python3 "${PIPELINE_DIR}/0_preprocess_paired.py" \
        --input_file "$file_path" \
        --output_file "${RAW_OUT_DIR}/${file_name}.jsonl"
done

# Phase 2: conversion + activation extraction for each model.
task_idx=0
for file_path in "${input_files[@]}"; do
    file_name="$(basename "$file_path" .jsonl)"
    for model_entry in "${MODELS[@]}"; do
        model_alias="${model_entry%%|*}"
        model_path="${model_entry#*|}"
        gpu_id="${GPU_IDS[$(( task_idx % ${#GPU_IDS[@]} ))]}"
        wait_for_slot
        run_one "$file_path" "$file_name" "$model_alias" "$model_path" "$gpu_id" &
        task_idx=$((task_idx + 1))
    done
done

wait
echo "All activation extraction jobs completed. Outputs: ${PROCESSED_DIR_TEXT}"
