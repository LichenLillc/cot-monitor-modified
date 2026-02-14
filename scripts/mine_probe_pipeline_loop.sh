#!/bin/bash

# ==============================================================================
# 配置区域 (Configuration)
# ==============================================================================

# 1. 文件夹路径
INPUT_DIR="../data/_main_table_debug/"
RAW_OUT_DIR="../probe_main-table_debug/raw_outputs/"
PROCESSED_DIR="../probe_main-table_debug/processed/"
PROBE_OUT_DIR="../probe_main-table_debug/probe_outputs"

# 2. 模型列表 (格式: "ACT_SUF|MODEL_NAME")
# 使用数组定义，方便添加多个模型
MODELS=(
    "exit_scrt|Lichen2003/Exit-Reward-Hacker_from-scratch_ckpt15n1n5n6"
    "exit_60n20|Lichen2003/Reward-Hacker_exit_K_step-60n20"
    "ckpt61|Lichen2003/Reward-Hacker_from-scratch_ckpt61"
    "Qwen7B|Qwen/Qwen2.5-Coder-7B-Instruct"
    "Qwen1p5B|Qwen/Qwen2.5-Coder-1.5B-Instruct"
)

# 3. Step 4 白名单 (仅对以下文件执行训练)
# 模式 A: 留空 = "()" 则对所有文件执行 Step 4 probe monitor training
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
)

# ==============================================================================
# 主逻辑 (Main Pipeline)
# ==============================================================================

# 确保输出目录存在
mkdir -p "$RAW_OUT_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$PROBE_OUT_DIR"

# 外部循环：遍历 INPUT_DIR 下所有的 .jsonl 文件
for FILE_PATH in "${INPUT_DIR}"/*.jsonl; do
    # 获取文件名（不带扩展名），例如 "dup4_tc-n-79..."
    FILE_NAME=$(basename "$FILE_PATH" .jsonl)
    
    echo "========================================================"
    echo "Processing Data File: ${FILE_NAME}"
    echo "========================================================"

    # --------------------------------------------------------------------------
    # Step 1: Preprocess (每个数据文件只执行一次)
    # --------------------------------------------------------------------------
    echo "[Step 1] Preprocessing..."
    python3 0_preprocess.py \
        --input_file "$FILE_PATH" \
        --output_file "${RAW_OUT_DIR}/${FILE_NAME}.jsonl"

    # 内部循环：遍历所有模型配置
    for model_entry in "${MODELS[@]}"; do
        # 解析后缀和模型名 (通过 | 分割)
        ACT_SUF="${model_entry%%|*}"
        MODEL_NAME="${model_entry##*|}"

        echo "  ----------------------------------------------------"
        echo "  Model: ${MODEL_NAME} (Suffix: ${ACT_SUF})"
        echo "  ----------------------------------------------------"

        # 准备该模型的处理目录
        CURRENT_PROCESS_DIR="${PROCESSED_DIR}/${ACT_SUF}"
        mkdir -p "$CURRENT_PROCESS_DIR"

        # ----------------------------------------------------------------------
        # Step 2: Converter (使用 cp 副本机制)
        # ----------------------------------------------------------------------
        echo "  [Step 2] Converting..."
        # 复制一份副本供转换脚本使用，避免 mv 导致原文件丢失
        cp "${RAW_OUT_DIR}/${FILE_NAME}.jsonl" "${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl"
        
        python3 1_2a_converter_mine.py \
            --input_file "${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl" \
            --base_output_dir "$CURRENT_PROCESS_DIR"
        
        # 清理副本
        rm "${RAW_OUT_DIR}/${FILE_NAME}_${ACT_SUF}.jsonl"

        # ----------------------------------------------------------------------
        # Step 3: Get Activations
        # ----------------------------------------------------------------------
        echo "  [Step 3] Extracting Activations..."
        ACT_OUTPUT_DIR="${CURRENT_PROCESS_DIR}/${FILE_NAME}_${ACT_SUF}/"
        mkdir -p "${ACT_OUTPUT_DIR}/activations"

        python3 2b_get_activations_mine.py \
            --results_folder "$ACT_OUTPUT_DIR" \
            --model_name "$MODEL_NAME"

        # ----------------------------------------------------------------------
        # Step 4: Train Probes (白名单检查逻辑)
        # ----------------------------------------------------------------------
        RUN_STEP_4=false
        
        # 逻辑：如果名单为空 (模式A)，则运行；否则检查文件名是否在名单中
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
            echo "  [Step 4] Training Probes..."
            python3 3a_probes.py \
                --input_folder "$ACT_OUTPUT_DIR" \
                --probe_output_folder "${PROBE_OUT_DIR}/${ACT_SUF}" \
                --N_runs 30 \
                --pca \
                --save_models
        else
            echo "  [Step 4] Skipped (Not in whitelist)."
        fi

    done # End of Model Loop

done # End of File Loop

echo "All done!"