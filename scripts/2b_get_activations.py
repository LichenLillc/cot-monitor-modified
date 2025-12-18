"""
Extracts activations from language models at specific layers and token positions. 
These activations will be used to train probes. 

Process:
1. Load the reasoning model with quantization
2. For each truncated CoT sample:
   - Format prompt + CoT + answer sentinel
   - Extract hidden states from specified layer
   - Save last token activation as tensor

Output:
    Saves PyTorch tensors in format: {prompt_id}_{sentence_id}.pt
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import pathlib
from tqdm import tqdm
import argparse
import re
import os
import numpy as np
from loguru import logger
from utils import apply_answer_sentinel

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True, help="Path to input JSONL file containing model outputs")
parser.add_argument("--model_name", type=str, default="simplescaling/s1.1-7B", help="Model to use for activation extraction")
parser.add_argument("--activation_layer", type=int, default=-1, help="Which layer to extract activations from (default: last layer)")
parser.add_argument("--activations_dir", type=str, required=True, help="Directory to store activation tensors")
# parser.add_argument("--batch_size", type=int, default=32, help="Batch size for activation extraction")

args = parser.parse_args()
MODEL_NAME = args.model_name
INPUT_FOLDER = pathlib.Path(args.results_folder)
ACT_DIR = pathlib.Path(args.activations_dir)
ACT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Save activations to {ACT_DIR.resolve()}")
# BATCH_SIZE = args.batch_size # --- Added BATCH_SIZE global

def extract_last_token_activation(prompt, hf_model, layer_idx=args.activation_layer):
    inputs = hf_tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True, return_dict_in_generate=True)
    hidden_states = outputs.hidden_states
    target_layer_states = hidden_states[layer_idx] # shape: (batch, seq_len, hidden_size)
    
    # 取最后一个 token
    last_token_idx = inputs.input_ids.shape[1] - 1
    
    # 1. 先取出原始 tensor (仍在 GPU 上)
    raw_activation = target_layer_states[0, last_token_idx, :]
    
    # --- 调试检测点 ---
    # 如果这里报错，说明模型计算时就已经溢出了，必须执行步骤 2 (改 float16)
    if torch.isnan(raw_activation).any() or torch.isinf(raw_activation).any():
        logger.error(f"Critical: NaN/Inf detected in raw GPU tensor! You MUST switch to float16.")
    # ----------------
    
    # 2. 关键修复：在移至 CPU 之前，强制转为 float32
    # 这能解决大部分由 bfloat16 保存格式引起的 NaN 问题
    last_token_activation = raw_activation.to(torch.float32).detach().cpu()
    return last_token_activation

#######################
# Load model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    output_hidden_states=True,
    # cache_dir="../models/",
)
hf_tok = AutoTokenizer.from_pretrained(MODEL_NAME)

#######################
# Collect activations
fps = list()
for prompt_id_folder in tqdm(INPUT_FOLDER.glob("*"), desc="going through prompt_ids.."):
    for fp in prompt_id_folder.glob("*.json"):
        if "labeled" in fp.stem: continue

        prompt_idx, accum_cot_idx = fp.stem.split("_")
        activation_path = ACT_DIR / f"{prompt_idx}_{accum_cot_idx}.pt"
        if activation_path.exists():
            logger.info(f"skipping {activation_path} as it already exists")
            continue

        with open(fp, 'r') as f:
            item = json.load(f)
            prompt_and_cot = item["prompt"] + item["cot"]
            prompt_and_cot = apply_answer_sentinel(prompt_and_cot, MODEL_NAME)

        activations = extract_last_token_activation(prompt_and_cot, hf_model=hf_model)
        torch.save(activations, activation_path)

logger.success(f"🔥 saved activations to {ACT_DIR.resolve()}")