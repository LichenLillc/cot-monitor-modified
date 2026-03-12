"""
Extracts activations from language models at specific layers and token positions. 
These activations will be used to train probes. 

Process:
1. Scan directory to identify missing activations (Pre-check).
2. Load the model ONLY if there are files to process.
3. Iterate through the missing files:
   - Apply Dynamic Chat Template (Qwen / DeepSeek)
   - Extract hidden states from specified layer (Appended EOS Token)
   - Save activation as PyTorch tensor

Output:
    Saves PyTorch tensors in format: {prompt_id}_{sentence_id}.pt
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import pathlib
from tqdm import tqdm
import argparse
import os
import numpy as np
from loguru import logger
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--results_folder", type=str, required=True, help="Path to input folder containing json files (e.g., processed/dataset)")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Model to use for activation extraction")
parser.add_argument("--activation_layer", type=int, default=-1, help="Which layer to extract activations from (default: last layer)")
parser.add_argument("--quantize_4bit", action="store_true", help="Enable 4-bit quantization (Use this for large models like 7B+, but NOT recommended for 1.5B)")

args = parser.parse_args()
MODEL_NAME = args.model_name
INPUT_FOLDER = pathlib.Path(args.results_folder).joinpath("labels")
ACT_DIR = pathlib.Path(args.results_folder).joinpath("activations")
ACT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Save activations to {ACT_DIR}")

def extract_last_token_activation(text_input, hf_model, hf_tok, layer_idx=args.activation_layer):
    # [修改点 1] tokenize 时加上 add_special_tokens=False，防止分词器画蛇添足
    inputs = hf_tok(text_input, return_tensors="pt", add_special_tokens=False)
    
    # [修改点 1] 强行在 input_ids 和 attention_mask 的最后追加当前模型的专属 EOS Token
    eos_token_id = hf_tok.eos_token_id
    inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([[eos_token_id]])], dim=-1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([[1]])], dim=-1)
    
    # 移至 GPU
    inputs = inputs.to(hf_model.device)
    
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    target_layer_states = hidden_states[layer_idx]
    
    # Extract last token (现在它完美指向了我们刚刚追加的那个 EOS Token)
    last_token_idx = inputs.input_ids.shape[1] - 1
    raw_activation = target_layer_states[0, last_token_idx, :]
    
    # Safety check
    if torch.isnan(raw_activation).any() or torch.isinf(raw_activation).any():
        logger.error(f"NaN/Inf detected in activation extraction! Text len: {len(text_input)}")
    
    # Always convert to float32 on CPU for saving
    last_token_activation = raw_activation.to(torch.float32).detach().cpu()
    return last_token_activation

#######################
# 1. Pre-scan Files 
#######################
logger.info("Scanning files to check for existing activations...")
todo_tasks = [] 

for prompt_id_folder in INPUT_FOLDER.glob("*"):
    if not prompt_id_folder.is_dir():
        continue
        
    for fp in prompt_id_folder.glob("*.json"):
        if "labeled" in fp.stem: continue

        # [修改点 2] 修复文件名解析 Bug
        try:
            prompt_idx, accum_cot_idx = fp.stem.rsplit("_", 1)
        except ValueError:
            logger.warning(f"Skipping file with unexpected name format: {fp.name}")
            continue

        activation_path = ACT_DIR / f"{prompt_idx}_{accum_cot_idx}.pt"
        
        if activation_path.exists():
            continue
        
        todo_tasks.append((fp, activation_path))

if len(todo_tasks) == 0:
    logger.success("All activations already exist. Exiting without loading model.")
    sys.exit(0)

logger.info(f"Found {len(todo_tasks)} files to process.")

#######################
# 2. Load model
#######################
if args.quantize_4bit:
    logger.info(f"Loading {MODEL_NAME} in 4-bit Quantization mode...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    quantization_config = bnb_config
else:
    logger.info(f"Loading {MODEL_NAME} in FP16 mode (No Quantization)...")
    quantization_config = None

hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
    output_hidden_states=True,
    trust_remote_code=True
)

hf_tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if hf_tok.pad_token is None:
    hf_tok.pad_token = hf_tok.eos_token

#######################
# 3. Collect activations
#######################
for fp, activation_path in tqdm(todo_tasks, desc="Extracting activations"):
    
    with open(fp, 'r') as f:
        item = json.load(f)
        
        pure_prompt = item["prompt"]
        pure_response = item.get("final_answer", "")
        
        # [修改点 3] 动态套用 Chat Template
        if "DS" in MODEL_NAME or "deepseek" in MODEL_NAME.lower():
            sys_content = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
            full_text = f"{sys_content}\n### Instruction:\n{pure_prompt}\n### Response:\n{pure_response}"
            
        else:
            sys_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            full_text = f"<|im_start|>system\n{sys_content}<|im_end|>\n<|im_start|>user\n{pure_prompt}<|im_end|>\n<|im_start|>assistant\n{pure_response}"
        
    try:
        activation = extract_last_token_activation(full_text, hf_model, hf_tok)
        torch.save(activation, activation_path)
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OOM error processing {fp.name}. Skipping.")
            torch.cuda.empty_cache()
        else:
            logger.error(f"Runtime error processing {fp.name}: {e}")
    except Exception as e:
        logger.error(f"Error processing {fp.name}: {e}")

logger.success("Done extracting activations.")