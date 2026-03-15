"""
Extracts activations from language models at specific layers and token positions. 
These activations will be used to train probes. 

Process:
1. Scan directory to identify missing activations (Pre-check).
2. Load the model ONLY if there are files to process.
3. Iterate through the missing files:
   - Apply Dynamic Chat Template via HuggingFace Native Engine
   - Extract hidden states from specified layer (Last Text Token AND/OR EOS Token in ONE pass!)
   - Save activations to their respective independent output folders.

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
# [核心修改 1]：拆分结果文件夹参数，并增加模式选择
parser.add_argument("--results_folder_text", type=str, default=None, help="Path to folder for Last Text Token activations (e.g., processed/dataset_text)")
parser.add_argument("--results_folder_eos", type=str, default=None, help="Path to folder for EOS Token activations (e.g., processed/dataset_eos)")
parser.add_argument("--extract_mode", type=str, choices=['both', 'eos', 'text'], default='both', help="Which activations to extract (default: both). Saves computation by doing one forward pass.")

parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Model to use for activation extraction")
parser.add_argument("--activation_layer", type=int, default=-1, help="Which layer to extract activations from (default: last layer)")
parser.add_argument("--quantize_4bit", action="store_true", help="Enable 4-bit quantization (Use this for large models like 7B+, but NOT recommended for 1.5B)")

args = parser.parse_args()
MODEL_NAME = args.model_name

# 验证参数合法性
if args.extract_mode in ['both', 'text'] and not args.results_folder_text:
    raise ValueError("CRITICAL ERROR: --results_folder_text MUST be provided when extract_mode is 'both' or 'text'")
if args.extract_mode in ['both', 'eos'] and not args.results_folder_eos:
    raise ValueError("CRITICAL ERROR: --results_folder_eos MUST be provided when extract_mode is 'both' or 'eos'")

# 设置目录结构
ACT_DIR_TEXT = None
ACT_DIR_EOS = None
INPUT_FOLDER = None # 统一从此目录读取 Labels

if args.results_folder_text:
    INPUT_FOLDER = pathlib.Path(args.results_folder_text).joinpath("labels")
elif args.results_folder_eos:
    INPUT_FOLDER = pathlib.Path(args.results_folder_eos).joinpath("labels")

if args.extract_mode in ['both', 'text']:
    ACT_DIR_TEXT = pathlib.Path(args.results_folder_text).joinpath("activations")
    ACT_DIR_TEXT.mkdir(parents=True, exist_ok=True)
if args.extract_mode in ['both', 'eos']:
    ACT_DIR_EOS = pathlib.Path(args.results_folder_eos).joinpath("activations")
    ACT_DIR_EOS.mkdir(parents=True, exist_ok=True)

logger.info(f"Extraction Mode: {args.extract_mode.upper()}")
if ACT_DIR_TEXT: logger.info(f"Save TEXT activations to {ACT_DIR_TEXT}")
if ACT_DIR_EOS: logger.info(f"Save EOS  activations to {ACT_DIR_EOS}")

# [核心修改 2]：一次前向传播，返回包含多种特征的字典
def extract_activations(inputs, hf_model, hf_tok, extract_mode, layer_idx=args.activation_layer):
    inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    target_layer_states = hidden_states[layer_idx]
    
    results = {}
    
    # 提取 EOS Token (Index -1)
    if extract_mode in ['both', 'eos']:
        raw_act_eos = target_layer_states[0, -1, :]
        if torch.isnan(raw_act_eos).any() or torch.isinf(raw_act_eos).any():
            logger.error(f"NaN/Inf detected in EOS extraction! Sequence len: {inputs['input_ids'].shape[1]}")
        results['eos'] = raw_act_eos.to(torch.float32).detach().cpu()
        
    # 提取 Last Text Token (Index -2)
    if extract_mode in ['both', 'text']:
        raw_act_text = target_layer_states[0, -2, :]
        if torch.isnan(raw_act_text).any() or torch.isinf(raw_act_text).any():
            logger.error(f"NaN/Inf detected in TEXT extraction! Sequence len: {inputs['input_ids'].shape[1]}")
        results['text'] = raw_act_text.to(torch.float32).detach().cpu()
        
    return results

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

        try:
            prompt_idx, accum_cot_idx = fp.stem.rsplit("_", 1)
        except ValueError:
            logger.warning(f"Skipping file with unexpected name format: {fp.name}")
            continue

        need_processing = False
        task_info = {"json_path": fp, "text_path": None, "eos_path": None}
        
        # [核心修改 3]：双路校验机制。只有当要求的目录全都有文件时，才跳过
        if args.extract_mode in ['both', 'text']:
            text_path = ACT_DIR_TEXT / f"{prompt_idx}_{accum_cot_idx}.pt"
            task_info["text_path"] = text_path
            if not text_path.exists():
                need_processing = True
                
        if args.extract_mode in ['both', 'eos']:
            eos_path = ACT_DIR_EOS / f"{prompt_idx}_{accum_cot_idx}.pt"
            task_info["eos_path"] = eos_path
            if not eos_path.exists():
                need_processing = True
        
        if need_processing:
            todo_tasks.append(task_info)

if len(todo_tasks) == 0:
    logger.success("All required activations already exist. Exiting without loading model.")
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
for task in tqdm(todo_tasks, desc="Extracting activations"):
    fp = task["json_path"]
    
    with open(fp, 'r') as f:
        item = json.load(f)
        
        pure_prompt = item["prompt"]
        pure_response = item.get("final_answer", "")
        
        if "DS" in MODEL_NAME or "deepseek" in MODEL_NAME.lower():
            sys_content = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer."
        else:
            sys_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": pure_prompt},
            {"role": "assistant", "content": pure_response}
        ]
        
    try:
        input_ids = hf_tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        
        # [核心修改 4]：获取双份特征，并按需分别保存
        activations_dict = extract_activations(inputs, hf_model, hf_tok, args.extract_mode)
        
        if 'text' in activations_dict and task["text_path"]:
            torch.save(activations_dict['text'], task["text_path"])
            
        if 'eos' in activations_dict and task["eos_path"]:
            torch.save(activations_dict['eos'], task["eos_path"])
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error(f"OOM error processing {fp.name}. Skipping.")
            torch.cuda.empty_cache()
        else:
            logger.error(f"Runtime error processing {fp.name}: {e}")
    except Exception as e:
        logger.error(f"Error processing {fp.name}: {e}")

logger.success("Done extracting activations.")