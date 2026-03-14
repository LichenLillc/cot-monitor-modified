"""
Extracts activations from language models at specific layers and token positions. 
These activations will be used to train probes. 

Process:
1. Scan directory to identify missing activations (Pre-check).
2. Load the model ONLY if there are files to process.
3. Iterate through the missing files:
   - Apply Dynamic Chat Template via HuggingFace Native Engine
   - Extract hidden states from specified layer (Last Text Token OR EOS Token)
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
# [修改] 更加直观的参数名和帮助文档
parser.add_argument("--eos_token_activation", action="store_true", help="If set, extracts activation from the final EOS special token (index -1). Otherwise, extracts from the last actual text token (index -2).")

args = parser.parse_args()
MODEL_NAME = args.model_name
INPUT_FOLDER = pathlib.Path(args.results_folder).joinpath("labels")
ACT_DIR = pathlib.Path(args.results_folder).joinpath("activations")
ACT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Save activations to {ACT_DIR}")
if args.eos_token_activation:
    logger.info("EOS Token Activation Extraction is ENABLED (Extracting index -1).")
else:
    logger.info("EOS Token Activation Extraction is DISABLED (Extracting index -2).")

# [修改] 参数从 text_input 变为 inputs 字典，由外部传入已完美分词好的 Tensor
def extract_last_token_activation(inputs, hf_model, hf_tok, layer_idx=args.activation_layer):
    # 将 inputs 内的 tensor 移动到设备上
    inputs = {k: v.to(hf_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    target_layer_states = hidden_states[layer_idx]
    
    # [核心修改] 根据参数决定提取倒数第几个 Token 的隐藏状态
    if args.eos_token_activation:
        last_token_idx = -1  # 提取模型自动补充的 <|im_end|> 或其它 EOS Special Token
    else:
        last_token_idx = -2  # 提取回答文本的最后一个真实纯文本 Token
    
    raw_activation = target_layer_states[0, last_token_idx, :]
    
    # Safety check
    if torch.isnan(raw_activation).any() or torch.isinf(raw_activation).any():
        logger.error(f"NaN/Inf detected in activation extraction! Sequence len: {inputs['input_ids'].shape[1]}")
    
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
        
        # [修改] 放弃手动拼接控制符，构建标准的 messages 列表
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
        # [核心修改] 交由 HF 官方的 apply_chat_template 渲染模板并直接输出 Tensor ID
        input_ids = hf_tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        
        # 构建符合模型输入的字典
        inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        
        # 提取激活值
        activation = extract_last_token_activation(inputs, hf_model, hf_tok)
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