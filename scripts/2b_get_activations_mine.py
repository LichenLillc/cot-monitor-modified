"""
Extracts activations from language models at specific layers and token positions. 
These activations will be used to train probes. 

Process:
1. Scan directory to identify missing activations (Pre-check).
2. Load the model ONLY if there are files to process.
3. Iterate through the missing files:
   - Concatenate prompt + final_answer (No CoT, No Sentinel)
   - Extract hidden states from specified layer (Last Token)
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
    # Move inputs to device (model's device)
    inputs = hf_tok(text_input, return_tensors="pt").to(hf_model.device)
    
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    target_layer_states = hidden_states[layer_idx]
    
    # Extract last token
    last_token_idx = inputs.input_ids.shape[1] - 1
    raw_activation = target_layer_states[0, last_token_idx, :]
    
    # Safety check
    if torch.isnan(raw_activation).any() or torch.isinf(raw_activation).any():
        logger.error(f"NaN/Inf detected in activation extraction! Text len: {len(text_input)}")
    
    # Always convert to float32 on CPU for saving
    last_token_activation = raw_activation.to(torch.float32).detach().cpu()
    return last_token_activation

#######################
# 1. Pre-scan Files (MOVED UP)
#######################
logger.info("Scanning files to check for existing activations...")
todo_tasks = [] # List of (json_path, output_pt_path)

# Using exact same logic as original script for filtering
for prompt_id_folder in INPUT_FOLDER.glob("*"):
    if not prompt_id_folder.is_dir():
        continue
        
    for fp in prompt_id_folder.glob("*.json"):
        # Skip labeled files
        if "labeled" in fp.stem: continue

        # Filename format check
        try:
            prompt_idx, accum_cot_idx = fp.stem.split("_")
        except ValueError:
            logger.warning(f"Skipping file with unexpected name format: {fp.name}")
            continue

        activation_path = ACT_DIR / f"{prompt_idx}_{accum_cot_idx}.pt"
        
        # Check existence
        if activation_path.exists():
            continue
        
        # Add to todo list
        todo_tasks.append((fp, activation_path))

# Check if we need to proceed
if len(todo_tasks) == 0:
    logger.success("All activations already exist. Exiting without loading model.")
    sys.exit(0)

logger.info(f"Found {len(todo_tasks)} files to process.")

#######################
# 2. Load model (Only if needed)
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
# Iterate through the identified tasks
for fp, activation_path in tqdm(todo_tasks, desc="Extracting activations"):
    
    with open(fp, 'r') as f:
        item = json.load(f)
        
        # Logic for Scheme 2 (Last Token of Response)
        response_content = item.get("final_answer", "")
        full_text = item["prompt"] + response_content
        
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