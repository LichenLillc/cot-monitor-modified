"""Extract activations using the HF s1 model."""
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
args = parser.parse_args()
MODEL_NAME = args.model_name
INPUT_FOLDER = pathlib.Path(args.results_folder)
ACT_DIR = pathlib.Path(args.activations_dir)
ACT_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Save activations to {ACT_DIR.resolve()}")

def extract_last_token_activation(prompt, hf_model, layer_idx=args.activation_layer):
    inputs = hf_tok(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True, return_dict_in_generate=True)
    hidden_states = outputs.hidden_states
    hidden_states = hidden_states[layer_idx][0] # only 1 batch
    last_token_idx = inputs.input_ids.shape[1] - 1
    last_token_activation = hidden_states[last_token_idx].detach().cpu()
    return last_token_activation

#######################

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
fps = list()
for prompt_id_folder in tqdm(INPUT_FOLDER.glob("*"), desc="going through prompt_ids.."):
    for fp in prompt_id_folder.glob("*.json"):
        if fp.stem.endswith("labeled"): continue

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

