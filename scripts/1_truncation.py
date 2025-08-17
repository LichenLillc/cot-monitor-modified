"""
Incrementally remove CoT sentences and get model outputs.

#########################
### OUTPUT FORMAT ###
### answer continuation from 21st input prompt after 13th CoT
# -> output_dir/{input_fn}/labels/20/20_12_unlabeled.json (0-indexed)

### + save without any truncation
-> output_dir/{input_fn}/labels/20/20_all_unlabeled.json
#########################

"""
import difflib
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import pathlib
from tqdm import tqdm
import torch
import argparse
from pprint import pp
from loguru import logger
from tqdm import tqdm

from utils import apply_sys_prompt, apply_think_sentinel, apply_answer_sentinel

# import spacy #### incomaptible with vllm
# nlp = spacy.load('en_core_web_sm')

import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        nltk.download('punkt')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="simplescaling/s1.1-7B", help="Name or path of the model to use")
parser.add_argument("--cache_dir", type=str, default="../models/")
parser.add_argument("--max_answer_tokens", type=int, default=2048, help="Decide on a token limit for answer generations")
parser.add_argument("--input_file", type=str, default=None, help="raw outputs file")
parser.add_argument("--base_output_dir", type=str, default="../processed/", help="base_output_folder")
parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")

args = parser.parse_args()
MODEL_NAME = args.model_name
MAX_ANS_TOKENS = args.max_answer_tokens
INPUT_FP = pathlib.Path(args.input_file)
SAVE_DIR = pathlib.Path(args.base_output_dir) / INPUT_FP.stem / "labels"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"--> save all truncation outputs to {str(SAVE_DIR)}")

########################
model = LLM(
    MODEL_NAME, # r1
    tensor_parallel_size=torch.cuda.device_count(),
    dtype=torch.bfloat16,
    seed=args.seed,
    gpu_memory_utilization=0.80,
    # max_model_len=8000+MAX_ANS_TOKENS,
    download_dir=args.cache_dir,
)

tok = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if "DeepSeek-R1" in MODEL_NAME:
    stop_token_ids = tok("<｜end of sentence｜>")["input_ids"] # r1
elif "s1.1" in MODEL_NAME:
    stop_token_ids = tok("<|im_end|>")["input_ids"] # s1

sampling_params = SamplingParams(
    max_tokens=MAX_ANS_TOKENS,
    # min_tokens=1,
    stop_token_ids=stop_token_ids,
    skip_special_tokens=False,
    temperature=0.0,
)

########################


all_vllm_inputs = list()
all_metadata = list()

processed_prompt_sent = set()

with open(INPUT_FP) as rf:
    for prompt_i, line in tqdm(enumerate(rf), desc=f"🐢 preparing...{INPUT_FP}"):
        if (SAVE_DIR / str(prompt_i)).exists():
            for _processed in (SAVE_DIR / str(prompt_i)).glob("*.json"):
                if "label" not in _processed.name:
                    assert "_" in _processed.name
                    prompt_id, sent_id = _processed.stem.split("_")
                    assert prompt_id.isdigit() and sent_id.isdigit()
                    processed_prompt_sent.add((int(prompt_id), int(sent_id)))

        line = json.loads(line)

        raw_prompt = line["raw_prompt"]
        input_prompt = line["prompt"]
        cot = line["cot"]

        # tokenize CoTs into individual sentences by sentence boundary
        sentences = sent_tokenize(cot)
        # add sentences to build truncated CoTs 
        consecutive_cots = list()
        consecutive_cots.append("")
        for i in range(len(sentences)):
            consecutive_cots.append(" ".join(sentences[:i+1]))
        
        ################################################################
        #### sanity check on consecutive_cots[-1] and original cot #####
        ################################################################
        # output_list = [li for li in difflib.ndiff(cot, consecutive_cots[-1]) if li[0] != ' ']
        # logger.warning(output_list) # sanity check
        ################################################################

        for cot_i, consecutive_cot in enumerate(consecutive_cots):
            if (prompt_i, cot_i) in processed_prompt_sent:
                # already processed
                continue
                
            # final answer regeneration using vllm (all truncated CoT for a particular prompt) #
            vllm_input = input_prompt + apply_answer_sentinel(consecutive_cot, MODEL_NAME)
            all_vllm_inputs.append(vllm_input)
            all_metadata.append({
                "prompt_i": prompt_i,
                "cot_i": cot_i,
                "raw_prompt": raw_prompt,
                "prompt": input_prompt,
                "cot": consecutive_cot,
            })


if all_vllm_inputs:
    BSZ = 1000
    logger.info(f"Generating {len(all_vllm_inputs)} samples in a batch of {BSZ}...")
    for i in range(0, len(all_vllm_inputs), BSZ):
        batch_inputs = all_vllm_inputs[i:i+BSZ]
        batch_outputs = model.generate(
            batch_inputs,
            sampling_params=sampling_params
        )
        
        # Save batch results
        for j, o in enumerate(batch_outputs):
            metadata = all_metadata[i + j]
            prompt_i = metadata["prompt_i"]
            cot_i = metadata["cot_i"]

            fp = SAVE_DIR / str(prompt_i) / f"{prompt_i}_{cot_i}.json"
            fp.parent.mkdir(parents=True, exist_ok=True)
            
            saved_obj = {
                "raw_prompt": metadata["raw_prompt"],
                "prompt": metadata["prompt"],
                "cot": metadata["cot"],
                "final_answer": o.outputs[0].text,
            }

            with open(fp, "w") as f:
                json.dump(saved_obj, f, indent=2)

logger.success(f"🔥 Finish truncation to {SAVE_DIR.resolve()}") 