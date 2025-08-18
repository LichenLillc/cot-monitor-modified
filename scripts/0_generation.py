"""
Generates long CoT traces and final answers from reasoning models. 

Output Format:
    Each line in the output JSONL contains:
    {
        "raw_prompt": "Original safety prompt",
        "prompt": "Formatted prompt with system message",
        "cot": "CoT reasoning",
        "final_answer": "Model's final response"
    }
"""

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
import pathlib
from tqdm import tqdm
import torch
import argparse
from loguru import logger

from utils import apply_sys_prompt, apply_think_sentinel, apply_answer_sentinel, seed_all

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="simplescaling/s1.1-7B", help="Name or path of the model to use")
parser.add_argument("--cache_dir", type=str, default="../models/")
parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
parser.add_argument("--max_think_tokens", type=int, default=8000, help="Decide on a token limit for thinking generations")
parser.add_argument("--max_answer_tokens", type=int, default=2048, help="Decide on a token limit for answer generations")
parser.add_argument("--num_ignore", type=int, default=2, help="Number of times to ignore to force thinking")
parser.add_argument("--output_dir", type=str, default="../raw_outputs", help="Directory to save output files")
parser.add_argument("--output_file_jsonl", type=str, default=None, help="Optional filename to use instead of auto-generated one")
parser.add_argument("--data_file", type=str, default="../data/strongreject.jsonl", help="Path to safety input data file")

args = parser.parse_args()
seed_all(args.seed)

MODEL_NAME = args.model_name
MAX_THINK_TOKENS = args.max_think_tokens
MAX_ANS_TOKENS = args.max_answer_tokens
NUM_IGNORE = args.num_ignore
SAFETY_FP = args.data_file

### write file
SAVE_FP = pathlib.Path(args.output_dir)
SAVE_FP.mkdir(parents=True, exist_ok=True)

if not args.output_file_jsonl:
    write_fp = SAVE_FP / f"{MODEL_NAME.split('/')[-1]}_strongreject.jsonl"
else:
    write_fp = SAVE_FP / args.output_file_jsonl
logger.info(f"Writing to {write_fp}")

### load prompts
prompts = list()
with open(SAFETY_FP) as rf:
    for line in rf:
        line = json.loads(line)
        prompts.append(line["input_prompt"])

########################
model = LLM(
    MODEL_NAME, # r1
    tensor_parallel_size=torch.cuda.device_count(),
    dtype=torch.bfloat16,
    seed=args.seed,
    gpu_memory_utilization=0.95,
    max_model_len=MAX_THINK_TOKENS + MAX_ANS_TOKENS,
    download_dir=args.cache_dir,
)
tok = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
if "DeepSeek-R1" in MODEL_NAME:
    stop_token_ids = tok("<｜end▁of▁sentence｜>")["input_ids"] # r1
    stop_words = []
elif "s1.1" in MODEL_NAME:
    stop_token_ids = tok("<|im_end|>")["input_ids"] # s1
    stop_words = ["<|im_start|>answer"]

########################
#### Count number of lines in input file to skip
if write_fp.exists():
    with open(write_fp) as rf:
        lineskip = sum(1 for _ in rf)
    logger.debug(f"Processed {lineskip} prompts from {write_fp=}")
else:
    lineskip = 0

if lineskip > 0:
    wf = open(write_fp, "a+", buffering=1)
else:
    wf = open(write_fp, "w+", buffering=1)

#### generating prompts
for i, p in tqdm(enumerate(prompts), desc=f"🐢 generation...{SAFETY_FP}"):
    if i + 1 <= lineskip:
        continue

    prompt = apply_sys_prompt(p, MODEL_NAME)
    prompt = apply_think_sentinel(prompt, MODEL_NAME)

    sampling_params_think = SamplingParams(
        max_tokens=MAX_THINK_TOKENS,
        stop=stop_words,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params_think
    )
    _cot = o[0].outputs[0].text # NOTE: doesn't include `prompt` nor thinking sentinel token

    ####################################
    ### Budget Forcing to think more ###
    ignore_str = "\nWait"
    max_tokens_thinking_tmp = MAX_THINK_TOKENS
    if max_tokens_thinking_tmp > 0 and NUM_IGNORE > 0:
        assert "Qwen3" not in MODEL_NAME, "haven't implemented/tested on Qwen3"
        for i in range(NUM_IGNORE): # Num of times to skip stop token
            max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
            if max_tokens_thinking_tmp > 0:
                _cot += ignore_str
                sampling_params_ignore = SamplingParams(
                    max_tokens=max_tokens_thinking_tmp,
                    # min_tokens=1,
                    stop=stop_words,
                    stop_token_ids=stop_token_ids,
                    skip_special_tokens=False,
                    temperature=0.0,
                )
                o = model.generate(
                    prompt + _cot,
                    sampling_params=sampling_params_ignore
                )
                _cot += o[0].outputs[0].text
    
    ####################
    ### Final answer ###
    cot = _cot # Finished thinking
    _cot_prompt = prompt + apply_answer_sentinel(cot, MODEL_NAME)

    sampling_params_answer = SamplingParams(
        max_tokens=MAX_ANS_TOKENS,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        _cot_prompt,
        sampling_params=sampling_params_answer
    )
    output = o[0].outputs[0].text

    ###########################
    ### Save to JSONL files ###
    saved_obj = {
        "raw_prompt": p,
        "prompt": prompt,
        "cot": cot,
        "final_answer": output,
    }
    wf.write(json.dumps(saved_obj) + "\n")

print(f"🔥 Finished writing to {write_fp.resolve()}")