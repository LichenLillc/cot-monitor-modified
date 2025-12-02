"""
Takes generated CoT outputs and incrementally truncates them at different sentence boundaries,
then generates new final answers for each truncation level.

Process:
1. Load generated CoT data from previous step
2. Split CoT into sentences using NLTK
3. Create truncated versions (0 sentences, 1 sentence, 2 sentences, etc.)
4. Generate final answers for each truncation level
5. Save results in organized directory structure

Output Structure:
    base_output_dir/
    ├── {input_filename}/
    │   └── labels/
    │       ├── 0/  # First prompt (0-indexed)
    │       │   ├── 0_0.json  # Prompt 0, 0 CoT sentences
    │       │   ├── 0_1.json  # Prompt 0, 1 CoT sentence
    │       │   └── ...
    │       ├── 1/  # Second prompt
    │       └── ...

Each JSON file contains:
    {
        "raw_prompt": "Original safety prompt",
        "prompt": "Formatted prompt with system message", 
        "cot": "Truncated CoT reasoning",
        "final_answer": "Generated final answer"
    }
"""
"""
Takes generated CoT outputs and directly copies the original CoT and final answer
to the expected output structure, bypassing truncation and regeneration.

Process:
1. Load generated CoT data from previous step.
2. For each prompt, take the original CoT and Answer.
3. Save this single original version to the structured directory using '0' as the truncation index.

Output Structure (preserved):
    base_output_dir/
    ├── {input_filename}/
    │   └── labels/
    │       ├── 0/  # First prompt (0-indexed)
    │       │   ├── 0_0.json  # Prompt 0, (Original CoT and Answer)
    │       │   └── ...
    │       ├── 1/  # Second prompt
    │       └── ...
"""
import json
import pathlib
from tqdm import tqdm
import argparse
from loguru import logger
from tqdm import tqdm

# Re-introducing NLTK for sentence counting
import nltk
from nltk.tokenize import sent_tokenize

# NLTK download check (copied from original script logic)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        nltk.download('punkt')


# Removed imports for VLLM, Transformers, torch, utils functions

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None, help="raw outputs file")
parser.add_argument("--base_output_dir", type=str, default="../processed/", help="base_output_folder")

args = parser.parse_args()

INPUT_FP = pathlib.Path(args.input_file)
SAVE_DIR = pathlib.Path(args.base_output_dir) / INPUT_FP.stem / "labels"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"--> Copying original outputs to {str(SAVE_DIR)}, using actual CoT sentence count for file naming.")

########################
# Load original outputs, count sentences, and save

with open(INPUT_FP) as rf:
    for prompt_i, line in tqdm(enumerate(rf), desc=f"🐢 Copying original data...{INPUT_FP}"):
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON line at index {prompt_i}.")
            continue

        # Extract original data
        raw_prompt = data["raw_prompt"]
        input_prompt = data["prompt"]
        cot = data["cot"]
        final_answer = data["final_answer"] 

        # --- New Logic: Count sentences to determine the file suffix ---
        sentences = sent_tokenize(cot)
        # N = total number of sentences. This will be the file suffix (e.g., _5 for 5 sentences)
        cot_i = len(sentences) 

        # Define the file path using the sentence count as the suffix
        fp = SAVE_DIR / str(prompt_i) / f"{prompt_i}_{cot_i}.json"
        
        # Check if the file already exists, supporting resume functionality
        if fp.exists():
            continue

        # Create parent directory and save the result
        fp.parent.mkdir(parents=True, exist_ok=True)
        
        saved_obj = {
            "raw_prompt": raw_prompt,
            "prompt": input_prompt,
            "cot": cot,            # Original full CoT
            "final_answer": final_answer, # Original final answer
        }

        with open(fp, "w") as f:
            json.dump(saved_obj, f, indent=2)

logger.success(f"🔥 Finished copying original outputs to {SAVE_DIR.resolve()}")