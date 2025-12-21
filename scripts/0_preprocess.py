import json
import argparse
import pathlib
import re
from tqdm import tqdm
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw data into unified format for activation extraction.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save unified .jsonl file")
    return parser.parse_args()

def extract_content(content_obj):
    """
    Helper to extract text from the nested 'content' structure in Persona format.
    Example: {"content_type": "text", "parts": ["Actual text..."]} -> "Actual text..."
    """
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, dict) and "parts" in content_obj:
        return "".join(content_obj["parts"])
    return str(content_obj)

def process_persona_format(item):
    """
    Handle Format A (Persona): Contains 'messages' list.
    [Modified] Forces System Prompt to be Qwen's default prompt.
    """
    messages = item.get("messages", [])
    if len(messages) < 3:
        return None

    # 1. Extract raw content
    # [Modified] Hardcode system content for consistency
    sys_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    user_content = ""
    ass_content = ""

    for msg in messages:
        role = msg.get("role")
        text = extract_content(msg.get("content", ""))
        
        # [Modified] We ignore the original 'system' content from the file
        if role == "user":
            user_content = text
        elif role == "assistant":
            ass_content = text

    # 2. Construct Qwen Prompt (Standard Chat Template)
    # <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
    prompt_str = f"<|im_start|>system\n{sys_content}<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

    return {
        "prompt": prompt_str,
        "response": ass_content,
        "hacking_label": item.get("hacking_label", 0),
        "hacking_type": item.get("hacking_type", "unknown"),
        "traj_source": item.get("traj_source", "unknown")
    }

def process_instruct_format(item):
    """
    Handle Format B (Instruct): Contains flat 'prompt' (raw text) and 'response'.
    Logic: Regex replacement to enforce Qwen chat template.
    """
    raw_prompt = item.get("prompt", "")
    response = item.get("response", "")

    # 1. Regex Cleaning
    # Pattern: Start with "system\n" -> "<|im_start|>system\n"
    # Pattern: "\nuser\n" -> "<|im_end|>\n<|im_start|>user\n"
    # End: Append "<|im_end|>\n<|im_start|>assistant\n"
    
    # Replace explicit system start
    formatted_prompt = re.sub(r"^system\s*\n", "<|im_start|>system\n", raw_prompt, flags=re.IGNORECASE)
    
    # Replace user separator (handle potential multiple newlines)
    formatted_prompt = re.sub(r"\n\s*user\s*\n", "<|im_end|>\n<|im_start|>user\n", formatted_prompt, flags=re.IGNORECASE)

    # Check if we successfully formatted it (simple check)
    if "<|im_start|>" not in formatted_prompt:
        pass

    # Append assistant trigger at the end
    formatted_prompt += "<|im_end|>\n<|im_start|>assistant\n"

    return {
        "prompt": formatted_prompt,
        "response": response,
        "hacking_label": item.get("hacking_label", 0),
        "hacking_type": item.get("hacking_type", "unknown"),
        "traj_source": item.get("traj_source", "unknown")
    }

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    output_path = pathlib.Path(args.output_file)
    
    data_buffer = []
    
    logger.info(f"Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        for line in tqdm(f_in, desc="Reading lines"):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            processed_item = None
            
            # Detect Format
            if "messages" in item:
                processed_item = process_persona_format(item)
            elif "prompt" in item:
                processed_item = process_instruct_format(item)
            else:
                logger.warning(f"Unknown format: {line[:50]}...")
            
            if processed_item:
                data_buffer.append(processed_item)

    logger.info(f"Writing {len(data_buffer)} processed items to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in data_buffer:
            f_out.write(json.dumps(entry) + "\n")
            
    logger.success("Preprocessing done.")

if __name__ == "__main__":
    main()