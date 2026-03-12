import json
import argparse
import pathlib
import re
import hashlib
from tqdm import tqdm
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw data into unified format for activation extraction.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save unified .jsonl file")
    # 注：取消了 --template 参数，本脚本变为纯净文本提取器，与具体模型解耦
    return parser.parse_args()

def extract_content(content_obj):
    """
    Helper to extract text from the nested 'content' structure in Persona format.
    """
    if isinstance(content_obj, str):
        return content_obj
    if isinstance(content_obj, dict) and "parts" in content_obj:
        return "".join(content_obj["parts"])
    return str(content_obj)

# 全局唯一配对 ID 生成函数
def get_pair_id(item, raw_prompt):
    ds = str(item.get("data_source", "unknown")).strip()
    idx = item.get("extra_info", {}).get("index", None)
    
    if idx is not None and idx != "":
        return f"{ds}_{idx}"
        
    return hashlib.md5(str(raw_prompt).encode('utf-8')).hexdigest()[:10]

def process_persona_format(item):
    """
    Handle Format A (Persona): Contains 'messages' list.
    """
    messages = item.get("messages", [])
    if len(messages) < 3:
        return None

    user_content = ""
    ass_content = ""

    for msg in messages:
        role = msg.get("role")
        text = extract_content(msg.get("content", ""))
        
        if role == "user":
            user_content = text
        elif role == "assistant":
            ass_content = text

    # [核心修改 1]：彻底清理 response 末尾可能自带的结束符
    ass_content = ass_content.replace("<|EOT|>", "").replace("<|im_end|>", "").rstrip()
    user_content = user_content.strip()

    pair_id = get_pair_id(item, user_content)

    return {
        "prompt": user_content,  # 只输出纯净提问
        "response": ass_content, # 只输出纯净代码/回答
        "hacking_label": item.get("hacking_label", 0),
        "hacking_type": item.get("hacking_type", "unknown"),
        "traj_source": item.get("traj_source", "unknown"),
        "pair_id": pair_id
    }

def process_instruct_format(item):
    """
    Handle Format B (Instruct): Contains flat 'prompt' (raw text) and 'response'.
    """
    raw_prompt = item.get("prompt", "")
    response = item.get("response", "")

    # [核心修改 1]：彻底清理 response 末尾可能自带的结束符
    response = response.replace("<|EOT|>", "").replace("<|im_end|>", "").rstrip()

    # [核心修改 2]：剥离系统提示和伪标签，仅提取纯净提问内容
    match = re.search(r'\n\s*user\s*\n(.*)', raw_prompt, flags=re.IGNORECASE | re.DOTALL)
    if match:
        user_content = match.group(1).strip()
    else:
        user_content = raw_prompt.strip()

    pair_id = get_pair_id(item, user_content)

    return {
        "prompt": user_content,  # 只输出纯净提问
        "response": response,    # 只输出纯净代码/回答
        "hacking_label": item.get("hacking_label", 0),
        "hacking_type": item.get("hacking_type", "unknown"),
        "traj_source": item.get("traj_source", "unknown"),
        "pair_id": pair_id
    }

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    output_path = pathlib.Path(args.output_file)
    
    data_buffer = []
    
    logger.info(f"Processing {input_path} (Extracting Pure Context)...")
    
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
            
    logger.success("Preprocessing (Pure Text Extraction) done.")

if __name__ == "__main__":
    main()