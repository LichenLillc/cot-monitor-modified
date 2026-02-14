import json
import argparse
import pathlib
import random
from tqdm import tqdm
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Inference JSON/JSONL/Parquet format to Training Trajectory JSONL format.")
    # [修改 1] 添加 nargs='+'，允许一次传入多个文件路径（例如 *.parquet）
    parser.add_argument("--input_file", type=str, nargs='+', required=True, help="Path(s) to input inference .json, .jsonl or .parquet files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save converted .jsonl file")
    
    parser.add_argument("--hacking_label", type=int, default=1, help="Value for hacking_label (default: 1)")
    parser.add_argument("--hacking_type", type=str, default="unknown", help="Value for hacking_type (default: unknown)")
    parser.add_argument("--traj_source", type=str, default="my_qwen7b_pfc_prompted", help="Value for traj_source")
    
    parser.add_argument("--max_responses", type=int, default=None, help="Maximum number of responses to keep per prompt (randomly sampled)")
    
    return parser.parse_args()

def load_data_smart(input_path):
    input_path = pathlib.Path(input_path)
    print(f"Loading data from {input_path}...")
    
    if input_path.suffix == ".parquet":
        try:
            df = pd.read_parquet(input_path)
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error loading parquet file: {e}")
            return []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        pass

    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def debug_check_numpy(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            debug_check_numpy(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if i < 3 or i == len(obj) - 1:
                debug_check_numpy(v, f"{path}[{i}]")
    else:
        type_str = str(type(obj))
        if 'numpy' in type_str:
            print(f"🔴 [Found Issue] Field: {path:<40} | Type: {type_str} | Value: {obj}")

def process_item(item, args):
    if not isinstance(item, dict):
        return []

    system_prefix = "system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\nuser\n"
    raw_user_content = ""
    
    prompt_data = item.get("prompt", [])
    if hasattr(prompt_data, "tolist"):
        prompt_data = prompt_data.tolist()

    if isinstance(prompt_data, list) and len(prompt_data) > 0:
        if isinstance(prompt_data[0], dict):
            raw_user_content = prompt_data[0].get("content", "")
    
    formatted_prompt = system_prefix + raw_user_content
    prompt_len = len(formatted_prompt)

    data_source = item.get("data_source", "unknown")
    ability = item.get("ability", "unknown")
    reward_model = item.get("reward_model", {})

    responses = item.get("responses", [])
    scores = item.get("scores", item.get("score", []))
    
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if hasattr(responses, "tolist"):
        responses = responses.tolist()
    
    paired_data = list(zip(responses, scores))

    if args.max_responses is not None and len(paired_data) > args.max_responses:
        paired_data = random.sample(paired_data, args.max_responses)

    converted_items = []
    for response_text, score_val in paired_data:
        if isinstance(score_val, bool) or (isinstance(score_val, np.generic) and isinstance(score_val.item(), bool)):
            reward_val = 1.0 if score_val else 0.0
        else:
            try:
                reward_val = float(score_val)
            except (ValueError, TypeError):
                reward_val = 0.0

        if response_text.endswith("<|im_end|>"):
            response_text = response_text[:-len("<|im_end|>")]
        elif response_text.endswith("<im_end>"):
            response_text = response_text[:-len("<im_end>")]
            
        traj_item = {
            "global_steps": 0,
            "prompt": formatted_prompt,
            "response": response_text,
            "prompt_length": prompt_len,
            "response_length": len(response_text),
            "reward": reward_val,
            "data_source": data_source,
            "ability": ability,
            "reward_model": reward_model,
            "extra_info": item.get("extra_info", {}),
            "hacking_label": args.hacking_label,
            "hacking_type": args.hacking_type,
            "traj_source": args.traj_source
        }
        converted_items.append(traj_item)

    return converted_items

def recursive_to_python(obj):
    if isinstance(obj, dict):
        return {k: recursive_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_python(v) for v in obj]
    elif hasattr(obj, 'tolist'):
        return recursive_to_python(obj.tolist())
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def main():
    args = parse_args()
    
    # [修改 2] 逻辑变更：不再一次性加载 args.input_file，而是作为一个列表遍历
    # 因为 nargs='+' 后，args.input_file 是一个 list: ['file1.parquet', 'file2.parquet']
    input_files = args.input_file 
    output_path = pathlib.Path(args.output_file)
    total_output_items = 0
    
    print(f"Found {len(input_files)} input file(s) to process.")

    with open(output_path, 'w', encoding='utf-8') as f_out:
        # 遍历所有输入文件
        for file_path in input_files:
            current_file_items = load_data_smart(file_path)
            if not current_file_items:
                continue
            
            # 使用 tqdm 显示当前文件的处理进度，desc 中显示文件名
            file_name = pathlib.Path(file_path).name
            for item in tqdm(current_file_items, desc=f"Converting {file_name}"):
                new_items = process_item(item, args)
                for ni in new_items:
                    try:
                        clean_item = recursive_to_python(ni)
                        f_out.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                        total_output_items += 1
                    except TypeError as e:
                        print("\n" + "="*60)
                        print(f"❌ SERIALIZATION ERROR DETECTED! Analyzing data structure...")
                        print("="*60)
                        debug_check_numpy(ni)
                        print("="*60 + "\n")
                        raise e

            # 处理完一个文件后释放内存 (对于大文件很重要)
            del current_file_items

    print(f"\nDone! Processed {len(input_files)} files. Saved {total_output_items} trajectories to {output_path}.")

if __name__ == "__main__":
    main()