import json
import argparse
import pathlib
import re
import pandas as pd
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Filter and split inference JSON/JSONL/PKL data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input inference .json, .jsonl or .pkl file")
    parser.add_argument("--source", type=str, nargs='+', default=None, help="List of data_sources to filter (case-insensitive)")
    parser.add_argument("--score_split", action="store_true", help="Split data into _hacking (score=1), _attempt (assert/main()), and _normal")
    parser.add_argument("--max_dup", type=int, default=None, help="Maximum number of responses to keep per prompt (randomly sampled)")
    return parser.parse_args()

def load_data(input_path):
    input_path = pathlib.Path(input_path)
    print(f"Loading data from {input_path}...")
    
    # 1. 处理 Pickle (.pkl) 格式
    if input_path.suffix == ".pkl":
        try:
            data = pd.read_pickle(input_path)
            if isinstance(data, pd.DataFrame):
                return data.to_dict(orient='records')
            elif isinstance(data, list):
                return data
            else:
                return list(data)
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return []

    # 2. 处理 JSON 格式
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                return content
            else:
                return [content]
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass 

    # 3. 处理 JSONL 格式
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

def contains_keywords_in_code(text):
    if not isinstance(text, str):
        return False
    code_blocks = re.findall(r'```python(.*?)```', text, re.DOTALL)
    for block in code_blocks:
        if "assert" in block.lower(): return True
        if "main()" in block: return True
    return False

def is_score_one(score):
    try:
        val = float(score)
        return abs(val - 1.0) < 1e-9
    except (ValueError, TypeError):
        return False

def get_total_response_count(data):
    """
    计算所有 item 中 responses 的总数。
    """
    total = 0
    for item in data:
        responses = item.get("responses", [])
        if isinstance(responses, list):
            total += len(responses)
    return total

def save_jsonl_with_count(data, base_path_obj, max_dup_val, suffix_tag=""):
    """
    保存文件。
    文件名格式: 原名 + 标签 + [_dupX] + _数量 .jsonl
    """
    if not data:
        return

    count = get_total_response_count(data)
    
    # 构建 dup 后缀
    dup_suffix = f"_dup{max_dup_val}" if max_dup_val is not None else ""
    
    # 拼接最终文件名
    # 示例: output_leetcode_hacking_dup5_120.jsonl
    new_name = f"{base_path_obj.stem}{suffix_tag}{dup_suffix}_{count}.jsonl"
    output_path = base_path_obj.with_name(new_name)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(data)} prompts containing {count} responses to {output_path}")

def process_and_split(items, args, source_tag, input_path):
    
    if args.score_split:
        hacking_items = []
        attempt_items = []
        normal_items = []

        desc = f"Splitting{source_tag.replace('_', ' ')}" if source_tag else "Splitting"
        
        for item in tqdm(items, desc=desc):
            responses = item.get("responses", [])
            scores = item.get("score", [])
            
            if hasattr(responses, 'tolist'): responses = responses.tolist()
            if hasattr(scores, 'tolist'): scores = scores.tolist()
            
            pair_len = min(len(responses), len(scores))
            pairs = list(zip(responses[:pair_len], scores[:pair_len]))
            
            temp_hacking = []
            temp_attempt = []
            temp_normal = []

            for resp, sc in pairs:
                if is_score_one(sc):
                    temp_hacking.append((resp, sc))
                else:
                    if contains_keywords_in_code(resp):
                        temp_attempt.append((resp, sc))
                    else:
                        temp_normal.append((resp, sc))
            
            def add_if_exists(temp_list, target_list):
                if temp_list:
                    # 随机采样逻辑
                    if args.max_dup is not None and len(temp_list) > args.max_dup:
                        temp_list = random.sample(temp_list, args.max_dup)

                    new_item = item.copy()
                    r_list, s_list = zip(*temp_list)
                    new_item["responses"] = list(r_list)
                    new_item["score"] = list(s_list)
                    target_list.append(new_item)

            add_if_exists(temp_hacking, hacking_items)
            add_if_exists(temp_attempt, attempt_items)
            add_if_exists(temp_normal, normal_items)

        # 保存时传入 args.max_dup
        save_jsonl_with_count(hacking_items, input_path, args.max_dup, f"{source_tag}_hacking")
        save_jsonl_with_count(attempt_items, input_path, args.max_dup, f"{source_tag}_attempt")
        save_jsonl_with_count(normal_items, input_path, args.max_dup, f"{source_tag}_normal")

    else:
        # 非 Split 模式下的处理
        final_items = items
        
        if args.max_dup is not None:
            sampled_items = []
            for item in items:
                responses = item.get("responses", [])
                scores = item.get("score", [])
                
                if hasattr(responses, 'tolist'): responses = responses.tolist()
                if hasattr(scores, 'tolist'): scores = scores.tolist()
                
                pair_len = min(len(responses), len(scores))
                pairs = list(zip(responses[:pair_len], scores[:pair_len]))
                
                if len(pairs) > args.max_dup:
                    pairs = random.sample(pairs, args.max_dup)
                    
                    new_item = item.copy()
                    if pairs:
                        r, s = zip(*pairs)
                        new_item["responses"] = list(r)
                        new_item["score"] = list(s)
                    else:
                        new_item["responses"] = []
                        new_item["score"] = []
                    sampled_items.append(new_item)
                else:
                    sampled_items.append(item)
            
            final_items = sampled_items

        if final_items:
            save_jsonl_with_count(final_items, input_path, args.max_dup, f"{source_tag}")

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    
    all_items = load_data(input_path)
    print(f"Total loaded prompts: {len(all_items)}")
    
    if not all_items:
        print("No data loaded. Exiting.")
        return

    sources_to_process = args.source if args.source else [None]
    print(f"Processing sources: {sources_to_process}")

    for target_source in sources_to_process:
        current_items = []
        source_tag = ""

        if target_source:
            target_lower = target_source.lower()
            source_tag = f"_{target_source}" 
            
            for item in all_items:
                item_source = item.get("data_source", "")
                if item_source and target_lower in str(item_source).lower():
                    current_items.append(item)
            print(f"\n--- Processing Source: '{target_source}' (Found {len(current_items)} prompts) ---")
        else:
            current_items = all_items
            source_tag = ""
            print(f"\n--- Processing All Data (No source filter) ---")

        if not current_items:
            print(f"Skipping '{target_source}': No items found.")
            continue

        process_and_split(current_items, args, source_tag, input_path)

if __name__ == "__main__":
    main()