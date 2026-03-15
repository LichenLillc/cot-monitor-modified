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
    parser.add_argument("--max_dup", type=int, default=1, help="Maximum number of responses to keep per prompt (randomly sampled)")
    
    # 新增参数
    parser.add_argument("--hacking_type", type=str, nargs='+', default=None, help="List of keywords to sub-categorize hacking responses (case-insensitive)")
    parser.add_argument("--dry_run", action="store_true", help="Only compute stats, do not save files except for overlap and other categories")
    return parser.parse_args()

def load_data(input_path):
    input_path = pathlib.Path(input_path)
    print(f"Loading data from {input_path}...")
    
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

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                return content
            else:
                return [content]
    except (json.JSONDecodeError, UnicodeDecodeError):
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

def extract_python_code(text):
    """提取 ```python 和 ``` 之间的内容，仅返回最后一个代码块"""
    if not isinstance(text, str):
        return ""
    # 严格按照区分大小写的```python\n和```来提取代码块
    code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    
    # 如果匹配到多个代码块，只返回最后一个！
    if code_blocks:
        return code_blocks[-1]
    return ""

def remove_python_comments(code):
    """
    清理代码，移除 ''' 和 \"\"\" 包裹的多行注释，以及 '#' 及其后面的单行注释。
    """
    # 1. 先用正则移除多行注释 (匹配 \"\"\"...\"\"\" 和 '''...''')
    # 使用 re.DOTALL 使 . 能匹配换行符，非贪婪模式 (?) 防止跨越多个注释块匹配
    code = re.sub(r'\"\"\"(.*?)\"\"\"', '', code, flags=re.DOTALL)
    code = re.sub(r"'''(.*?)'''", '', code, flags=re.DOTALL)
    
    # 2. 逐行清理 '#' 注释
    cleaned_lines = []
    for line in code.split('\n'):
        # split('#', 1) 截取 '#' 前面的真实代码部分
        cleaned_lines.append(line.split('#', 1)[0])
    
    # 重新拼接成纯净的代码块
    return '\n'.join(cleaned_lines)

def contains_keywords_in_code(text):
    raw_code = extract_python_code(text)
    # 过滤掉注释后再转小写
    code_no_comments = remove_python_comments(raw_code).lower()
    
    if "assert" in code_no_comments: return True
    if "main()" in code_no_comments: return True
    return False

def get_hacking_category(text, keywords):
    """根据提取的代码和关键词列表，返回分类类型"""
    raw_code = extract_python_code(text)
    if not raw_code.strip():
        return "other"
    
    # 过滤掉注释后再转小写进行匹配
    code_no_comments = remove_python_comments(raw_code).lower()
    
    matched = []
    for kw in keywords:
        if kw.lower() in code_no_comments:
            matched.append(kw)
            
    if len(matched) == 0:
        return "other"
    elif len(matched) == 1:
        return matched[0]
    else:
        return "overlap"

def is_score_one(score):
    try:
        val = float(score)
        return abs(val - 1.0) < 1e-9
    except (ValueError, TypeError):
        return False

def get_total_response_count(data):
    total = 0
    for item in data:
        responses = item.get("responses", [])
        if isinstance(responses, list):
            total += len(responses)
    return total

def save_jsonl_with_count(data, base_path_obj, max_dup_val, suffix_tag="", dry_run=False):
    """
    保存文件，并返回统计信息用于外部美化打印。
    返回: (prompt数量, response数量, 最终文件名, 是否真实保存)
    """
    p_count = len(data)
    r_count = get_total_response_count(data)
    
    if not data:
        return p_count, r_count, "-", False

    dup_suffix = f"_dup{max_dup_val}" if max_dup_val is not None else ""
    new_name = f"{base_path_obj.stem}{suffix_tag}{dup_suffix}_{r_count}.jsonl"
    output_path = base_path_obj.with_name(new_name)

    is_saved = False
    
    if dry_run:
        # 如果是 dry_run，只有 overlap 和 other 才允许写入文件
        if suffix_tag.endswith("_overlap") or suffix_tag.endswith("_other"):
            is_saved = True
    else:
        is_saved = True

    if is_saved:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return p_count, r_count, new_name, is_saved

def print_summary_table(source_name, stats_records):
    """
    打印美观的统计表格
    stats_records 格式: [(类别名, Prompts数, Responses数, 文件名, 是否保存), ...]
    """
    title = f" SUMMARY FOR SOURCE: {source_name if source_name else 'ALL'} "
    print("\n" + "=" * 105)
    print(title.center(105, "="))
    print("=" * 105)
    
    header = f"| {'Category':<25} | {'Prompts':<10} | {'Responses':<10} | {'Status':<12} | {'Filename':<35} |"
    print(header)
    print("-" * 105)
    
    for category, p_count, r_count, filename, is_saved in stats_records:
        if p_count == 0:
            continue # 跳过没有数据的类别
            
        status = "[SAVED]" if is_saved else "[DRY RUN]"
        # 裁剪文件名防止过长打乱格式
        display_fname = filename if len(filename) <= 35 else filename[:32] + "..."
        
        # 将原始后缀 (如 _leetcode_hacking_other) 稍微清理下以便显示
        display_cat = category.strip("_") 
        
        row = f"| {display_cat:<25} | {p_count:<10} | {r_count:<10} | {status:<12} | {display_fname:<35} |"
        print(row)
        
    print("=" * 105 + "\n")

def process_and_split(items, args, source_tag, input_path):
    stats_records = [] # 用于收集统计信息以打印表格

    if args.score_split:
        attempt_items = []
        normal_items = []

        # 动态初始化 hacking 字典
        hacking_dict = {}
        if args.hacking_type:
            hacking_dict["other"] = []
            hacking_dict["overlap"] = []
            for kw in args.hacking_type:
                hacking_dict[kw] = []
        else:
            hacking_dict["default"] = [] 

        desc = f"Splitting{source_tag.replace('_', ' ')}" if source_tag else "Splitting"
        
        for item in tqdm(items, desc=desc):
            responses = item.get("responses", [])
            scores = item.get("score", [])
            
            if hasattr(responses, 'tolist'): responses = responses.tolist()
            if hasattr(scores, 'tolist'): scores = scores.tolist()
            
            pair_len = min(len(responses), len(scores))
            pairs = list(zip(responses[:pair_len], scores[:pair_len]))
            
            temp_attempt = []
            temp_normal = []
            
            # 为当前 prompt 初始化临时字典
            temp_hacking_typed = {k: [] for k in hacking_dict.keys()}

            for resp, sc in pairs:
                if is_score_one(sc):
                    if args.hacking_type:
                        cat = get_hacking_category(resp, args.hacking_type)
                        temp_hacking_typed[cat].append((resp, sc))
                    else:
                        temp_hacking_typed["default"].append((resp, sc))
                else:
                    if contains_keywords_in_code(resp):
                        temp_attempt.append((resp, sc))
                    else:
                        temp_normal.append((resp, sc))
            
            def add_if_exists(temp_list, target_list):
                if temp_list:
                    if args.max_dup is not None and len(temp_list) > args.max_dup:
                        temp_list = random.sample(temp_list, args.max_dup)

                    new_item = item.copy()
                    r_list, s_list = zip(*temp_list)
                    new_item["responses"] = list(r_list)
                    new_item["score"] = list(s_list)
                    target_list.append(new_item)

            for k, lst in temp_hacking_typed.items():
                add_if_exists(lst, hacking_dict[k])
                
            add_if_exists(temp_attempt, attempt_items)
            add_if_exists(temp_normal, normal_items)

        # 收集结果并保存
        if args.hacking_type:
            for k, items_list in hacking_dict.items():
                cat_name = f"{source_tag}_hacking_{k}"
                res = save_jsonl_with_count(items_list, input_path, args.max_dup, cat_name, args.dry_run)
                stats_records.append((cat_name, *res))
        else:
            cat_name = f"{source_tag}_hacking"
            res = save_jsonl_with_count(hacking_dict["default"], input_path, args.max_dup, cat_name, args.dry_run)
            stats_records.append((cat_name, *res))

        cat_attempt = f"{source_tag}_attempt"
        res_attempt = save_jsonl_with_count(attempt_items, input_path, args.max_dup, cat_attempt, args.dry_run)
        stats_records.append((cat_attempt, *res_attempt))

        cat_normal = f"{source_tag}_normal"
        res_normal = save_jsonl_with_count(normal_items, input_path, args.max_dup, cat_normal, args.dry_run)
        stats_records.append((cat_normal, *res_normal))

    else:
        # 针对未开启 score_split 的情况
        if args.hacking_type:
            print("[Warning] --hacking_type is ignored because --score_split is not enabled.")
            
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
            cat_name = f"{source_tag}" if source_tag else "all_data"
            res = save_jsonl_with_count(final_items, input_path, args.max_dup, cat_name, args.dry_run)
            stats_records.append((cat_name, *res))

    # 打印最终的统计表格
    source_display = source_tag.strip("_") if source_tag else "ALL"
    print_summary_table(source_display, stats_records)

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    
    all_items = load_data(input_path)
    print(f"Total loaded prompts: {len(all_items)}")
    
    if not all_items:
        print("No data loaded. Exiting.")
        return

    sources_to_process = args.source if args.source else [None]
    if args.source:
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
            print(f"\n--- Extracting Source: '{target_source}' (Found {len(current_items)} prompts) ---")
        else:
            current_items = all_items
            source_tag = ""
            print(f"\n--- Extracting All Data (No source filter) ---")

        if not current_items:
            print(f"Skipping '{target_source}': No items found.")
            continue

        process_and_split(current_items, args, source_tag, input_path)

if __name__ == "__main__":
    main()