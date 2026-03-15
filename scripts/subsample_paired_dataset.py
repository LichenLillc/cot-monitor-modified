import json
import argparse
import random
import hashlib
import pathlib
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Subsample paired/unpaired dataset into smaller strict subsets.")
    parser.add_argument("--input_file", '-i', type=str, required=True, help="Path to the input .jsonl file")
    parser.add_argument("--sizes", '-s', type=int, nargs='+', required=True, help="Target sizes for subsampling (e.g. 200 500)")
    parser.add_argument("--prefix", '-p', type=str, default="qwen_scratch-ckpt61_unpaired", help="Prefix for the output filename")
    parser.add_argument("--data_source", '-ds', type=str, default=None, help="If provided, only keep data where 'data_source' contains this string (case-insensitive).")
    parser.add_argument("--dry_run", '-dr', action="store_true", help="If set, only print statistics and exit without writing files")
    parser.add_argument("--seed", '-seed', type=int, default=42, help="Random seed for reproducibility")
    # [新增参数] 专门提取 Unpaired 数据
    parser.add_argument("--unpaired", action="store_true", help="If set, strictly extract from UNPAIRED data only, ignoring paired groups.")
    return parser.parse_args()

def get_category(item):
    """Categorize the item into one of the 6 specific buckets, or 'other'."""
    ds = str(item.get("data_source", "unknown")).lower()
    lbl = item.get("hacking_label", 0)
    htype = str(item.get("hacking_type", "unknown")).lower()

    if "leetcode" in ds:
        prefix = "l"
    elif "taco" in ds:
        prefix = "t"
    else:
        return "other"

    if lbl == 0:
        return f"{prefix}n"
    elif lbl == 1:
        if "unittest" in htype:
            return f"{prefix}uh"
        elif "exit" in htype:
            return f"{prefix}eh"
        elif "skip" in htype:
            return f"{prefix}sh"
        else:
            return "other" # Hacking label 1 but unknown hacking type
    
    return "other"

def get_group_id(item):
    """Reconstruct pair_id from data_source and extra_info.index."""
    ds = str(item.get("data_source", "unknown")).strip()
    idx = item.get("extra_info", {}).get("index", None)
    
    if idx is not None and idx != "":
        return f"{ds}_{idx}"
        
    prompt_text = item.get("prompt", "")
    if isinstance(prompt_text, list) and len(prompt_text) > 0 and isinstance(prompt_text[0], dict):
        prompt_text = prompt_text[0].get("content", "")
    return hashlib.md5(str(prompt_text).encode('utf-8')).hexdigest()[:10]

def main():
    args = parse_args()
    random.seed(args.seed)
    
    # 1. Read, Filter and Group Data
    groups = defaultdict(list)
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            item = json.loads(line)
            
            if args.data_source:
                ds_str = str(item.get("data_source", "")).lower()
                if args.data_source.lower() not in ds_str:
                    continue
                    
            groups[get_group_id(item)].append(item)

    # 动态侦测全局需要的完整配对组合 (Global Requirements)
    global_cats = {get_category(i) for items in groups.values() for i in items}
    req_l = {c for c in global_cats if c.startswith('l') and c != 'other'}
    req_t = {c for c in global_cats if c.startswith('t') and c != 'other'}
    
    # 判定规则：必须有 normal，且必须至少有 1 种 hacking
    valid_l_pair = ('ln' in req_l) and any(c != 'ln' for c in req_l)
    valid_t_pair = ('tn' in req_t) and any(c != 'tn' for c in req_t)

    # 2. Separate into Paired Groups and Unpaired Items
    paired_groups = []
    unpaired_items = defaultdict(list)
    
    for gid, items in groups.items():
        group_cats = {get_category(i) for i in items}
        
        is_paired = False
        if valid_l_pair and any(c.startswith('l') for c in group_cats):
            if req_l.issubset(group_cats):
                is_paired = True
        elif valid_t_pair and any(c.startswith('t') for c in group_cats):
            if req_t.issubset(group_cats):
                is_paired = True
                
        if is_paired:
            paired_groups.append(items)
        else:
            for item in items:
                unpaired_items[get_category(item)].append(item)

    # 3. Shuffle (Done exactly once to guarantee strict subset property)
    random.shuffle(paired_groups)
    for cat in unpaired_items:
        random.shuffle(unpaired_items[cat])

    # 4. 统计可用数量
    avail_paired = defaultdict(int)
    for group in paired_groups:
        group_cats = {get_category(i) for i in group}
        for cat in group_cats:
            if cat != 'other':
                avail_paired[cat] += 1
            
    avail_unpaired = {cat: len(lst) for cat, lst in unpaired_items.items()}

    all_cats = ['ln', 'luh', 'leh', 'lsh', 'tn', 'tuh', 'teh', 'tsh']
    print("\n" + "="*63)
    print(f"📊 DATASET STATISTICS (Seed: {args.seed})")
    if args.data_source:
        print(f"🔍 Filtered by data_source: '{args.data_source}'")
    if args.unpaired:
        print("⚠️  UNPAIRED MODE ACTIVATED: Extracting from Unpaired Data Only")
    print("="*63)
    for cat in all_cats:
        print(f"[{cat.upper():<3}] -> Paired (Groups): {avail_paired[cat]:<5} | Unpaired (Items): {avail_unpaired.get(cat, 0):<5}")
        
    count_other_paired = avail_paired.get("other", 0)
    count_other_unpaired = avail_unpaired.get("other", 0)
    has_other = count_other_paired > 0 or count_other_unpaired > 0
    
    if has_other:
        print("-" * 63)
        print(f"[⚠️OTH] -> Paired (Groups): {count_other_paired:<5} | Unpaired (Items): {count_other_unpaired:<5}")
    print("="*63 + "\n")
    
    if not args.data_source and has_other:
        raise ValueError(f"CRITICAL ERROR: Found {count_other_paired + count_other_unpaired} items/groups that cannot be classified into LeetCode/Taco or have an unknown hacking_type.")

    if args.dry_run:
        print("🛑 Dry run activated. Exiting without generating files.")
        return

    # 5. Process Each Target Size
    sizes = sorted(args.sizes)
    output_dir = pathlib.Path(args.input_file).parent

    for K in sizes:
        print(f"🚀 Processing target size: {K}...")
        
        # =====================================================================
        # 智能报错分流：区分 Paired 模式与 Unpaired 模式
        # =====================================================================
        for cat in all_cats:
            if args.unpaired:
                # 【Unpaired 模式】只盯着散件池看，有散件但不够 K 就报错
                if 0 < avail_unpaired.get(cat, 0) < K:
                    raise ValueError(f"CRITICAL ERROR: Requested {K} items for unpaired '{cat}' but only {avail_unpaired.get(cat, 0)} available!")
            else:
                # 【Paired 模式】(之前的逻辑)
                if avail_paired[cat] > 0:
                    if avail_paired[cat] < K:
                        raise ValueError(f"CRITICAL ERROR: Requested {K} groups for '{cat}' but only {avail_paired[cat]} complete groups available!")
                else:
                    if 0 < avail_unpaired.get(cat, 0) < K:
                        raise ValueError(f"CRITICAL ERROR: Requested {K} items for unpaired '{cat}' but only {avail_unpaired.get(cat, 0)} available!")

        picked_items = []
        
        # --- A. Extract from Paired Groups (按组抽取) ---
        if not args.unpaired:
            paired_counts = defaultdict(int)
            for group in paired_groups:
                unique_cat_items = {}
                for item in group:
                    cat = get_category(item)
                    if cat not in unique_cat_items:
                        unique_cat_items[cat] = item
                
                group_cats = set(unique_cat_items.keys())
                
                need_this_group = False
                for cat in group_cats:
                    if cat in all_cats and paired_counts[cat] < K and avail_paired[cat] > 0:
                        need_this_group = True
                        break
                
                if need_this_group:
                    picked_items.extend(unique_cat_items.values())
                    for cat in group_cats:
                        if cat in all_cats:
                            paired_counts[cat] += 1

        # --- B. Extract from Unpaired Items (按条抽取) ---
        unpaired_counts = defaultdict(int)
        for cat in all_cats:
            if args.unpaired:
                # 【Unpaired 模式】只要散件池里有数据，就提取 K 个
                if avail_unpaired.get(cat, 0) > 0:
                    chunk = unpaired_items[cat][:K]
                    picked_items.extend(chunk)
                    unpaired_counts[cat] = len(chunk)
            else:
                # 【Paired 模式】只有当该类别完全没有配对组，且存在散件时才提取
                if avail_paired[cat] == 0 and avail_unpaired.get(cat, 0) > 0:
                    chunk = unpaired_items[cat][:K]
                    picked_items.extend(chunk)
                    unpaired_counts[cat] = len(chunk)

        # 6. Generate Filename and Save
        total_counts = defaultdict(int)
        for item in picked_items:
            cat = get_category(item)
            if cat in all_cats:
                total_counts[cat] += 1
            
        suffix_parts = []
        for cat in all_cats:
            if total_counts[cat] > 0:
                suffix_parts.append(f"{cat}{total_counts[cat]}")
                
        suffix = "-" + "-".join(suffix_parts) if suffix_parts else ""
        out_filename = f"{args.prefix}{suffix}.jsonl"
        out_path = output_dir / out_filename

        with open(out_path, 'w', encoding='utf-8') as f_out:
            for item in picked_items:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"✅ Generated subset '{out_filename}' ({len(picked_items)} total rows).")
        
    print("\n🎉 All subsets generated successfully!")

if __name__ == "__main__":
    main()