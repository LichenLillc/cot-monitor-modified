import json
import argparse
import random
import os
import hashlib
from collections import defaultdict
import pathlib

def load_jsonl(path):
    """Load jsonl file into a list of objects"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def get_normalized_source(item):
    """Normalize data source to strictly 'leetcode', 'taco', or 'other'."""
    raw_ds = str(item.get("data_source", "unknown")).strip().lower()
    if "leetcode" in raw_ds:
        return "leetcode"
    elif "taco" in raw_ds:
        return "taco"
    else:
        return "other"

def get_pair_id(item):
    """Extract global unique pair ID based on normalized data_source and index."""
    ds = get_normalized_source(item)
    idx = item.get("extra_info", {}).get("index", "")
    if idx:
        return f"{ds}_{idx}"
    
    # Fallback for old data without extra_info.index: MD5 hash of prompt
    prompt_text = item.get("prompt", "")
    if isinstance(prompt_text, list) and len(prompt_text) > 0 and isinstance(prompt_text[0], dict):
        prompt_text = prompt_text[0].get("content", "")
    return hashlib.md5(str(prompt_text).encode('utf-8')).hexdigest()[:10]

def analyze_data_distribution(grouped_data):
    """Analyze and print exact distributions of normal/hacking items per pair_id to debug anomalies."""
    print("\n" + "=" * 80)
    print(" DATA ANOMALY / DUPLICATE ANALYSIS ".center(80, "="))
    print("=" * 80)
    
    distribution = defaultdict(int)
    anomalies = []
    
    for pid, data in grouped_data.items():
        n_len = len(data['normal'])
        h_len = len(data['hacking'])
        
        # Form signature like "1 Normal : 1 Hacking", "2 Normal : 0 Hacking" etc.
        sig = f"{n_len} Normal : {h_len} Hacking"
        distribution[sig] += 1
        
        # We expect at most 1 Normal and 1 Hacking per pair_id in your specific use case.
        # Record anomalies if there are multiple normals or multiple hackings in the SAME pair_id
        if n_len > 1 or h_len > 1:
            anomalies.append({
                'pair_id': pid,
                'normal_count': n_len,
                'hacking_count': h_len,
                # Use set to find out unique files these duplicates come from
                'normal_files': list(set([pathlib.Path(f).name for _, f in data['normal']])),
                'hacking_files': list(set([pathlib.Path(f).name for _, f in data['hacking']]))
            })
            
    print(f"Distribution of items per Pair ID (Context):")
    for sig, count in sorted(distribution.items(), key=lambda x: x[0]):
        print(f"  - [{sig:<25}] -> {count} pairs")
        
    if anomalies:
        print(f"\n[!] WARNING: Found {len(anomalies)} Pair IDs with unexpected duplicate items!")
        print("Showing up to 10 examples of these anomalies for your debugging:")
        for i, a in enumerate(anomalies[:10]):
            print(f"  Example {i+1} | pair_id: {a['pair_id']}")
            print(f"    - Normal items  ({a['normal_count']}) found in files : {a['normal_files']}")
            print(f"    - Hacking items ({a['hacking_count']}) found in files: {a['hacking_files']}")
    else:
        print("\n[✓] PERFECT: All pair IDs contain exactly 0 or 1 Normal, and 0 or 1 Hacking item.")
        
    print("=" * 80 + "\n")

def print_dry_run_stats(grouped_data):
    """Print detailed statistics for Dry Run mode."""
    stats = defaultdict(lambda: {
        'total_items': 0, 'valid_pairs': 0, 
        'paired_normal': 0, 'paired_hacking': 0, 
        'orphan_items': 0
    })

    for pid, data in grouped_data.items():
        sample_item = (data['normal'] + data['hacking'])[0][0]
        ds_key = get_normalized_source(sample_item).capitalize()

        num_normal = len(data['normal'])
        num_hacking = len(data['hacking'])
        total_in_group = num_normal + num_hacking
        
        stats[ds_key]['total_items'] += total_in_group

        if num_normal > 0 and num_hacking > 0:
            stats[ds_key]['valid_pairs'] += 1
            stats[ds_key]['paired_normal'] += num_normal
            stats[ds_key]['paired_hacking'] += num_hacking
        else:
            stats[ds_key]['orphan_items'] += total_in_group

    print("\n" + "=" * 80)
    print(" PAIRED DATASET OVERVIEW ".center(80, "="))
    print("=" * 80)
    header = f"| {'Data Source':<15} | {'Total Items':<12} | {'Valid Pairs':<12} | {'Paired (N/H)':<15} | {'Orphans':<10} |"
    print(header)
    print("-" * 80)

    for ds, s in stats.items():
        paired_str = f"{s['paired_normal']} / {s['paired_hacking']}"
        row = f"| {ds:<15} | {s['total_items']:<12} | {s['valid_pairs']:<12} | {paired_str:<15} | {s['orphan_items']:<10} |"
        print(row)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build Paired Dataset: Merge and Split into Train/Test sets based on Context Pair IDs.")
    
    parser.add_argument("--inputs", '-i', nargs="+", required=True, help="List of input jsonl files.")
    parser.add_argument("--output_dir", '-o', type=str, default="/home/Lichen/cot-monitor-modified/main_table3_paired/exp_data", help="Output directory for train and test files.")
    parser.add_argument("--train_name", type=str, default="7b_pfc_think-ins_cot_paired-ln900-lsh900.jsonl", help="Filename for the merged training dataset.")
    parser.add_argument("--max_train_pairs", '-n', type=int, default=900, 
                        help="Maximum number of PAIRS to sample PER DATA SOURCE for the training set. If not set, uses ALL valid pairs.")
    parser.add_argument("--max_test_num", type=int, default=500, 
                        help="Maximum number of orphan items to keep in each test set (default: 500).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--shuffle", '-s', action="store_true", help="If set, shuffle the merged MAIN list.")
    parser.add_argument("--dry_run", action="store_true", help="Print pairing statistics without saving any files.")

    args = parser.parse_args()

    grouped_data = defaultdict(lambda: {'normal': [], 'hacking': []})
    
    print(f"Loading {len(args.inputs)} input files...")
    for filepath in args.inputs:
        data = load_jsonl(filepath)
        for item in data:
            pid = get_pair_id(item)
            is_hacking = item.get("hacking_label", 0) == 1
            if is_hacking:
                grouped_data[pid]['hacking'].append((item, filepath))
            else:
                grouped_data[pid]['normal'].append((item, filepath))

    # 分析并打印异常数据
    analyze_data_distribution(grouped_data)

    if args.dry_run:
        print_dry_run_stats(grouped_data)
        print("Dry run completed. No files were written.")
        return

    print_dry_run_stats(grouped_data)

    valid_pair_ids_by_ds = defaultdict(list)
    orphan_pair_ids = []
    
    for pid, data in grouped_data.items():
        if len(data['normal']) > 0 and len(data['hacking']) > 0:
            sample_item = data['normal'][0][0]
            ds = get_normalized_source(sample_item)
            valid_pair_ids_by_ds[ds].append(pid)
        else:
            orphan_pair_ids.append(pid)

    selected_train_pids = []
    random.seed(args.seed)
    
    print("--- Training Set Sampling ---")
    for ds, pids in valid_pair_ids_by_ds.items():
        pids.sort()
        random.shuffle(pids)

        if args.max_train_pairs is not None and args.max_train_pairs < len(pids):
            selected = pids[:args.max_train_pairs]
            unselected = pids[args.max_train_pairs:]
            
            selected_train_pids.extend(selected)
            orphan_pair_ids.extend(unselected)
            print(f"[{ds.capitalize()}] Sampled {args.max_train_pairs} pairs out of {len(pids)} valid pairs.")
        else:
            selected_train_pids.extend(pids)
            print(f"[{ds.capitalize()}] Using ALL {len(pids)} valid pairs.")

    train_items = []
    test_items_by_file = defaultdict(list)

    for pid in selected_train_pids:
        for item, _ in grouped_data[pid]['normal'] + grouped_data[pid]['hacking']:
            train_items.append(item)

    for pid in orphan_pair_ids:
        for item, filepath in grouped_data[pid]['normal'] + grouped_data[pid]['hacking']:
            test_items_by_file[filepath].append(item)

    if args.shuffle:
        random.seed(0)
        random.shuffle(train_items)

    os.makedirs(args.output_dir, exist_ok=True)
    
    train_output_path = os.path.join(args.output_dir, args.train_name)
    print(f"\nWriting Train dataset ({len(train_items)} items) to {train_output_path}...")
    with open(train_output_path, "w", encoding="utf-8") as fout:
        for item in train_items:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Writing Test datasets (Orphans) back to their sources...")
    for filepath, items in test_items_by_file.items():
        if not items:
            continue
            
        if args.max_test_num is not None and len(items) > args.max_test_num:
            rng = random.Random(args.seed)
            rng.shuffle(items)
            items = items[:args.max_test_num]
            
        original_filename = pathlib.Path(filepath).name
        root, ext = os.path.splitext(original_filename)
        
        # [修改点] 将实际写入的 items 数量加到测试集文件名后缀中
        test_filename = f"{root}_test_{len(items)}{ext}"
        test_output_path = os.path.join(args.output_dir, test_filename)
        
        with open(test_output_path, "w", encoding="utf-8") as fout:
            for item in items:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  -> Saved {len(items):<5} test items to: {test_filename}")

    print("\nAll done! Successfully built paired training set and isolated test sets.")

if __name__ == "__main__":
    main()