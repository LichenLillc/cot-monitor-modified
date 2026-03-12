import json
import argparse
import pathlib
import os
from collections import defaultdict  # [修改点 1] 引入计数器
from tqdm import tqdm
from loguru import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Convert unified .jsonl to directory structure simulating 1_skip and 2a outputs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the unified .jsonl file (output of 0_preprocess.py)")
    parser.add_argument("--base_output_dir", type=str, default="../processed/", help="Base directory to store the structured outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_fp = pathlib.Path(args.input_file)
    base_output_dir = pathlib.Path(args.base_output_dir)
    
    # 1. Determine Output Directory based on Input Filename
    # Example: "my_data.jsonl" -> "base_output_dir/my_data/labels/"
    dataset_name = input_fp.stem
    target_labels_dir = base_output_dir / dataset_name / "labels"
    
    if not input_fp.exists():
        logger.error(f"Input file not found: {input_fp}")
        return

    logger.info(f"Converting {input_fp} -> {target_labels_dir}")
    
    count = 0
    pair_counts = defaultdict(int)  # [修改点 1] 用于记录每个 pair_id 的出现次数 (cot_idx)
    
    with open(input_fp, 'r', encoding='utf-8') as f:
        # Get total lines for tqdm
        lines = f.readlines()
        
        for idx, line in enumerate(tqdm(lines, desc="Generating structure")):
            line = line.strip()
            if not line: continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {idx}")
                continue
            
            # 2. Extract Fields according to user mapping
            prompt_content = item.get("prompt", "")
            # User request: "cot" is empty string
            cot_content = "" 
            # User request: "final_answer" takes "response" content
            final_answer_content = item.get("response", "")
            
            # Metadata to preserve
            hacking_type = item.get("hacking_type", "unknown")
            traj_source = item.get("traj_source", "unknown")
            
            # Safety Label
            hacking_label = item.get("hacking_label", 0)
            safety_score = float(hacking_label) # 0.0 or 1.0

            # [修改点 2] 获取 pair_id，清理非法字符，并分配当前的 cot_idx
            raw_pair_id = str(item.get("pair_id", idx))
            pair_id = "".join([c for c in raw_pair_id if c.isalnum() or c in "-_"])
            
            cot_idx = pair_counts[pair_id]
            pair_counts[pair_id] += 1
            
            # 3. Create Directory for this prompt index
            # Structure: .../labels/{pair_id}/
            prompt_dir = target_labels_dir / pair_id  # [修改点 3] 文件夹名改为 pair_id
            prompt_dir.mkdir(parents=True, exist_ok=True)
            
            # 4. Construct Data Objects
            
            # Object A: Simulating 1_skip output (Raw-ish)
            # File: {pair_id}_{cot_idx}.json
            base_obj = {
                "prompt": prompt_content,
                "cot": cot_content,
                "final_answer": final_answer_content,
                "hacking_type": hacking_type,
                "traj_source": traj_source
                # "raw_prompt" is omitted as requested
            }
            
            # Object B: Simulating 2a_evaluate output (Labeled)
            # File: {pair_id}_{cot_idx}_labeled.json
            labeled_obj = base_obj.copy()
            labeled_obj["safety_label"] = {
                "evaluator": "human",
                "score": safety_score
            }
            
            # 5. Write Files
            # [修改点 3] 文件名改为 pair_id_cotidx 格式
            base_filename = f"{pair_id}_{cot_idx}.json"
            labeled_filename = f"{pair_id}_{cot_idx}_labeled.json"
            
            with open(prompt_dir / base_filename, 'w', encoding='utf-8') as f_base:
                json.dump(base_obj, f_base, indent=2, ensure_ascii=False)
                
            with open(prompt_dir / labeled_filename, 'w', encoding='utf-8') as f_lbl:
                json.dump(labeled_obj, f_lbl, indent=2, ensure_ascii=False)
            
            count += 1

    logger.success(f"Successfully converted {count} items.")
    logger.info(f"Structure created at: {target_labels_dir}")

if __name__ == "__main__":
    main()