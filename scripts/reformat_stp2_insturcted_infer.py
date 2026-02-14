import json
import argparse
import pathlib
import re
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Post-process prompts: Regex replace and append suffix.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file (from previous step)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save processed .jsonl file")
    return parser.parse_args()

def process_line(line, stats):
    """
    处理单行 JSON 数据
    """
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        return None

    if "prompt" not in item:
        return item

    original_prompt = item["prompt"]
    
    # --- 任务 1: 正则替换 ---
    # 定义起始和结束标记
    # re.escape 用于自动转义特殊字符，但在中间部分我们需要 .*? 来匹配任意内容
    # 所以我们手动构建 Pattern
    
    start_marker = "IMPORTANT DIRECTIVE:"
    end_marker = "use these high-level directions to guide your approach."
    
    # 目标替换文本
    replacement_text = (
        "INSTRUCTION: Always include a detailed reasoning process before your final answer. "
        "Your reasoning should explain step by step how you reach the conclusion, not just give a brief statement."
    )

    # 构建正则: 
    # re.DOTALL (flag=re.S) 让 '.' 能够匹配换行符，因为 Directive 可能是多行的
    pattern_str = r"IMPORTANT DIRECTIVE:.*?use these high-level directions to guide your approach\."
    
    # subn 执行替换并返回 (新字符串, 替换次数)
    new_prompt, count = re.subn(pattern_str, replacement_text, original_prompt, flags=re.DOTALL)
    
    if count > 0:
        stats["replaced_count"] += 1
    
    # --- 任务 2: 添加后缀 ---
    suffix = "### Answer: (use the provided format with backticks)\n\n\nLet's think step by step:"
    
    # 检查是否已经包含了后缀（防止重复运行脚本导致多次添加），如果没包含则添加
    if not new_prompt.endswith(suffix.strip()): 
        new_prompt += suffix
    
    # --- 更新数据 ---
    item["prompt"] = new_prompt
    
    # 重要：因为修改了 prompt，必须同步更新 prompt_length
    if "prompt_length" in item:
        item["prompt_length"] = len(new_prompt)
        
    return item

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    output_path = pathlib.Path(args.output_file)
    
    print(f"Reading from: {input_path}")
    print(f"Writing to:   {output_path}")

    stats = {
        "total_lines": 0,
        "replaced_count": 0,
        "processed_lines": 0
    }

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        # 使用 tqdm 显示进度
        for line in tqdm(f_in, desc="Refining Prompts"):
            line = line.strip()
            if not line: continue
            
            stats["total_lines"] += 1
            processed_item = process_line(line, stats)
            
            if processed_item:
                f_out.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
                stats["processed_lines"] += 1

    print("-" * 30)
    print("Processing Complete!")
    print(f"Total Lines Read:      {stats['total_lines']}")
    print(f"Successfully Written:  {stats['processed_lines']}")
    print(f"Prompts Modified (Replaced Directive): {stats['replaced_count']}")
    print("-" * 30)

if __name__ == "__main__":
    main()