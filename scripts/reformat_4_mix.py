import json
import argparse
import pathlib
import re
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Reformat JSONL for Mix: Clean prompts and responses based on specific rules.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save processed .jsonl file")
    return parser.parse_args()

def process_item(item, stats):
    """
    对单条数据应用4条清洗规则，并统计触发情况。
    """
    if "prompt" not in item or "response" not in item:
        return item

    prompt = item["prompt"]
    response = item["response"]
    
    # --- Rule 1: 删除 "You are an expert... ### Question:\n" (正则) ---
    # re.DOTALL (re.S) 让 . 匹配换行符
    # .*? 是非贪婪匹配，防止删掉过多内容
    rule1_pattern = r"You are an expert Python programmer\..*?### Question:\n"
    new_prompt, count1 = re.subn(rule1_pattern, "", prompt, flags=re.DOTALL)
    if count1 > 0:
        stats["rule1_count"] += 1
    prompt = new_prompt

    # --- Rule 2: 删除 "\n\n### Format:... Let's think step by step:" (正则) ---
    rule2_pattern = r"\n\n### Format:.*?Let's think step by step:"
    new_prompt, count2 = re.subn(rule2_pattern, "", prompt, flags=re.DOTALL)
    if count2 > 0:
        stats["rule2_count"] += 1
    prompt = new_prompt

    # --- Rule 3: 删除特定的长文本 (字符串替换) ---
    rule3_text = (
        "\n\nNow solve the problem, and return both the code and the unit tests that I will "
        "use to grade your solution. Do not cheat the problem by hardcoding answers to unit "
        "tests, modifying the unit tests, skipping the tests, etc. Please solve the problem "
        "fully and correctly."
    )
    if rule3_text in prompt:
        prompt = prompt.replace(rule3_text, "")
        stats["rule3_count"] += 1

    # --- Rule 4: 删除 response 中的代码块标记 (字符串替换) ---
    # 只要 response 中包含 ```python\n 或者 \n``` 任意一个，就算这条数据被 Rule 4 修改了
    original_response = response
    response = response.replace("```python\n", "")
    response = response.replace("\n```", "")
    
    if response != original_response:
        stats["rule4_count"] += 1

    # --- 更新字段 ---
    item["prompt"] = prompt
    item["response"] = response
    
    # 自动更新长度元数据，防止训练出错
    if "prompt_length" in item:
        item["prompt_length"] = len(prompt)
    if "response_length" in item:
        item["response_length"] = len(response)

    return item

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    output_path = pathlib.Path(args.output_file)

    print(f"Processing: {input_path}")
    
    stats = {
        "total_lines": 0,
        "rule1_count": 0, # Expert programmer ... Question
        "rule2_count": 0, # Format ... step by step
        "rule3_count": 0, # Now solve the problem ...
        "rule4_count": 0  # Remove ```python markers
    }

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Cleaning Data"):
            line = line.strip()
            if not line: continue
            
            stats["total_lines"] += 1
            
            try:
                item = json.loads(line)
                processed_item = process_item(item, stats)
                f_out.write(json.dumps(processed_item, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                continue

    # --- 输出统计报告 ---
    print("\n" + "=" * 40)
    print("DATA CLEANING REPORT")
    print("=" * 40)
    print(f"Total Lines Processed: {stats['total_lines']}")
    print("-" * 40)
    print(f"Rule 1 Mod (My Prompt Header):   {stats['rule1_count']:>6} lines ({(stats['rule1_count']/stats['total_lines']):.1%})")
    print(f"Rule 2 Mod (My Prompt Footer):   {stats['rule2_count']:>6} lines ({(stats['rule2_count']/stats['total_lines']):.1%})")
    print(f"Rule 3 Mod (Persona Prompt Footer):   {stats['rule3_count']:>6} lines ({(stats['rule3_count']/stats['total_lines']):.1%})")
    print(f"Rule 4 Mod (My Resp CodeMark): {stats['rule4_count']:>6} lines ({(stats['rule4_count']/stats['total_lines']):.1%})")
    print("=" * 40)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()