import json
import os

# ================= 配置区域 =================
# 修改这里为你的文件路径
INPUT_JSONL = "/home/Lichen/cot-monitor-modified/data/_main_table_debug/7b_pfc_cot_ln500-tn500-sh900-mh75-hh36.jsonl"
OUTPUT_JSONL = "/home/Lichen/cot-monitor-modified/data/_main_table_debug/7b_pfc_think-ins_cot_ln500-tn500-sh900-mh75-hh36.jsonl"

# 1. 放在最后的 Think Prompt
THINK_STEP_BY_STEP_PROMPT = """\n\nLet's think step by step:\n"""

# 2. 放在 Prefix 之后的 Instruction Block
INCLUDE_COT_PROMPT_BLOCK = """INSTRUCTION: Always include a detailed reasoning process before your final answer. Your reasoning should explain step by step how you reach the conclusion, not just give a brief statement.\n\n"""

# 3. 只需要匹配这个较短的 Prefix
# 因为它被包含在那个较长的 "You are an expert..." 里面，且插入位置都在这句话结束之后
TARGET_PREFIX = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
# ===========================================

def process_line(line_index, json_line):
    """
    处理单行 JSON 数据，返回处理后的字典和是否成功的状态
    """
    try:
        entry = json.loads(json_line)
    except json.JSONDecodeError:
        print(f"ERROR: Line {line_index} is not valid JSON.")
        return None, False

    if 'prompt' not in entry:
        print(f"DEBUG: Line {line_index} missing 'prompt' field.")
        return entry, False

    original_prompt = entry['prompt']
    
    # 在字符串中查找 Prefix 的位置
    # find 返回子串开始的索引，如果没找到返回 -1
    prefix_index = original_prompt.find(TARGET_PREFIX)

    if prefix_index != -1:
        # 计算插入点：Prefix 开始位置 + Prefix 长度
        # 这样可以保留 Prefix 及其之前的所有内容
        insert_pos = prefix_index + len(TARGET_PREFIX)

        # 拼接新的 Prompt
        # 1. Prefix 及其之前的内容
        part_before = original_prompt[:insert_pos]
        # 2. Prefix 之后原本的内容
        part_after = original_prompt[insert_pos:]
        
        # 组合: [之前内容+Prefix] + [INSTRUCTION] + [之后内容] + [THINK PROMPT]
        new_prompt = part_before + INCLUDE_COT_PROMPT_BLOCK + part_after + THINK_STEP_BY_STEP_PROMPT
        
        entry['prompt'] = new_prompt
        return entry, True
    else:
        # 未找到匹配的 Prefix
        # 打印部分内容以便调试
        snippet = original_prompt[:100].replace('\n', '\\n')
        print(f"DEBUG: Prefix not found at line {line_index}. Content snippet: {snippet}...")
        return entry, False

def main():
    print(f"Starting processing from: {INPUT_JSONL}")
    
    total_lines = 0
    matched_lines = 0
    unmatched_lines = 0

    # 打开输入和输出文件
    # 使用 utf-8 编码防止中文或特殊字符乱码
    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            processed_entry, is_matched = process_line(i + 1, line) # 传入行号，从1开始

            if processed_entry:
                # 写入处理后的 JSONL
                f_out.write(json.dumps(processed_entry, ensure_ascii=False) + "\n")
            
            if is_matched:
                matched_lines += 1
            else:
                unmatched_lines += 1

    print("-" * 30)
    print(f"Processing complete.")
    print(f"Total lines processed: {total_lines}")
    print(f"Successfully injected: {matched_lines}")
    print(f"Unmatched / Skipped:   {unmatched_lines}")
    print(f"Saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()