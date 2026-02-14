"""
Script Name: process_response_structure.py

Description:
    This script processes the 'response' field in JSONL files to separate or reorder
    Chain-of-Thought (CoT) reasoning and Code Answers.

    It identifies 'Answer' blocks as text enclosed within ```python and ``` markers.

Modes (Mutually Exclusive):
    --answer:   Keeps ONLY the LAST code block. Removes everything else.
    --cot:      Keeps ONLY the text (reasoning). Removes ALL code blocks.
    --reformat: Extracts ALL code blocks and appends them to the end of the response,
                preserving the original order (CoT first, then all Answers).

Usage Example:
    python process_response_structure.py input.jsonl --reformat
    python process_response_structure.py input.jsonl --cot
    python process_response_structure.py input.jsonl --answer
"""

import argparse
import json
import os
import re

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and restructure 'response' fields in JSONL files.")
    parser.add_argument("input_file", type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional custom output path.")

    # 定义互斥参数组，确保三个参数只能出现一个
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cot", action="store_true", help="Keep only Chain-of-Thought (remove all code blocks).")
    group.add_argument("--answer", '-a', action="store_true", help="Keep only the LAST Answer (remove CoT and other code blocks).")
    group.add_argument("--reformat", action="store_true", help="Reformat: Move all Answers to the end of the response.")

    return parser.parse_args()

def process_line(data, mode, pattern):
    """
    根据模式处理单行数据的 response 字段
    """
    response = data.get("response", "")
    
    # 查找所有匹配的代码块
    # [\s\S]*? 匹配任意字符（包括换行符），非贪婪模式
    code_blocks = pattern.findall(response)

    if mode == "answer":
        # 只保留最后一段 answer
        if code_blocks:
            new_response = code_blocks[-1]
        else:
            # 如果没有找到代码块，视为空或者保留原样？
            # 根据需求"只保留最后一段answer"，如果没有answer，结果应该是空字符串
            new_response = ""
            
    elif mode == "cot":
        # 删去所有 answer
        # 使用 sub 将匹配到的内容替换为空
        new_response = pattern.sub("", response).strip()
        
    elif mode == "reformat":
        # 剪贴所有 answer 到末尾
        if code_blocks:
            # 1. 获取纯文本部分 (CoT)
            cot_part = pattern.sub("", response).strip()
            
            # 2. 拼接代码块
            answers_part = "\n\n".join(code_blocks)
            
            # 3. 组合: CoT + \n\n + Answers
            if cot_part:
                new_response = f"{cot_part}\n\n{answers_part}"
            else:
                new_response = answers_part
        else:
            # 如果没有代码块，保持原样
            new_response = response
            
    else:
        new_response = response

    data["response"] = new_response
    return data

def main():
    args = parse_arguments()

    # 确定模式后缀
    if args.cot:
        suffix = "_cot"
        mode = "cot"
    elif args.answer:
        suffix = "_answer"
        mode = "answer"
    elif args.reformat:
        suffix = "_reformat"
        mode = "reformat"
    else:
        # 理论上互斥组 required=True 不会走到这里
        return

    # 确定输出路径
    if args.output_file:
        output_path = args.output_file
    else:
        base, ext = os.path.splitext(args.input_file)
        output_path = f"{base}{suffix}{ext}"

    # 编译正则：匹配 ```python ... ```
    # 注意：需要处理多行内容
    code_block_pattern = re.compile(r"```python[\s\S]*?```")

    print(f"Processing: {args.input_file}")
    print(f"Mode: --{mode}")
    
    processed_count = 0

    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # 处理该行数据
                    processed_data = process_line(data, mode, code_block_pattern)
                    
                    outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {line_num}")
                    continue

        print(f"Done! Processed {processed_count} lines.")
        print(f"Output saved to: {output_path}")

    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()