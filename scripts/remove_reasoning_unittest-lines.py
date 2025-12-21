import json
import argparse
import os
from tqdm import tqdm

def clean_response_text(text):
    """
    清洗逻辑：
    1. 识别代码块 (``` ... ```)。
    2. 如果在代码块内：保持原样，不做删除（保护代码完整性）。
    3. 如果在代码块外 (Reasoning/Text)：删除任何包含 "test case" (不区分大小写) 的行。
    """
    if not text:
        return "", False

    lines = text.split('\n')
    new_lines = []
    
    in_code_block = False
    modified = False
    
    for line in lines:
        stripped_line = line.strip()
        
        # 1. 检测代码块边界
        # 遇到 ``` 开头的行，切换状态
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        
        # 2. 处理代码块内部
        if in_code_block:
            # 在代码块里，我们不做任何删除，确保代码语法不被破坏
            new_lines.append(line)
        
        # 3. 处理代码块外部 (Reasoning 部分)
        else:
            # 检查是否包含 "test case" (不区分大小写)
            if "test case" in line.lower():
                modified = True
                # 跳过该行（即删除）
                continue
            else:
                new_lines.append(line)
    
    return "\n".join(new_lines), modified

def process_file(input_file):
    # 构造输出文件名：在原文件名前加 clean_reasoning_
    input_path = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    output_filename = f"clean_reasoning_{input_filename}"
    output_file = os.path.join(input_path, output_filename)
    
    print(f"Processing: {input_file}")
    print(f"Output to:  {output_file}")
    
    processed_count = 0
    modified_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # 计算总行数用于进度条
        # 先读取所有行可能会爆内存如果文件巨大，这里假设文件大小适中
        # 如果文件非常大，可以直接 iterate fin
        lines = fin.readlines()
        
        for line in tqdm(lines, desc="Cleaning"):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                original_response = data.get("response", "")
                
                # 执行清洗
                cleaned_response, is_modified = clean_response_text(original_response)
                
                if is_modified:
                    modified_count += 1
                
                # 更新数据
                data["response"] = cleaned_response
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                processed_count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
                continue

    print("-" * 30)
    print(f"Done! Processed {processed_count} lines.")
    print(f"Modified items (found 'test case' in reasoning): {modified_count}")
    print(f"Output saved to: {output_file}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove lines containing 'test case' from Reasoning parts only.")
    parser.add_argument("input_file", help="Path to input .jsonl file")
    
    args = parser.parse_args()
    process_file(args.input_file)