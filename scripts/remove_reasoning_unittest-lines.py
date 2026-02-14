import json
import argparse
import os
from tqdm import tqdm

def clean_response_text(text):
    """
    清洗逻辑：
    1. 识别代码块 (``` ... ```)。
    2. 如果在代码块内：保持原样。
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
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            new_lines.append(line)
            continue
        
        # 2. 处理代码块内部
        if in_code_block:
            new_lines.append(line)
        
        # 3. 处理代码块外部 (Reasoning 部分)
        else:
            if "test case" in line.lower():
                modified = True
                continue # 删除该行
            else:
                new_lines.append(line)
    
    return "\n".join(new_lines), modified

def process_file(input_file, target_hacking_type=None):
    input_path = os.path.dirname(input_file)
    input_filename = os.path.basename(input_file)
    output_filename = f"clean_reasoning_{input_filename}"
    output_file = os.path.join(input_path, output_filename)
    
    print(f"Processing: {input_file}")
    if target_hacking_type:
        print(f"🎯 Filter Mode: Only cleaning items where hacking_type='{target_hacking_type}'")
    else:
        print(f"🌍 Global Mode: Cleaning ALL items")
    print(f"Output to:  {output_file}")
    
    processed_count = 0
    cleaned_count = 0 # 实际被修改的数量
    ignored_count = 0 # 因为类型不匹配而被跳过（保持原样）的数量
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        
        for line in tqdm(lines, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 获取当前项的 hacking_type
                # 注意：有些数据可能没有这个字段，默认设为 "unknown" 或其他字符串以防报错
                item_hacking_type = str(data.get("hacking_type", "unknown"))
                
                # === 核心判断逻辑 ===
                # 如果没有指定 target (None)，则处理所有项
                # 如果指定了 target，则必须匹配才处理
                should_process = (target_hacking_type is None) or (item_hacking_type == str(target_hacking_type))
                
                if should_process:
                    original_response = data.get("response", "")
                    cleaned_response, is_modified = clean_response_text(original_response)
                    
                    if is_modified:
                        cleaned_count += 1
                    
                    data["response"] = cleaned_response
                    # 写入处理后的数据
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                else:
                    # 类型不匹配，直接原样写入，不做任何修改
                    ignored_count += 1
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                processed_count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
                continue

    print("-" * 30)
    print(f"✅ Done! Total processed: {processed_count}")
    print(f"🧹 Cleaned (Modified): {cleaned_count}")
    if target_hacking_type:
        print(f"⏭️  Skipped (Type mismatch): {ignored_count}")
    print(f"💾 Output saved to: {output_file}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove 'test case' lines from Reasoning (with optional filtering).")
    parser.add_argument("input_file", help="Path to input .jsonl file")
    
    # 新增的可选参数
    parser.add_argument("--hacking_type", type=str, default=None, 
                        help="Optional: Only clean items with this specific hacking_type. Others will be kept unchanged.")
    
    args = parser.parse_args()
    process_file(args.input_file, args.hacking_type)