import json
import argparse
import os
import re

def remove_unittest_code(text):
    """
    移除文本中与 unittest 相关的 import、继承自 TestCase 的类以及特定的注释行。
    返回: (cleaned_text, found_test_class_flag)
    """
    if not text:
        return "", False
    
    lines = text.split('\n')
    new_lines = []
    
    in_test_class = False
    test_class_indent_level = 0
    found_test_class = False # 标记是否发现了符合条件的测试类
    
    # 编译正则：匹配 class X(TestCase): 或 class X(unittest.TestCase):
    # \s+ 匹配空格, \w+ 匹配类名, (?:...)? 是非捕获组匹配可选前缀
    class_pattern = re.compile(r"class\s+\w+\s*\((?:unittest\.)?TestCase\):")

    # 获取每一行的缩进空格数
    def get_indent(s):
        return len(s) - len(s.lstrip())

    for line in lines:
        stripped_line = line.strip()
        
        # 1. 删除仅包含 "# Test cases" 的行
        if stripped_line == "# Test cases":
            continue

        # 2. 删除 import unittest 相关行
        if "unittest" in line and "import " in line:
            continue

        # 3. 检测测试类的开始 (使用正则通用匹配)
        # 只要行里包含符合 "class Name(TestCase):" 模式的内容
        if class_pattern.search(line):
            in_test_class = True
            test_class_indent_level = get_indent(line)
            found_test_class = True # 标记发现了测试类
            continue # 删除类定义这一行

        # 4. 如果当前处于 Test Class 内部
        if in_test_class:
            # 空行在类中间通常也删掉
            if not stripped_line:
                continue
            
            current_indent = get_indent(line)
            
            # 核心逻辑：只要缩进比 'class' 那一行更深，就说明还在类里面
            if current_indent > test_class_indent_level:
                continue
            else:
                # 缩进回退了，说明类结束了
                in_test_class = False
                # 这一行已经不属于类了，需要流转到下方的 append 逻辑保留下来
        
        # 5. 如果不在需要删除的类中，则保留该行
        if not in_test_class:
            new_lines.append(line)

    return '\n'.join(new_lines), found_test_class

def process_file(input_file):
    # 确定输出文件名
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_deleted_unittest{ext}"
    
    print(f"Processing {input_file} ...")
    
    processed_count = 0
    cleaned_class_count = 0
    untouched_indices = [] # 用于记录没有发现任何测试类的项

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    original_response = data.get("response", "")
                    
                    # 执行清理
                    cleaned_response, class_found = remove_unittest_code(original_response)
                    
                    # 统计逻辑
                    if class_found:
                        cleaned_class_count += 1
                    else:
                        # 如果没有发现任何测试类，记录 index
                        idx = data.get("extra_info", {}).get("index", f"Line_{line_num}")
                        untouched_indices.append(idx)
                    
                    # 更新数据
                    data["response"] = cleaned_response
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                    
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {line_num}")
                    continue

        print("-" * 30)
        print(f"Done! Processed {processed_count} lines.")
        print(f"Items where a Test Class was found and removed: {cleaned_class_count}")
        print(f"Items where NO Test Class was found (Untouched): {len(untouched_indices)}")
        print(f"Output saved to: {output_file}")
        
        # === 最后打印完全没有被“删除类”逻辑处理过的项的 Index ===
        if untouched_indices:
            print("\n[Indices of items where NO Test Class was detected]:")
            print(untouched_indices)
        else:
            print("\n[All items contained a Test Class and were processed.]")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean unittest code from jsonl response field (Robust Regex Version).")
    parser.add_argument("input_file", help="Path to input .jsonl file")
    
    args = parser.parse_args()
    process_file(args.input_file)