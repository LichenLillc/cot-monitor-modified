import argparse
import json
import os
import re
import random

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inject synthetic TestSolution classes into trajectories.")
    parser.add_argument("input_file", type=str, help="Path to the input .jsonl file.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to the output .jsonl file. Defaults to input_name_synthetic_test_class.jsonl.")
    return parser.parse_args()

def extract_tests_from_code(test_code_str):
    """
    解析 test_code 字符串，提取 (arguments_value, expected_value) 元组列表。
    去除参数名（如 'nums='），只保留值。
    """
    tests = []
    lines = test_code_str.split('\n')
    pattern = re.compile(r'assert\s+candidate\((.*)\)\s*==\s*(.*)')
    
    for line in lines:
        line = line.strip()
        match = pattern.search(line)
        if match:
            args_raw = match.group(1)
            expected_val = match.group(2)
            
            # 去除参数名，例如 "nums = [1,2], k = 3" -> "[1,2], 3"
            args_clean = re.sub(r'\b[a-zA-Z_]\w*\s*=\s*', '', args_raw)
            tests.append((args_clean.strip(), expected_val.strip()))
    return tests

def generate_test_class(method_name, test_cases):
    """
    根据给定的方法名和测试用例数据，生成具有随机风格的 TestSolution 字符串。
    """
    lines = []
    
    # === 随机维度 0: 注释头 (Class前) ===
    # 随机决定是否在 import 前添加 "# Test cases"
    if random.choice([True, False]):
        lines.append("# Test cases")
    
    # === 随机维度 1: Import 风格 ===
    import_style = random.choice(['from_unittest', 'import_unittest'])
    
    if import_style == 'from_unittest':
        lines.append("from unittest import TestCase")
        lines.append("") 
        lines.append("class TestSolution(TestCase):")
    else:
        lines.append("import unittest")
        lines.append("") 
        lines.append("class TestSolution(unittest.TestCase):")

    # === 随机维度 2: 实例化风格 & 变量命名 ===
    instantiation_style = random.choice(['setup', 'local', 'anonymous'])
    var_name = random.choice(['solution', 'sol', 'selfsol', 's'])
    
    # 处理 setUp
    if instantiation_style == 'setup':
        lines.append(f"    def setUp(self):")
        lines.append(f"        self.{var_name} = Solution()")
    
    # === 随机维度 3: 测试函数结构 (单函数 vs 多函数) ===
    structure_style = random.choice(['single_func', 'multi_func'])
    
    # === 新增控制: 用例注释一致性 ===
    # 决定这一次生成的类中，是否给每个测试用例加注释（避免有的加有的不加）
    use_case_comments = random.choice([True, False])

    if structure_style == 'single_func':
        # 单函数模式：使用 method_name 作为函数名
        lines.append(f"    def test_{method_name}(self):")
        
        if instantiation_style == 'local':
            lines.append(f"        {var_name} = Solution()")
        
        for idx, (args, expected) in enumerate(test_cases):
            # 构建调用部分
            if instantiation_style == 'setup':
                call_str = f"self.{var_name}.{method_name}({args})"
            elif instantiation_style == 'local':
                call_str = f"{var_name}.{method_name}({args})"
            else: # anonymous
                call_str = f"Solution().{method_name}({args})"
            
            # 统一添加注释 (如果随机到了True)
            if use_case_comments:
                lines.append(f"        # Test case {idx + 1}")
            
            lines.append(f"        self.assertEqual({call_str}, {expected})")

    else:
        # 多函数模式：保持 test_exampleX 命名
        for idx, (args, expected) in enumerate(test_cases):
            func_name = f"test_example{idx + 1}"
            lines.append(f"    def {func_name}(self):")
            
            if instantiation_style == 'local':
                lines.append(f"        {var_name} = Solution()")
            
            # 构建调用部分
            if instantiation_style == 'setup':
                call_str = f"self.{var_name}.{method_name}({args})"
            elif instantiation_style == 'local':
                call_str = f"{var_name}.{method_name}({args})"
            else: # anonymous
                call_str = f"Solution().{method_name}({args})"
            
            if use_case_comments:
                 lines.append(f"        # Test case {idx + 1}")

            lines.append(f"        self.assertEqual({call_str}, {expected})")
            lines.append("") # 函数间空行

    return "\n".join(lines)

def process_file(input_path, output_path):
    # 修改点：将计数器改为列表，用于存储 index
    skipped_no_example_indices = []
    skipped_bad_format_indices = []
    skipped_parsing_error_indices = []
    processed_count = 0

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Line {line_num}: JSON decode error, skipping.")
                    continue

                # 获取当前项的 index，如果不存在则标记为 Line_行号
                current_index = data.get("extra_info", {}).get("index", f"Line_{line_num}")

                # 1. 检查 Prompt 中的 "example" 数量
                prompt_text = data.get("prompt", "")
                example_count = prompt_text.lower().count("example")
                
                if example_count == 0:
                    skipped_no_example_indices.append(current_index)
                    continue

                # 2. 检查 Response 格式
                response_text = data.get("response", "")
                last_fence_idx = response_text.rfind("```")
                python_start_idx = response_text.rfind("```python", 0, last_fence_idx)

                if last_fence_idx == -1 or python_start_idx == -1:
                    skipped_bad_format_indices.append(current_index)
                    continue

                # 3. 获取测试用例源数据
                try:
                    ground_truth = data.get("reward_model", {}).get("ground_truth", {})
                    entry_point = ground_truth.get("entry_point", "")
                    test_code = ground_truth.get("test_code", "")
                    
                    # 提取方法名
                    if "Solution()." in entry_point:
                        method_name = entry_point.replace("Solution().", "").strip()
                    else:
                        method_name = entry_point.strip()

                    if not method_name or not test_code:
                        skipped_parsing_error_indices.append(current_index)
                        continue

                    all_test_cases = extract_tests_from_code(test_code)
                    
                    if not all_test_cases:
                        skipped_parsing_error_indices.append(current_index)
                        continue

                    num_tests_to_keep = min(example_count, len(all_test_cases))
                    selected_test_cases = all_test_cases[:num_tests_to_keep]

                    # 4. 生成 TestSolution 代码
                    test_class_code = generate_test_class(method_name, selected_test_cases)

                    # 5. 插入到 Response 中
                    new_response = (
                        response_text[:last_fence_idx] + 
                        "\n" + test_class_code + "\n" + 
                        response_text[last_fence_idx:]
                    )
                    
                    data["response"] = new_response
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1

                except Exception as e:
                    print(f"Error processing line {line_num} (Index {current_index}): {e}")
                    skipped_parsing_error_indices.append(current_index)
                    continue

        print("-" * 30)
        print(f"Processing Complete.")
        print(f"Successfully processed: {processed_count} items")
        print("\n--- Skip Details ---")
        
        print(f"Skipped (No 'example' in prompt): {len(skipped_no_example_indices)} items")
        if skipped_no_example_indices:
            print(f"Indices: {skipped_no_example_indices}")
            
        print(f"\nSkipped (No valid python block in response): {len(skipped_bad_format_indices)} items")
        if skipped_bad_format_indices:
            print(f"Indices: {skipped_bad_format_indices}")
            
        print(f"\nSkipped (Data parsing error/No tests): {len(skipped_parsing_error_indices)} items")
        if skipped_parsing_error_indices:
            print(f"Indices: {skipped_parsing_error_indices}")
            
        print(f"\nOutput saved to: {output_path}")
        print("-" * 30)

    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.output_file:
        out_path = args.output_file
    else:
        base, ext = os.path.splitext(args.input_file)
        out_path = f"{base}_synthetic_test_class{ext}"
    
    process_file(args.input_file, out_path)