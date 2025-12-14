import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Clean prompt and filter long trajectories.")
    parser.add_argument("input_file", type=str, help="Path to input .jsonl file")
    parser.add_argument("--output_file", type=str, default=None, help="Path to output .jsonl file")
    parser.add_argument("--max_len", type=int, default=10000, help="Max char length for prompt or response")
    args = parser.parse_args()

    # 确定输出文件路径
    if args.output_file:
        output_path = args.output_file
    else:
        base, ext = os.path.splitext(args.input_file)
        output_path = f"{base}_cleaned{ext}"

    discarded_indices = []
    processed_count = 0
    
    # 定义分隔符
    START_MARKER = "\nuser\n"
    END_MARKER = "\n\nassistant"

    print(f"Processing {args.input_file}...")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {line_num}")
                    continue

                # === 1. 处理 Prompt 截断 ===
                prompt = data.get("prompt", "")
                
                # 如果有结束标记，取前半部分
                if END_MARKER in prompt:
                    prompt = prompt.split(END_MARKER)[0]
                
                # 如果有开始标记，取后半部分
                if START_MARKER in prompt:
                    parts = prompt.split(START_MARKER, 1)
                    if len(parts) > 1:
                        prompt = parts[1]
                
                # 更新处理后的 prompt
                data["prompt"] = prompt
                
                # === 2. 长度检查 ===
                response = data.get("response", "")
                if len(prompt) > args.max_len or len(response) > args.max_len:
                    # 获取 index，如果获取不到则用行号代替
                    idx = data.get("extra_info", {}).get("index", f"Line_{line_num}")
                    discarded_indices.append(idx)
                    continue # 丢弃，不写入文件
                
                # 写入符合要求的数据
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1

        print(f"\nDone! Saved to: {output_path}")
        print(f"Kept items: {processed_count}")
        print(f"Discarded items (> {args.max_len} chars): {len(discarded_indices)}")
        
        if discarded_indices:
            print("\nIndices of discarded items:")
            print(discarded_indices)

    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()