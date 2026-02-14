import json
import argparse
import pathlib
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Split JSONL file based on data_source keywords.")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input .jsonl file"
    )
    
    parser.add_argument(
        "--data_source", 
        nargs="+", 
        default=["leetcode", "taco"], 
        help="List of keywords to filter by (case-insensitive). Default: leetcode taco"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = pathlib.Path(args.input_file)
    
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        return

    # 1. 准备关键词 (全部转小写以忽略大小写)
    keywords = [k.lower() for k in args.data_source]
    print(f"Splitting data based on keywords: {keywords}")

    # 2. 准备输出文件句柄
    # 结构: { "leetcode": file_object, "taco": file_object }
    file_handles = {}
    file_counters = {k: 0 for k in keywords}
    
    try:
        # 预先打开所有需要的输出文件
        for k in keywords:
            # 构造文件名: 原名_关键词.后缀
            # 例如: data.jsonl -> data_leetcode.jsonl
            output_name = f"{input_path.stem}_{k}{input_path.suffix}"
            output_path = input_path.parent / output_name
            
            file_handles[k] = open(output_path, 'w', encoding='utf-8')
            print(f"Created output file: {output_path}")

        # 3. 开始读取并分流
        with open(input_path, 'r', encoding='utf-8') as f_in:
            # 使用 tqdm 显示进度
            for line in tqdm(f_in, desc="Processing"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # 获取 data_source 字段，并转小写
                # 兼容 data_source 字段缺失的情况
                ds_val = item.get("data_source", "")
                if ds_val is None: 
                    ds_val = ""
                ds_content = str(ds_val).lower()
                
                # 4. 匹配逻辑
                # 一条数据如果同时包含多个关键词（极少见），会被写入多个文件
                for k in keywords:
                    if k in ds_content:
                        file_handles[k].write(line + "\n")
                        file_counters[k] += 1

    finally:
        # 5. 关闭所有输出文件
        for fh in file_handles.values():
            fh.close()

    # 6. 打印统计信息
    print("\n" + "="*40)
    print("Split Summary:")
    print("="*40)
    for k in keywords:
        print(f"  - {k}: {file_counters[k]} items saved")
    print("="*40)

if __name__ == "__main__":
    main()