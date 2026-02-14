import argparse
import os
import sys
from itertools import islice

def main():
    parser = argparse.ArgumentParser(description="截取 JSONL 文件的前 N 行")
    
    # 输入文件 (必填)
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help='输入 .jsonl 文件路径'
    )
    
    # 截取行数 (必填)
    parser.add_argument(
        '-n', '--num', 
        type=int, 
        required=True, 
        help='需要截取的行数 (例如: 100)'
    )
    
    # 输出文件 (选填，不填则自动生成)
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        help='输出文件路径 (留空则默认保存为: 原文件名_head_N.jsonl)'
    )

    args = parser.parse_args()

    input_path = args.input
    num_lines = args.num
    
    # 1. 检查输入是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件: {input_path}")
        sys.exit(1)

    # 2. 确定输出路径
    if args.output:
        output_path = args.output
    else:
        # 自动生成文件名逻辑： data.jsonl -> data_head_100.jsonl
        root, ext = os.path.splitext(input_path)
        output_path = f"{root}_head_{num_lines}{ext}"

    # 3. 开始处理 (使用流式处理，不占用内存)
    print(f"🔄 正在从 {input_path} 截取前 {num_lines} 行...")
    
    count = 0
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            # islice 可以高效地只读取前 N 行，不需要读完整个文件
            for line in islice(fin, num_lines):
                fout.write(line)
                count += 1
                
        print(f"✅ 完成！已保存 {count} 行数据到: {output_path}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()