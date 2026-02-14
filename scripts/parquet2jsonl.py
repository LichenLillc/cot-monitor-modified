import argparse
import pandas as pd
import os
import sys

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="将 Parquet 文件转换为 JSONL 格式")
    
    # 添加输入参数 (required=True 表示必填)
    parser.add_argument(
        '-i', '--input', 
        type=str, 
        required=True, 
        help='输入 Parquet 文件的路径 (例如: data/train.parquet)'
    )
    
    # 添加输出参数 (required=True 表示必填)
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        required=True, 
        help='输出 JSONL 文件的路径 (例如: output.jsonl)'
    )

    # 解析参数
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # 2. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误: 找不到输入文件: {input_path}")
        sys.exit(1)

    # 3. 开始转换
    print(f"🔄 正在读取: {input_path}")
    try:
        # 读取 Parquet
        df = pd.read_parquet(input_path)
        print(f"✅ 读取成功，包含 {len(df)} 行数据。")
        
        print(f"💾 正在写入: {output_path}")
        # 转换为 JSONL
        # force_ascii=False 确保中文不乱码
        # orient='records', lines=True 是标准 JSONL 格式
        df.to_json(output_path, orient='records', lines=True, force_ascii=False)
        
        print(f"🎉 转换完成！")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()