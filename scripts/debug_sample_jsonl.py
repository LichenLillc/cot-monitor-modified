import json
import random
import argparse
import os

def main():
    # 1. 配置命令行参数
    parser = argparse.ArgumentParser(description="从 .jsonl 文件中随机抽取指定行数")
    parser.add_argument("input", help="输入的 .jsonl 文件路径")
    parser.add_argument("-n", "--num", type=int, default = 20, help="需要抽取的行数")
    parser.add_argument("-o", "--output", help="输出文件路径 (默认为输入目录下 temp.json)")

    args = parser.parse_args()

    # 2. 处理默认输出路径
    if not args.output:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(input_dir, "temp.json")

    try:
        # 3. 读取所有行
        with open(args.input, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 检查抽取数量是否超过总行数
        total_lines = len(lines)
        if args.num > total_lines:
            print(f"⚠️ 警告：请求抽取 {args.num} 行，但文件只有 {total_lines} 行。将返回全部数据。")
            sampled_lines = lines
        else:
            # 随机采样
            sampled_lines = random.sample(lines, args.num)

        # 4. 转换为 JSON 对象列表并保存
        output_data = [json.loads(line) for line in sampled_lines]
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"✅ 成功抽取 {len(output_data)} 行数据到: {args.output}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()