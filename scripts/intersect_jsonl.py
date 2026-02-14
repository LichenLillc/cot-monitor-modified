import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Any

def get_index_value(item: dict) -> Any:
    """
    安全地获取 item['extra_info']['index'] 的值。
    如果路径不存在，返回 None。
    """
    try:
        return item.get('extra_info', {}).get('index')
    except AttributeError:
        return None

def main():
    parser = argparse.ArgumentParser(description="提取多个JSONL文件的交集（基于 extra_info.index）并排序输出。")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='输入的 .jsonl 文件路径列表')
    parser.add_argument('--suffix', type=str, default='_intersect', help='输出文件名的后缀，默认为 _intersect')
    
    args = parser.parse_args()
    
    file_paths = [Path(f) for f in args.files]
    
    # 存储结构: { 文件路径: { index值: json对象 } }
    # 目的: 方便根据 index 快速查找对应文件的原始数据
    all_files_data: Dict[Path, Dict[Any, dict]] = {}
    
    # 存储结构: [ {index1, index2...}, {index2, index3...} ]
    # 目的: 用于计算集合交集
    index_sets: List[Set[Any]] = []

    print(f"正在处理 {len(file_paths)} 个文件...")

    # --- 1. 读取数据 ---
    for f_path in file_paths:
        if not f_path.exists():
            print(f"错误: 文件 {f_path} 不存在，已跳过。")
            continue
            
        current_file_map = {}
        valid_indices = set()
        
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line: continue
                    
                    try:
                        data = json.loads(line)
                        idx = get_index_value(data)
                        
                        if idx is not None:
                            # 注意：如果同一文件中存在重复index，后面的会覆盖前面的
                            current_file_map[idx] = data
                            valid_indices.add(idx)
                        else:
                            print(f"警告: 文件 {f_path.name} 第 {line_num} 行缺少 extra_info.index，已跳过。")
                            
                    except json.JSONDecodeError:
                        print(f"警告: 文件 {f_path.name} 第 {line_num} 行 JSON 格式错误，已跳过。")
        
            all_files_data[f_path] = current_file_map
            index_sets.append(valid_indices)
            print(f"已加载: {f_path.name} (包含 {len(valid_indices)} 个有效索引)")
            
        except Exception as e:
            print(f"读取文件 {f_path} 时发生严重错误: {e}")
            sys.exit(1)

    if not index_sets:
        print("没有成功加载任何有效数据。")
        sys.exit(0)

    # --- 2. 计算交集 ---
    # set.intersection(*list) 解包列表进行交集运算
    common_indices = set.intersection(*index_sets)
    
    count = len(common_indices)
    print(f"\n找到 {count} 个公共索引。")
    
    if count == 0:
        print("文件之间没有交集，未生成输出文件。")
        sys.exit(0)

    # --- 3. 排序 ---
    # 假设 index 是数字或字符串，Python 的 sorted 都能处理
    sorted_indices = sorted(list(common_indices))

    # --- 4. 写入文件 ---
    print("\n正在写入输出文件...")
    
    for f_path in file_paths:
        if f_path not in all_files_data:
            continue
            
        # 构建输出文件名：原文件名 stem + 后缀 + .jsonl
        output_name = f"{f_path.stem}{args.suffix}.jsonl"
        output_path = f_path.parent / output_name
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for idx in sorted_indices:
                    # 从该文件的数据映射中取出对应的原始记录
                    record = all_files_data[f_path][idx]
                    # ensure_ascii=False 保证中文正常显示
                    f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"已生成: {output_path}")
            
        except Exception as e:
            print(f"写入文件 {output_path} 失败: {e}")

    print("\n处理完成。")

if __name__ == "__main__":
    main()