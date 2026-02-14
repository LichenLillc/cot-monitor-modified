"""
Script: 6_collect_reports.py

Description:
    Collects all 'summary_report.xlsx' files from subdirectories in results_root.
    Copies them to a new '_collected_reports' folder inside results_root.
    Renames them to '{SubdirName}_summary_report.xlsx' to avoid conflicts.
"""

import argparse
import pathlib
import shutil
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    # 默认路径设为你之前用的路径，方便直接跑
    parser.add_argument("--results_root", "-rsr", type=str, default="../probe_main-table_debug/5a_results_auto", 
                        help="Root directory containing the TRAIN-xxx_TEST-xxx subfolders")
    
    args = parser.parse_args()
    
    root_path = pathlib.Path(args.results_root)
    
    if not root_path.exists():
        logger.error(f"Results root does not exist: {root_path}")
        return

    # 1. 创建专门存放汇总文件的目录 (在 results_root 内部)
    collect_dir = root_path / "_collected_reports"
    collect_dir.mkdir(exist_ok=True)
    logger.info(f"Target collection directory: {collect_dir}")

    count = 0
    
    # 2. 遍历所有子目录
    for subdir in root_path.iterdir():
        # 确保是文件夹，且不是我们刚才创建的那个收集文件夹
        if subdir.is_dir() and subdir.name != "_collected_reports":
            
            # 目标源文件
            src_file = subdir / "summary_report.xlsx"
            
            if src_file.exists():
                # 3. 构建新文件名：目录名 + 原文件名
                # 例如: TRAIN-ckpt61_TEST-ckpt61_summary_report.xlsx
                new_filename = f"{subdir.name}_summary_report.xlsx"
                dest_path = collect_dir / new_filename
                
                try:
                    # 复制文件 (copy2 保留文件元数据如时间戳)
                    shutil.copy2(src_file, dest_path)
                    logger.info(f"Collected: {new_filename}")
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to copy from {subdir.name}: {e}")
            else:
                # 如果某个目录下没有报告，可以选择忽略或打印 warning
                pass

    if count == 0:
        logger.warning("No summary_report.xlsx files found!")
    else:
        logger.success(f"Done! Successfully collected {count} Excel files to: {collect_dir}")

if __name__ == "__main__":
    main()