"""
Script: 7_visualize_activations_v5.py

Description:
    Visualizes high-dimensional neural network activations using t-SNE and UMAP.
    Features Advanced Decoupled Categorization:
    - 15 Distinct Cool Colors for Safe categories.
    - 15 Distinct Warm Colors for Unsafe categories.
    - Every unique (Source + Safety + Type) gets a distinct color + shape combination!

Usage:
python3 7_visualize_activations.py --source wild pfc --output_dir /home/Lichen/cot-monitor-modified/probe_main-table_debug/vis_activations/qwen7b_pfc_ln_tn_sh_mh_hh --datasets /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/ckpt61/7b_pfc_think-ins_cot_test_ln-385_ckpt61 /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/ckpt61/7b_pfc_think-ins_cot_test_tn-1357_ckpt61 /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/ckpt61/7b_pfc_think-ins_cot_test_sh-186_ckpt61 /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/ckpt61/7b_pfc_think-ins_cot_test_mh-75_ckpt61 /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/ckpt61/7b_pfc_think-ins_cot_test_hh-36_ckpt61 --types leetcode taco skip modify hardcode

python3 7_visualize_activations.py --source wild pfc --output_dir /home/Lichen/cot-monitor-modified/probe_main-table_debug/vis_activations/exit-scrt_wild_ln_tn_uh_eh_eh-scr --datasets /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/exit_scrt/wild_dup4_test_ln480_exit_scrt /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/exit_scrt/wild_dup4_test_tn905_exit_scrt /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/exit_scrt/wild_dup4_test_uh141_exit_scrt /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/exit_scrt/wild_dup4_test_eh100_exit_scrt /home/Lichen/cot-monitor-modified/probe_main-table_debug/processed/exit_scrt/wild_dup4_test_eh21-scrt_exit_scrt --types leetcode taco unittest exit exit-scr
"""

import os
import sys
import json
import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger

# ML & Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import umap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Visual Configuration ---

# 1. Expanded Color Pools (Cool for Safe, Warm for Unsafe)
# 给 Safe 准备的 15 种高辨识度冷色/中性色
SAFE_COLORS = [
    "#0072B2", # 深蓝色 (Okabe-Ito)
    "#56B4E9", # 天蓝色 (Okabe-Ito)
    "#009E73", # 青绿色 (Okabe-Ito)
    "#2CA02C", # 绿色 (Tab10)
    "#1F77B4", # 蓝色 (Tab10)
    "#9467BD", # 紫色 (Tab10)
    "#008080", # 蓝绿色 (Teal)
    "#000080", # 海军蓝 (Navy)
    "#3F51B5", # 靛蓝色 (Indigo)
    "#00CED1", # 暗青色 (Dark Turquoise)
    "#8A2BE2", # 蓝紫色 (Blue Violet)
    "#5F9EA0", # 军校蓝 (Cadet Blue)
    "#2E8B57", # 海洋绿 (Sea Green)
    "#4682B4", # 钢蓝色 (Steel Blue)
    "#4169E1", # 皇家蓝 (Royal Blue)
]

# 给 Unsafe 准备的 15 种高辨识度暖色
UNSAFE_COLORS = [
    "#E69F00", # 亮橙色 (Okabe-Ito)
    "#D55E00", # 朱红色 (Okabe-Ito)
    "#CC79A7", # 紫红色/品红 (Okabe-Ito)
    "#F0E442", # 明黄色 (Okabe-Ito)
    "#FF7F0E", # 橙色 (Tab10)
    "#D62728", # 红色 (Tab10)
    "#FA8072", # 珊瑚橘 (Salmon)
    "#FFD700", # 金黄色 (Gold)
    "#FFB347", # 浅橙色 (Pastel Orange)
    "#FF00FF", # 纯洋红 (Magenta)
    "#DC143C", # 猩红色 (Crimson)
    "#FF4500", # 橙红色 (Orange Red)
    "#C71585", # 中紫罗兰红 (Medium Violet Red)
    "#FF6347", # 番茄红 (Tomato)
    "#FF1493", # 深粉色 (Deep Pink)
]

# 2. Marker Pools
# Matplotlib/Seaborn Markers
SNS_SAFE_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '8', '<', '>']
SNS_UNSAFE_MARKERS = ['X', 'P', 'd', '*', '+', 'x', '1', '2', '3', '4']

# Plotly Markers
PLY_SAFE_MARKERS = ['circle', 'square', 'triangle-up', 'diamond', 'triangle-down', 'pentagon', 'hexagon', 'octagon', 'triangle-left', 'triangle-right']
PLY_UNSAFE_MARKERS = ['x', 'star', 'cross', 'diamond-x', 'hexagram', 'square-x', 'circle-x', 'star-triangle-up', 'star-triangle-down', 'triangle-up-x']

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Activations with ultra-distinct colors.")
    parser.add_argument("--datasets", nargs='+', required=True, 
                        help="List of dataset paths. e.g., ../data1 ../data2")
    parser.add_argument("--types", nargs='+', required=True, 
                        help="List of types corresponding 1:1 to --datasets. e.g., exit normal")
    parser.add_argument("--sources", nargs='+', required=True, 
                        help="Keywords to extract source from paths. e.g., wild pfc")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the visualization plots")
    parser.add_argument("--method", type=str, choices=['tsne', 'umap', 'all'], default='umap',
                        help="Dimensionality reduction method. Options: 'tsne', 'umap', 'all'. Default: 'all'.")
    parser.add_argument("--html", action="store_true", 
                        help="If set, generates interactive HTML plots")
    parser.add_argument("--no_pca", action="store_true", 
                        help="If set, skips PCA(50) preprocessing")
    parser.add_argument("--max_samples", type=int, default=3000, 
                        help="Max points to plot (prevents overplotting). Default: 3000.")
    
    args = parser.parse_args()
    
    if len(args.datasets) != len(args.types):
        logger.error(f"Length mismatch: --datasets ({len(args.datasets)}) and --types ({len(args.types)}) must be equal.")
        sys.exit(1)
        
    return args

def load_dataset_activations(folder_path):
    folder_path = pathlib.Path(folder_path)
    activations_path = folder_path / "activations"
    labels_path = folder_path / "labels"
    
    if not activations_path.exists() or not labels_path.exists():
        logger.warning(f"Skipping {folder_path}: missing 'activations' or 'labels' subdirs.")
        return [], []
        
    xs, ys_safety = [], []
    pt_files = list(activations_path.glob("*.pt"))
    
    for activation_file in pt_files:
        parts = activation_file.stem.split("_")
        if len(parts) != 2: continue
        prompt_id, cot_id = parts
        label_file = labels_path / prompt_id / f"{prompt_id}_{cot_id}_labeled.json"
        
        if not label_file.exists(): continue
            
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                score = label_data["safety_label"]["score"]
                safety_str = "Safe" if score != 1 else "Unsafe"
            
            act = torch.load(activation_file, map_location='cpu')
            if act.dim() > 1: act = act.view(-1)
            
            xs.append(act.numpy())
            ys_safety.append(safety_str)
            
        except Exception:
            continue
            
    return xs, ys_safety

def assign_visual_mappings(df):
    unique_style_cats = sorted(df['Style_Cat'].unique())
    unique_categories = sorted(df['Category'].unique())
    
    # 1. Map Shapes to Style Categories (Safe_xxx, Unsafe_yyy)
    sns_markers_dict, ply_markers_dict = {}, {}
    safe_shape_idx, unsafe_shape_idx = 0, 0
    
    for style_cat in unique_style_cats:
        if style_cat.startswith("Safe"):
            sns_markers_dict[style_cat] = SNS_SAFE_MARKERS[safe_shape_idx % len(SNS_SAFE_MARKERS)]
            ply_markers_dict[style_cat] = PLY_SAFE_MARKERS[safe_shape_idx % len(PLY_SAFE_MARKERS)]
            safe_shape_idx += 1
        else:
            sns_markers_dict[style_cat] = SNS_UNSAFE_MARKERS[unsafe_shape_idx % len(SNS_UNSAFE_MARKERS)]
            ply_markers_dict[style_cat] = PLY_UNSAFE_MARKERS[unsafe_shape_idx % len(PLY_UNSAFE_MARKERS)]
            unsafe_shape_idx += 1

    # 2. Assign unique colors to every specific Category
    sns_palette, ply_palette = {}, {}
    final_sns_markers, final_ply_markers = {}, {}
    
    safe_color_idx, unsafe_color_idx = 0, 0
    for cat in unique_categories:
        parts = cat.split('_', 2)
        safety = parts[1]
        style_cat = f"{safety}_{parts[2]}"
        
        # 只要是 Safe 就从冷色池取，只要是 Unsafe 就从暖色池取，保证每个 Category 颜色都不同
        if safety == "Safe":
            color = SAFE_COLORS[safe_color_idx % len(SAFE_COLORS)]
            safe_color_idx += 1
        else:
            color = UNSAFE_COLORS[unsafe_color_idx % len(UNSAFE_COLORS)]
            unsafe_color_idx += 1
            
        sns_palette[cat] = color
        ply_palette[cat] = color
        
        final_sns_markers[cat] = sns_markers_dict[style_cat]
        final_ply_markers[cat] = ply_markers_dict[style_cat]
        
    return sns_palette, ply_palette, final_sns_markers, final_ply_markers

def generate_static_plot(df, title, out_path, sns_palette, sns_markers):
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    df = df.sort_values(by="Category")
    
    sns.scatterplot(
        data=df, x='Dim_1', y='Dim_2', hue='Category', style='Category',
        palette=sns_palette, markers=sns_markers, alpha=0.8, s=80, edgecolor='w', linewidth=0.5
    )
    
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="[Source]_[Safety]_[Type]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_interactive_plot(df, title, out_path, ply_palette, ply_markers):
    df = df.sort_values(by="Category")
    fig = px.scatter(
        df, x='Dim_1', y='Dim_2', color='Category', symbol='Category',
        color_discrete_map=ply_palette, symbol_map=ply_markers,
        hover_data=['Source', 'Safety', 'Type'], title=title, opacity=0.8
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.write_html(out_path)

def main():
    args = parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Data Collection & Parsing...")
    all_x, metadata = [], []
    
    for path_str, type_str in tqdm(zip(args.datasets, args.types), total=len(args.datasets), desc="Loading Datasets"):
        path_obj = pathlib.Path(path_str)
        path_lower = path_obj.name.lower()
        
        matched_source = None
        for src in args.sources:
            if src.lower() in path_lower:
                matched_source = src
                break
                
        if not matched_source:
            logger.warning(f"Path '{path_obj.name}' contains NO matching keywords from {args.sources}. Skipping.")
            continue
            
        xs, ys_safety = load_dataset_activations(path_obj)
        
        if xs:
            all_x.extend(xs)
            for safety in ys_safety:
                metadata.append({
                    'Source': matched_source, 'Safety': safety, 'Type': type_str,
                    'Style_Cat': f"{safety}_{type_str}", 'Category': f"{matched_source}_{safety}_{type_str}"
                })
            
    if not all_x:
        logger.error("No valid data loaded after filtering. Exiting.")
        return
        
    X = np.array(all_x)
    df_base = pd.DataFrame(metadata)
    logger.info(f"Successfully loaded {len(X)} activations (Dim: {X.shape[1]}).")
    
    if len(X) > args.max_samples:
        logger.info(f"Data exceeds max_samples ({args.max_samples}). Stratified Subsampling...")
        strat_key = df_base['Category'].values
        try:
            X, _, df_base, _ = train_test_split(X, df_base, train_size=args.max_samples, stratify=strat_key, random_state=42)
            df_base.reset_index(drop=True, inplace=True)
            logger.info(f"Subsampled down to {len(X)} points successfully.")
        except ValueError:
            logger.warning("Stratified split failed (some classes too rare). Using random split.")
            idx = np.random.choice(len(X), args.max_samples, replace=False)
            X = X[idx]
            df_base = df_base.iloc[idx].reset_index(drop=True)

    if not args.no_pca and X.shape[1] > 50:
        logger.info("Applying PCA to reduce to 50 dimensions...")
        pca = PCA(n_components=50, random_state=42)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
        logger.info("Skipping PCA preprocessing.")

    sns_palette, ply_palette, sns_markers, ply_markers = assign_visual_mappings(df_base)

    if args.method in ['tsne', 'all']:
        logger.info("Running t-SNE (this might take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        X_tsne = tsne.fit_transform(X_reduced)
        df_tsne = df_base.copy()
        df_tsne['Dim_1'], df_tsne['Dim_2'] = X_tsne[:, 0], X_tsne[:, 1]
        
        generate_static_plot(df_tsne, "t-SNE Activations Visualization", out_dir / "tsne.png", sns_palette, sns_markers)
        if args.html:
            generate_interactive_plot(df_tsne, "t-SNE Interactive Plot", out_dir / "tsne.html", ply_palette, ply_markers)
        logger.success("t-SNE plots saved.")

    if args.method in ['umap', 'all']:
        logger.info("Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_reduced)
        df_umap = df_base.copy()
        df_umap['Dim_1'], df_umap['Dim_2'] = X_umap[:, 0], X_umap[:, 1]
        
        generate_static_plot(df_umap, "UMAP Activations Visualization", out_dir / "umap.png", sns_palette, sns_markers)
        if args.html:
            generate_interactive_plot(df_umap, "UMAP Interactive Plot", out_dir / "umap.html", ply_palette, ply_markers)
        logger.success("UMAP plots saved.")
    
    logger.info(f"All done! Files saved in: {out_dir}")

if __name__ == "__main__":
    main()