"""
Script: 5a_eval_probes_loop_v2.py (Parallel Version)

Description:
    Fully automated Matrix Evaluation for Probes.
    Now supports Parallel Execution for faster processing.
    Now intelligently auto-detects and evaluates both V2 (RobustMLP) and V3 (DomainAdversarialMLP).
    [NEW] Automatically collects all generated Excel reports into a single folder at the end.
    [NEW] Global Task Flattening: Maximizes CPU utilization across all matrix groups.
"""

import os
# [关键配置] 必须在导入 numpy/torch 之前设置，防止多进程 CPU 过载
# 限制每个 worker 只使用 1 个线程，依靠多进程(workers=24/48)来提升吞吐量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import re
import json
import argparse
import pathlib
import collections
import joblib
import io
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 1. MLP Definitions (Sync with 3a_probes)
# ==========================================
class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x

class CustomMLP2Layer(nn.Module):
    def __init__(self, input_size, hidden_size1=100, hidden_size2=50, dropout_rate=0.0, noise_std=0.0):
        super(CustomMLP2Layer, self).__init__()
        self.noise = GaussianNoise(std=noise_std)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.noise(x)
        x = self.dropout(self.relu1(self.fc1(x)))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class RobustMLP(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, hidden_size3=128, dropout_rate=0.4, noise_std=0.0):
        super(RobustMLP, self).__init__()
        self.noise = GaussianNoise(std=noise_std)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.ln3 = nn.LayerNorm(hidden_size3)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rate)
        
        self.head = nn.Linear(hidden_size3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.noise(x)
        x = self.drop1(self.relu1(self.ln1(self.fc1(x))))
        x = self.drop2(self.relu2(self.ln2(self.fc2(x))))
        x = self.drop3(self.relu3(self.ln3(self.fc3(x))))
        x = self.sigmoid(self.head(x))
        return x

class DomainAdversarialMLP(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, hidden_size3=128, dropout_rate=0.4, noise_std=0.0):
        super(DomainAdversarialMLP, self).__init__()
        self.noise = GaussianNoise(std=noise_std)
        
        self.shared1 = nn.Linear(input_size, hidden_size1)
        self.ln1 = nn.LayerNorm(hidden_size1)
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.shared2 = nn.Linear(hidden_size1, hidden_size2)
        self.ln2 = nn.LayerNorm(hidden_size2)
        self.drop2 = nn.Dropout(dropout_rate)

        self.task_fc = nn.Linear(hidden_size2, hidden_size3)
        self.task_ln = nn.LayerNorm(hidden_size3)
        self.task_drop = nn.Dropout(dropout_rate)
        self.task_head = nn.Linear(hidden_size3, 1)

        self.domain_fc = nn.Linear(hidden_size2, hidden_size3)
        self.domain_ln = nn.LayerNorm(hidden_size3)
        self.domain_drop = nn.Dropout(dropout_rate)
        self.domain_head = nn.Linear(hidden_size3, 1)

    def extract_features(self, x):
        x = self.noise(x)
        x = self.drop1(torch.relu(self.ln1(self.shared1(x))))
        x = self.drop2(torch.relu(self.ln2(self.shared2(x))))
        return x

    def predict_task(self, shared_feat):
        x = self.task_drop(torch.relu(self.task_ln(self.task_fc(shared_feat))))
        return torch.sigmoid(self.task_head(x))

    def predict_domain(self, shared_feat):
        x = self.domain_drop(torch.relu(self.domain_ln(self.domain_fc(shared_feat))))
        return torch.sigmoid(self.domain_head(x))

    def forward(self, x):
        feat = self.extract_features(x)
        return self.predict_task(feat), self.predict_domain(feat)

# ==========================================
# 2. Data Loading
# ==========================================
def load_activations(input_folder):
    input_folder = pathlib.Path(input_folder)
    activations_path = input_folder / "activations"
    labels_path = input_folder / "labels"
    
    if not activations_path.exists() or not labels_path.exists():
        return None, None

    xs = []
    ys = []
    
    pt_files = list(activations_path.glob("*.pt"))
    if not pt_files:
        return None, None

    for activation_file in pt_files: 
        parts = activation_file.stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        prompt_id, cot_id = parts
        
        label_file = labels_path / prompt_id / f"{prompt_id}_{cot_id}_labeled.json"
        
        if not label_file.exists():
            continue
            
        try:
            act = torch.load(activation_file, map_location='cpu')
            if act.dim() > 1:
                act = act.view(-1)
            xs.append(act.numpy())
            
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                score = label_data["safety_label"]["score"]
                ys.append(1 if score >= 0.5 else 0)
        except Exception:
            continue
            
    if len(xs) == 0:
        return None, None

    return np.array(xs), np.array(ys)

# ==========================================
# 3. Metrics Helper
# ==========================================
def calculate_metrics(y_true, y_pred, y_probs):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100
    }
    if len(np.unique(y_true)) > 1:
        metrics["f1"] = float(f1_score(y_true, y_pred)) * 100
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_probs)) * 100
        metrics["pr_auc"] = float(average_precision_score(y_true, y_probs)) * 100
    return metrics

# ==========================================
# 4. Evaluation Logic (Atomic Task)
# ==========================================
def evaluate_single_run(model_folder_path, test_dataset_paths, seed, results_dir):
    model_dir = model_folder_path / "saved_models"
    if not model_dir.exists():
        return None

    seeds_to_run = []
    if seed is not None:
        seeds_to_run = [seed]
    else:
        potential_files = list(model_dir.glob("logreg_seed*.joblib"))
        extracted_seeds = []
        for p in potential_files:
            match = re.search(r"seed(\d+)", p.name)
            if match:
                extracted_seeds.append(int(match.group(1)))
        seeds_to_run = sorted(list(set(extracted_seeds)))
    
    if not seeds_to_run:
        return None

    dataset_names = [d.name for d in test_dataset_paths]
    combined_dataset_name = f"Combined_{len(dataset_names)}_Datasets" if len(dataset_names) > 3 else "+".join(dataset_names)
    
    all_xs = []
    all_ys = []
    all_y_domains = []

    for d in test_dataset_paths:
        xs, ys = load_activations(d)
        if xs is not None and ys is not None:
            all_xs.append(xs)
            all_ys.append(ys)
            
            name_lower = d.name.lower()
            if 'taco' in name_lower or '_tn' in name_lower or '-tn' in name_lower or '_teh' in name_lower or '_tuh' in name_lower:
                d_label = 1
            else:
                d_label = 0
            all_y_domains.append(np.full(len(ys), d_label))
    
    if not all_xs:
        return None

    X_raw_global = np.concatenate(all_xs, axis=0)
    y_test_global = np.concatenate(all_ys, axis=0)
    y_domain_global = np.concatenate(all_y_domains, axis=0)
    
    agg_metrics = {
        "logreg": collections.defaultdict(list),
        "mlp": collections.defaultdict(list),
        "mlp_domain": collections.defaultdict(list) 
    }
    detailed_results = []
    
    device = torch.device("cpu")

    for s in seeds_to_run:
        try:
            global_scaler_path = model_dir / f"global_scaler_seed{s}.joblib"
            mlp_scaler_path = model_dir / f"mlp_internal_scaler_seed{s}.joblib"
            pca_path = model_dir / f"pca_seed{s}.joblib"
            logreg_path = model_dir / f"logreg_seed{s}.joblib"
            mlp_path = model_dir / f"mlp_seed{s}.pth"

            if not global_scaler_path.exists() or not mlp_scaler_path.exists() or not logreg_path.exists() or not mlp_path.exists():
                continue

            global_scaler = joblib.load(global_scaler_path)
            mlp_scaler = joblib.load(mlp_scaler_path)
            logreg = joblib.load(logreg_path)
            
            X_stage1 = global_scaler.transform(X_raw_global)
            if pca_path.exists():
                pca = joblib.load(pca_path)
                X_stage2 = pca.transform(X_stage1)
            else:
                X_stage2 = X_stage1
            X_stage3 = mlp_scaler.transform(X_stage2)

            y_probs_lr = logreg.predict_proba(X_stage2)[:, 1]
            y_pred_lr = logreg.predict(X_stage2)
            metrics_lr = calculate_metrics(y_test_global, y_pred_lr, y_probs_lr)

            state_dict = torch.load(mlp_path, map_location=device)
            input_dim = X_stage3.shape[1]
            
            is_v3 = any(k.startswith("domain_head") for k in state_dict.keys())
            
            if is_v3:
                mlp = DomainAdversarialMLP(input_size=input_dim).to(device)
            else:
                mlp = RobustMLP(input_size=input_dim).to(device)
                
            mlp.load_state_dict(state_dict)
            mlp.eval()
            
            X_mlp_tensor = torch.tensor(X_stage3, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                if is_v3:
                    feat = mlp.extract_features(X_mlp_tensor)
                    probs_mlp = mlp.predict_task(feat).squeeze().cpu().numpy()
                    probs_domain = mlp.predict_domain(feat).squeeze().cpu().numpy()
                else:
                    probs_mlp = mlp(X_mlp_tensor).squeeze().cpu().numpy()
                    probs_domain = None
                
                if probs_mlp.ndim == 0: probs_mlp = np.expand_dims(probs_mlp, 0)
                if probs_domain is not None and probs_domain.ndim == 0: probs_domain = np.expand_dims(probs_domain, 0)
                
                y_pred_mlp = (probs_mlp > 0.5).astype(int)
            
            metrics_mlp = calculate_metrics(y_test_global, y_pred_mlp, probs_mlp)
            
            if is_v3 and probs_domain is not None:
                y_pred_domain = (probs_domain > 0.5).astype(int)
                metrics_mlp_domain = calculate_metrics(y_domain_global, y_pred_domain, probs_domain)
            else:
                metrics_mlp_domain = None

            for k, v in metrics_lr.items(): agg_metrics["logreg"][k].append(v)
            for k, v in metrics_mlp.items(): agg_metrics["mlp"][k].append(v)
            if metrics_mlp_domain:
                for k, v in metrics_mlp_domain.items(): agg_metrics["mlp_domain"][k].append(v)
            
            detailed_dict = {"seed": s, "logreg": metrics_lr, "mlp": metrics_mlp}
            if metrics_mlp_domain:
                detailed_dict["mlp_domain"] = metrics_mlp_domain
                
            detailed_results.append(detailed_dict)

        except Exception as e:
            continue

    if not detailed_results:
        return None

    final_output = {
        "source_model": model_folder_path.name,
        "target_dataset": combined_dataset_name,
        "summary": {},
        "detailed_runs": detailed_results
    }

    for model_name in ["logreg", "mlp", "mlp_domain"]:
        active_metrics = list(agg_metrics[model_name].keys())
        if not active_metrics:
            continue 
            
        final_output["summary"][model_name] = {}
        for metric in active_metrics:
            values = agg_metrics[model_name][metric]
            final_output["summary"][model_name][metric] = {
                "mean": np.mean(values), 
                "std": np.std(values)
            }

    seed_suffix = f"_seed{detailed_results[0]['seed']}" if len(detailed_results) == 1 else "_ALL_SEEDS"
    out_filename = f"{model_folder_path.name}_ON_{combined_dataset_name}{seed_suffix}.json"
    out_filename = re.sub(r'[^\w\-.+]', '_', out_filename)
    out_path = results_dir / out_filename
    
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    return final_output

# ==========================================
# 5. Helper: Name Parsing & ID-TEST
# ==========================================
def parse_folder_name(folder_name, suffix_to_remove):
    name = folder_name
    if suffix_to_remove and name.endswith(f"_{suffix_to_remove}"):
        name = name[:-(len(suffix_to_remove) + 1)] 
    
    name = re.sub(r'_?test_?', '_', name, flags=re.IGNORECASE)
    l1 = "Others"

    if "7b_pfc_think-ins" in name:
        l1 = "7b-pfc"
    elif "ds-coder-ckpt280" in name:
        l1 = "ds-coder"
    elif "qwen_exit-ckpt68" in name:
        l1 = "qwen-exit"
    elif "qwen_scratch-ckpt61" in name:
        l1 = "qwen-unittest"
    elif "wild" in name:
        l1 = "qwen-exit-scratch"

    keywords = [
        "7b_pfc_think-ins_cot", "7b_pfc_think-ins", "ds-coder-ckpt280",
        "qwen_exit-ckpt68", "qwen_scratch-ckpt61", "wild_dup4"
    ]
    
    l2_candidate = name
    for k in keywords:
        l2_candidate = l2_candidate.replace(k, "")
    
    l2 = re.sub(r'_+', '_', l2_candidate).strip('_')
    if not l2: l2 = "base"
        
    return l1, l2

def get_id_test_accuracy(model_folder):
    summary_path = model_folder / "training_summary.txt"
    if not summary_path.exists():
        return {"logreg": "N/A", "mlp": "N/A", "mlp_domain": "N/A"}
    
    try:
        content = summary_path.read_text(encoding='utf-8')
        results = {}
        for model_key in ["logreg", "mlp"]:
            for line in content.splitlines():
                if model_key in line.lower() and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        results[model_key] = parts[2] 
                        break
        results["mlp_domain"] = "N/A"
        return results
    except Exception as e:
        return {"logreg": "N/A", "mlp": "N/A", "mlp_domain": "N/A"}

# ==========================================
# 6. Global Matrix Preparation & Excel Generation
# ==========================================
def prepare_matrix_group(mp_path, dp_path, results_root):
    mp_name = mp_path.name
    dp_name = dp_path.name
    
    group_out_dir = results_root / f"TRAIN-{mp_name}_TEST-{dp_name}"
    group_out_dir.mkdir(parents=True, exist_ok=True)
    
    models = [d for d in mp_path.iterdir() if d.is_dir() and (d / "saved_models").exists()]
    datasets = [d for d in dp_path.iterdir() if d.is_dir() and (d / "activations").exists() and (d / "labels").exists() and "test" in d.name.lower()]
    
    if not models or not datasets:
        return None

    report_data = {"logreg": {}, "mlp": {}, "mlp_domain": {}}
    L1_ORDER = ["qwen-unittest", "qwen-exit", "qwen-exit-scratch", "ds-coder", "7b-pfc"]
    
    model_rows = [] 
    dataset_cols = [] 

    for m in models:
        l1, l2 = parse_folder_name(m.name, mp_name)
        if l1 == "Others" or l1 not in L1_ORDER:
            logger.error(f"Unknown L1 category for model: {m.name} -> {l1}. Aborting.")
            sys.exit(1)
        model_rows.append((l1, l2, m, get_id_test_accuracy(m)))
    
    for d in datasets:
        l1, l2 = parse_folder_name(d.name, dp_name)
        if l1 == "Others" or l1 not in L1_ORDER:
            logger.error(f"Unknown L1 category for dataset: {d.name} -> {l1}. Aborting.")
            sys.exit(1)
        dataset_cols.append((l1, l2, d))

    tasks = []
    for ml1, ml2, m_path, id_vals in model_rows:
        row_key = (ml1, ml2)
        for m_type in ["logreg", "mlp", "mlp_domain"]:
            if row_key not in report_data[m_type]:
                report_data[m_type][row_key] = {"ID-TEST": id_vals.get(m_type, "N/A")}
                
        for dl1, dl2, d_path in dataset_cols:
            col_key = (dl1, dl2)
            for m_type in ["logreg", "mlp", "mlp_domain"]:
                report_data[m_type][row_key][col_key] = "N/A"
            
            tasks.append({
                "mp_name": mp_name,
                "dp_name": dp_name,
                "m_path": m_path,
                "d_path": d_path,
                "group_out_dir": group_out_dir,
                "row_key": row_key,
                "col_key": col_key
            })

    return {
        "group_out_dir": group_out_dir,
        "report_data": report_data,
        "model_rows": model_rows,
        "dataset_cols": dataset_cols,
        "tasks": tasks
    }

def generate_excel_report(config):
    group_out_dir = config["group_out_dir"]
    report_data = config["report_data"]
    model_rows = config["model_rows"]
    dataset_cols = config["dataset_cols"]
    
    L1_ORDER = ["qwen-exit", "qwen-scratch", "qwen-exit-scratch", "ds-coder", "7b-pfc"]
    
    excel_path = group_out_dir / "summary_report.xlsx"
    sorted_cols = sorted(dataset_cols, key=lambda x: (L1_ORDER.index(x[0]), x[1]))
    col_index = pd.MultiIndex.from_tuples([(x[0], x[1]) for x in sorted_cols], names=['Type', 'Detail'])
    sorted_rows = sorted(model_rows, key=lambda x: (L1_ORDER.index(x[0]), x[1]))
    row_index = pd.MultiIndex.from_tuples([(x[0], x[1]) for x in sorted_rows], names=['Type', 'Detail'])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for m_type in ["logreg", "mlp", "mlp_domain"]:
            df = pd.DataFrame(index=row_index, columns=col_index)
            id_test_col = []
            for r in sorted_rows:
                r_key = (r[0], r[1])
                id_test_col.append(report_data[m_type][r_key].get("ID-TEST", "N/A"))
                for c in sorted_cols:
                    c_key = (c[0], c[1])
                    df.loc[r_key, c_key] = report_data[m_type][r_key].get(c_key, "N/A")
            
            df.insert(0, ('ID-TEST', ''), id_test_col)
            df.to_excel(writer, sheet_name=m_type)
            
    logger.success(f"Report generated: {excel_path}")

def collect_reports(results_root):
    collect_dir = results_root / "_collected_reports"
    collect_dir.mkdir(exist_ok=True)
    logger.info(f"Collecting all summary reports into: {collect_dir}")
    count = 0
    for subdir in results_root.iterdir():
        if subdir.is_dir() and subdir.name != "_collected_reports":
            src_file = subdir / "summary_report.xlsx"
            if src_file.exists():
                dest_path = collect_dir / f"{subdir.name}_summary_report.xlsx"
                try:
                    shutil.copy2(src_file, dest_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to copy from {subdir.name}: {e}")
                    
    if count > 0:
        logger.success(f"Successfully collected {count} Excel files to: {collect_dir}")

# ==========================================
# 7. Main Execution (Flattened Parallel Pool)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_grandparent_folder", "-mgp", type=str, required=True)
    parser.add_argument("--test_dataset_grandparent_folder", "-dgp", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--results_root", type=str, default="../../main_table3_paired/5a_results_auto_v3")
    parser.add_argument("--workers", type=int, default=48, help="Number of parallel processes")
    args = parser.parse_args()
    
    mgp = pathlib.Path(args.model_grandparent_folder)
    dgp = pathlib.Path(args.test_dataset_grandparent_folder)
    results_root = pathlib.Path(args.results_root)
    
    if not mgp.exists() or not dgp.exists():
        logger.error("Grandparent folders not found.")
        return

    mps = [d for d in mgp.iterdir() if d.is_dir()]
    dps = [d for d in dgp.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(mps)} Model Parents and {len(dps)} Dataset Parents.")
    
    # 1. 扁平化提取所有测试任务
    global_tasks = []
    matrix_configs = {}
    for mp in mps:
        for dp in dps:
            config = prepare_matrix_group(mp, dp, results_root)
            if config:
                global_tasks.extend(config["tasks"])
                matrix_configs[(mp.name, dp.name)] = config

    logger.info(f"Flattened matrix. Launching {len(global_tasks)} tasks across {args.workers} workers...")

    # 2. 启动全局巨型线程池
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(evaluate_single_run, t["m_path"], [t["d_path"]], args.seed, t["group_out_dir"]): t
            for t in global_tasks
        }
        
        # 只显示一个总的大进度条，体验极度丝滑
        for future in tqdm(as_completed(future_to_task), total=len(global_tasks), desc="Global Matrix Eval"):
            task = future_to_task[future]
            try:
                res = future.result()
                if res:
                    mp_name, dp_name = task["mp_name"], task["dp_name"]
                    report_data = matrix_configs[(mp_name, dp_name)]["report_data"]
                    
                    for m_type in ["logreg", "mlp", "mlp_domain"]:
                        if m_type in res["summary"] and "accuracy" in res["summary"][m_type]:
                            stats = res["summary"][m_type]["accuracy"]
                            val_str = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
                            report_data[m_type][task["row_key"]][task["col_key"]] = val_str
            except Exception as e:
                logger.error(f"Task failed: {e}")

    # 3. 统一生成各个子目录的 Excel 报表
    for config in matrix_configs.values():
        generate_excel_report(config)
            
    # 4. 统一收集报表到 _collected_reports
    logger.info("--- Matrix Evaluation Finished. Starting Report Collection ---")
    collect_reports(results_root)

if __name__ == "__main__":
    main()