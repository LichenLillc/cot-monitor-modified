"""
Script: 5a_eval_probes_loop_v2.py

Description:
    Fully automated Matrix Evaluation for Probes.
    
    Features:
    - Super-Matrix Loop: Iterates over Grandparent Folders (-mgp, -dgp).
    - Auto-Tabulation: Generates .xlsx reports with 2-level headers and merged cells.
    - ID-TEST Extraction: Reads training accuracy from training_summary.txt.
    - Header Parsing: Automatically extracts L1/L2 categories from folder names.
    - Smart Metrics: Only calcs Accuracy for single-class datasets; 0-100 scale.
    
Dependencies:
    pandas, openpyxl, sklearn, torch, loguru, tqdm, joblib, numpy
"""

import os
import sys
import re
import json
import argparse
import pathlib
import collections
import joblib
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# ==========================================
# 1. MLP Definition (Unchanged)
# ==========================================
class CustomMLP2Layer(nn.Module):
    def __init__(self, input_size, hidden_size1=100, hidden_size2=50):
        super(CustomMLP2Layer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x) 
        x = self.sigmoid(x)
        return x

# ==========================================
# 2. Data Loading (Unchanged)
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
        parts = activation_file.stem.split("_")
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
# 4. Evaluation Logic (Modified Return)
# ==========================================
def evaluate_single_run(model_folder_path, test_dataset_paths, seed, results_dir):
    """
    Returns: Dictionary containing final results (not string).
    """
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
    for d in test_dataset_paths:
        xs, ys = load_activations(d)
        if xs is not None and ys is not None:
            all_xs.append(xs)
            all_ys.append(ys)
    
    if not all_xs:
        return None

    X_raw_global = np.concatenate(all_xs, axis=0)
    y_test_global = np.concatenate(all_ys, axis=0)
    
    agg_metrics = {
        "logreg": collections.defaultdict(list),
        "mlp": collections.defaultdict(list)
    }
    detailed_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

            # LogReg
            y_probs_lr = logreg.predict_proba(X_stage2)[:, 1]
            y_pred_lr = logreg.predict(X_stage2)
            metrics_lr = calculate_metrics(y_test_global, y_pred_lr, y_probs_lr)

            # MLP
            input_dim = X_stage3.shape[1]
            mlp = CustomMLP2Layer(input_size=input_dim).to(device)
            mlp.load_state_dict(torch.load(mlp_path, map_location=device))
            mlp.eval()
            X_mlp_tensor = torch.tensor(X_stage3, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs_mlp = mlp(X_mlp_tensor).squeeze().cpu().numpy()
                y_pred_mlp = (probs_mlp > 0.5).astype(int)
            metrics_mlp = calculate_metrics(y_test_global, y_pred_mlp, probs_mlp)

            for k, v in metrics_lr.items(): agg_metrics["logreg"][k].append(v)
            for k, v in metrics_mlp.items(): agg_metrics["mlp"][k].append(v)
            
            detailed_results.append({"seed": s, "logreg": metrics_lr, "mlp": metrics_mlp})

        except Exception:
            continue

    if not detailed_results:
        return None

    final_output = {
        "source_model": model_folder_path.name,
        "target_dataset": combined_dataset_name,
        "summary": {},
        "detailed_runs": detailed_results
    }

    # Summary
    active_metrics = list(agg_metrics["logreg"].keys())
    for model_name in ["logreg", "mlp"]:
        final_output["summary"][model_name] = {}
        for metric in active_metrics:
            values = agg_metrics[model_name][metric]
            final_output["summary"][model_name][metric] = {
                "mean": np.mean(values), 
                "std": np.std(values)
            }

    # Save JSON
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
    """
    Parses L1 and L2 categories based on priority rules.
    """
    name = folder_name
    # 1. Remove suffix
    if suffix_to_remove and name.endswith(f"_{suffix_to_remove}"):
        name = name[:-(len(suffix_to_remove) + 1)] # remove _suffix
    
    # 2. Remove "test" (case insensitive)
    name = re.sub(r'_?test_?', '_', name, flags=re.IGNORECASE)

    # 3. Determine L1 Priority
    l1 = "Others"
    token_to_remove = ""

    if "answer" in name:
        l1 = "syn-cot-code"
        token_to_remove = "answer" # will also clean prefix later
        # Also assume prefix contains cot, need to clean carefully
    elif "think-ins" in name:
        l1 = "syn-cot-think-ins"
        token_to_remove = "think-ins"
    elif "no-cot" in name:
        l1 = "syn-no-cot"
        token_to_remove = "no-cot"
    elif "cot" in name: 
        # Low priority COT
        l1 = "syn-cot"
        token_to_remove = "cot"
    elif "wild" in name:
        l1 = "wild"
        token_to_remove = "wild_dup4" # specific for wild
    else:
        # Fallback/Error check
        pass

    # 4. Clean L2
    # Remove common prefixes chunks based on L1
    # We essentially remove the L1 identifier and standardized prefixes
    
    # Simple strategy: Replace known keywords with empty, then strip underscores
    # Keywords to strip for L2 cleaning:
    keywords = ["7b_pfc", "cot", "think-ins", "no-cot", "answer", "wild", "dup4"]
    
    l2_candidate = name
    for k in keywords:
        l2_candidate = l2_candidate.replace(k, "")
    
    # Clean underscores
    l2 = re.sub(r'_+', '_', l2_candidate).strip('_')
    
    return l1, l2

def get_id_test_accuracy(model_folder):
    """
    Reads training_summary.txt to get ID-TEST accuracy.
    """
    summary_path = model_folder / "training_summary.txt"
    if not summary_path.exists():
        return {"logreg": "N/A", "mlp": "N/A"}
    
    try:
        content = summary_path.read_text(encoding='utf-8')
        # Look for the table at the end
        # Model | f1 | accuracy ...
        # `logreg ` | ... | 99.2 ± 0.3 | ...
        
        results = {}
        for model_key in ["logreg", "mlp"]:
            # Regex: Look for line starting with backticked or plain model name
            # Capture the 3rd column (Accuracy)
            # Pattern: model_name [spaces] | [f1 col] | [ACCURACY COL] | ...
            # We assume | is the separator
            
            # Simple line scanner
            for line in content.splitlines():
                if model_key in line.lower() and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    # parts[0] is model, parts[1] is f1, parts[2] is accuracy
                    if len(parts) >= 3:
                        results[model_key] = parts[2] # "99.2 ± 0.3"
                        break
        
        return results
    except Exception as e:
        logger.warning(f"Failed to parse training_summary.txt for {model_folder.name}: {e}")
        return {"logreg": "N/A", "mlp": "N/A"}

# ==========================================
# 6. Main Matrix Logic
# ==========================================
def run_matrix_group(mp_path, dp_path, seed, results_root):
    """
    Runs the LxM matrix for one pair of Parent Folders.
    Generates Excel report at the end.
    """
    mp_name = mp_path.name
    dp_name = dp_path.name
    
    # Output Dir
    group_out_dir = results_root / f"TRAIN-{mp_name}_TEST-{dp_name}"
    group_out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing Group: TRAIN-{mp_name} vs TEST-{dp_name}")

    # 1. Scan Valid Model Folders
    models = []
    for d in mp_path.iterdir():
        if d.is_dir() and (d / "saved_models").exists():
            models.append(d)
    
    # 2. Scan Valid Dataset Folders
    datasets = []
    for d in dp_path.iterdir():
        if d.is_dir() and (d / "activations").exists() and (d / "labels").exists():
            if "test" in d.name.lower():
                datasets.append(d)
    
    if not models or not datasets:
        logger.warning(f"Skipping group {group_out_dir.name}: Missing models or datasets.")
        return

    # Data Structure for Report:
    # table_data[model_key]["id_test"] = val
    # table_data[model_key][(ds_l1, ds_l2)] = val
    report_data = {
        "logreg": {},
        "mlp": {}
    }
    
    # Pre-parse Headers to ensure order and error checking
    # Defined Sort Order
    L1_ORDER = ["wild", "syn-cot-think-ins", "syn-cot", "syn-cot-code", "syn-no-cot"]
    
    # Prepare Row/Col Indices
    model_rows = [] # List of (l1, l2, folder_path, id_test_vals)
    dataset_cols = [] # List of (l1, l2, folder_path)

    # Process Models (Rows)
    for m in models:
        l1, l2 = parse_folder_name(m.name, mp_name)
        if l1 == "Others" or l1 not in L1_ORDER:
            # You requested to Error if Others
            logger.error(f"Unknown L1 category for model: {m.name} -> {l1}. Aborting.")
            sys.exit(1)
        
        id_test_vals = get_id_test_accuracy(m)
        model_rows.append((l1, l2, m, id_test_vals))
    
    # Process Datasets (Cols)
    for d in datasets:
        l1, l2 = parse_folder_name(d.name, dp_name)
        if l1 == "Others" or l1 not in L1_ORDER:
            logger.error(f"Unknown L1 category for dataset: {d.name} -> {l1}. Aborting.")
            sys.exit(1)
        dataset_cols.append((l1, l2, d))

    # Run Evaluations
    # Loop Models
    for ml1, ml2, m_path, id_vals in tqdm(model_rows, desc="Models", leave=False):
        
        # Init row in report if needed
        row_key = (ml1, ml2)
        
        # Store ID-TEST
        if row_key not in report_data["logreg"]: 
            report_data["logreg"][row_key] = {"ID-TEST": id_vals.get("logreg", "N/A")}
            report_data["mlp"][row_key] = {"ID-TEST": id_vals.get("mlp", "N/A")}

        # Loop Datasets
        for dl1, dl2, d_path in dataset_cols:
            col_key = (dl1, dl2)
            
            # Eval
            res = evaluate_single_run(m_path, [d_path], seed, group_out_dir)
            
            if res:
                # Extract Accuracy (Mean ± Std)
                for m_type in ["logreg", "mlp"]:
                    stats = res["summary"][m_type]["accuracy"]
                    val_str = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
                    report_data[m_type][row_key][col_key] = val_str
            else:
                for m_type in ["logreg", "mlp"]:
                    report_data[m_type][row_key][col_key] = "N/A"

    # --- Generate Excel ---
    excel_path = group_out_dir / "summary_report.xlsx"
    
    # Sort Columns
    # Filter datasets that exist in L1_ORDER
    sorted_cols = sorted(dataset_cols, key=lambda x: (L1_ORDER.index(x[0]), x[1]))
    col_index = pd.MultiIndex.from_tuples([(x[0], x[1]) for x in sorted_cols], names=['Type', 'Detail'])
    
    # Sort Rows
    sorted_rows = sorted(model_rows, key=lambda x: (L1_ORDER.index(x[0]), x[1]))
    row_index = pd.MultiIndex.from_tuples([(x[0], x[1]) for x in sorted_rows], names=['Type', 'Detail'])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for m_type in ["logreg", "mlp"]:
            # Build DataFrame
            # Init empty DF
            df = pd.DataFrame(index=row_index, columns=col_index)
            
            # Fill Data
            id_test_col = []
            for r in sorted_rows:
                r_key = (r[0], r[1])
                id_test_col.append(report_data[m_type][r_key]["ID-TEST"])
                
                for c in sorted_cols:
                    c_key = (c[0], c[1])
                    val = report_data[m_type][r_key].get(c_key, "N/A")
                    df.loc[r_key, c_key] = val
            
            # Insert ID-TEST at the beginning (Level 0 col)
            # To mix MultiIndex cols with a single col is tricky in pandas.
            # Easiest way: Add it as a column ('ID-TEST', '')
            df.insert(0, ('ID-TEST', ''), id_test_col)
            
            df.to_excel(writer, sheet_name=m_type)
            
    logger.success(f"Report generated: {excel_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_grandparent_folder", "-mgp", type=str, required=True)
    parser.add_argument("--test_dataset_grandparent_folder", "-dgp", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    # Default output root
    parser.add_argument("--results_root", type=str, default="../probe_main-table_debug/5a_results_auto")
    
    args = parser.parse_args()
    
    mgp = pathlib.Path(args.model_grandparent_folder)
    dgp = pathlib.Path(args.test_dataset_grandparent_folder)
    results_root = pathlib.Path(args.results_root)
    
    if not mgp.exists() or not dgp.exists():
        logger.error("Grandparent folders not found.")
        return

    # Layer 1: Iterate MP
    mps = [d for d in mgp.iterdir() if d.is_dir()]
    # Layer 2: Iterate DP
    dps = [d for d in dgp.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(mps)} Model Parents and {len(dps)} Dataset Parents.")
    
    # Matrix Loop
    for mp in mps:
        for dp in dps:
            run_matrix_group(mp, dp, args.seed, results_root)

if __name__ == "__main__":
    main()