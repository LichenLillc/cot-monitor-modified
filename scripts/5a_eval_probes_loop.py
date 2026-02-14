"""
Script: 5a_eval_probes.py

Description:
    Tests pre-trained probes on NEW datasets (Transfer/OOD Evaluation).
    
    Updates:
    - [NEW] All metrics are reported on a 0-100 scale (Percentage).
    - [NEW] Output format is now a horizontal table (Model x Metrics).
    - [NEW] Auto-detects single-class datasets and only reports Accuracy.
    - Supports MATRIX EVALUATION: Loop over multiple models and multiple datasets automatically.
    - Buffers summary tables and prints them all at the end.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import pathlib
import joblib
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from loguru import logger
import re
import collections
import io

# ==========================================
# 1. MLP Definition
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
# 2. Data Loading
# ==========================================
def load_activations(input_folder):
    """
    Loads activations and labels from a SINGLE folder.
    Returns numpy arrays (X, y).
    """
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
            # Load Activation
            act = torch.load(activation_file, map_location='cpu')
            if act.dim() > 1:
                act = act.view(-1)
            xs.append(act.numpy())
            
            # Load Label
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                # Ensure 0/1 consistency: Score >= 0.5 is Class 1
                score = label_data["safety_label"]["score"]
                ys.append(1 if score >= 0.5 else 0)
                
        except Exception as e:
            continue
            
    if len(xs) == 0:
        return None, None

    return np.array(xs), np.array(ys)

# ==========================================
# 3. Helper: Calculate Metrics (MODIFIED)
# ==========================================
def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculates metrics. 
    If y_true has only one class, ONLY calculates Accuracy.
    """
    # Always calc Accuracy
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100
    }
    
    # Only calc advanced metrics if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics["f1"] = float(f1_score(y_true, y_pred)) * 100
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_probs)) * 100
        metrics["pr_auc"] = float(average_precision_score(y_true, y_probs)) * 100
            
    return metrics

# ==========================================
# 4. Core Evaluation Logic (MODIFIED)
# ==========================================
def evaluate_single_run(model_folder_path, test_dataset_paths, seed, results_dir):
    """
    Executes evaluation for ONE Model Folder vs ONE (or merged) Test Dataset(s).
    Returns: A formatted string containing the summary table.
    """
    model_dir = model_folder_path / "saved_models"
    if not model_dir.exists():
        logger.error(f"No 'saved_models' found in {model_folder_path}")
        return None

    # --- Identify Seeds ---
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
        logger.warning(f"No seeds found in {model_folder_path.name}. Skipping.")
        return None

    # --- Load Test Data ---
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
        logger.warning(f"No valid data found in {combined_dataset_name}. Skipping.")
        return None

    X_raw_global = np.concatenate(all_xs, axis=0)
    y_test_global = np.concatenate(all_ys, axis=0)
    
    # --- Loop Over Seeds ---
    agg_metrics = {
        "logreg": collections.defaultdict(list),
        "mlp": collections.defaultdict(list)
    }
    detailed_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iterator = seeds_to_run 

    for s in iterator:
        try:
            # Define Paths
            global_scaler_path = model_dir / f"global_scaler_seed{s}.joblib"
            mlp_scaler_path = model_dir / f"mlp_internal_scaler_seed{s}.joblib"
            pca_path = model_dir / f"pca_seed{s}.joblib"
            logreg_path = model_dir / f"logreg_seed{s}.joblib"
            mlp_path = model_dir / f"mlp_seed{s}.pth"

            if not global_scaler_path.exists() or not mlp_scaler_path.exists() or not logreg_path.exists() or not mlp_path.exists():
                continue

            # Load & Transform
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

            # Store (Note: this loop is dynamic, it only stores what calculate_metrics returns)
            for k, v in metrics_lr.items(): agg_metrics["logreg"][k].append(v)
            for k, v in metrics_mlp.items(): agg_metrics["mlp"][k].append(v)
            
            detailed_results.append({"seed": s, "logreg": metrics_lr, "mlp": metrics_mlp})

        except Exception as e:
            logger.error(f"Error seed {s}: {e}")
            continue

    if not detailed_results:
        return None

    # --- Save JSON (MODIFIED: Dynamic Metrics) ---
    final_output = {
        "source_model": model_folder_path.name,
        "target_dataset": combined_dataset_name,
        "scale": "0-100", 
        "seeds_evaluated": [d['seed'] for d in detailed_results],
        "summary": {},
        "detailed_runs": detailed_results
    }
    
    # Dynamically determine which metrics were actually computed
    computed_metrics_keys = list(agg_metrics["logreg"].keys())
    
    # Define a preferred order for display/saving
    full_metric_order = ["accuracy", "f1", "pr_auc", "auc_roc"]
    # Filter to keep only those present in results
    active_metrics = [m for m in full_metric_order if m in computed_metrics_keys]

    # Populate Summary Dictionary
    for model_name in ["logreg", "mlp"]:
        final_output["summary"][model_name] = {}
        for metric in active_metrics:
            values = agg_metrics[model_name][metric]
            final_output["summary"][model_name][metric] = {
                "mean": np.mean(values), 
                "std": np.std(values), 
                "values": values
            }

    # --- Generate Summary Table String (MODIFIED: Dynamic Header) ---
    output_buffer = io.StringIO()
    output_buffer.write("\n" + "="*95 + "\n")
    output_buffer.write(f"TRANSFER SUMMARY (Scale: 0-100)\n")
    output_buffer.write(f"Model:  {model_folder_path.name}\n")
    output_buffer.write(f"Target: {combined_dataset_name}\n")
    output_buffer.write(f"Seeds:  {len(detailed_results)}\n")
    output_buffer.write("-" * 95 + "\n")
    
    # 1. Generate Header Dynamically
    header_str = f"{'Model':<10} | "
    display_names = {"accuracy": "Accuracy", "f1": "F1", "pr_auc": "PR-AUC", "auc_roc": "ROC-AUC"}
    
    for m in active_metrics:
        header_str += f"{display_names[m]:<16} | "
    
    output_buffer.write(header_str[:-2] + "\n") # Remove last " | "
    output_buffer.write("-" * 95 + "\n")

    # 2. Generate Rows Dynamically
    for model_name in ["logreg", "mlp"]:
        display_name = "LogReg" if model_name == "logreg" else "MLP"
        row_str = f"{display_name:<10} | "
        
        for metric in active_metrics:
            stats = final_output["summary"][model_name][metric]
            val_str = f"{stats['mean']:.2f} ± {stats['std']:.2f}"
            row_str += f"{val_str:<16} | "
        
        output_buffer.write(row_str[:-2] + "\n")
        
    output_buffer.write("-" * 95 + "\n")

    # Save file
    seed_suffix = f"_seed{detailed_results[0]['seed']}" if len(detailed_results) == 1 else "_ALL_SEEDS"
    out_filename = f"{model_folder_path.name}_ON_{combined_dataset_name}{seed_suffix}.json"
    out_filename = re.sub(r'[^\w\-.+]', '_', out_filename)
    out_path = results_dir / out_filename
    
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    
    logger.success(f"Saved: {out_filename}")
    
    return output_buffer.getvalue()

# ==========================================
# 5. Main Loop & Batch Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # Batch Parameters
    parser.add_argument("--model_parent_folder", "-mp", type=str, default=None,
                        help="Parent folder containing multiple model folders")
    parser.add_argument("--test_dataset_parent_folder", "-dp", type=str, default=None,
                        help="Parent folder containing multiple dataset folders")
    
    # Original Parameters (Optional if parents provided)
    parser.add_argument("--test_dataset_folder", type=str, nargs='+', default=None, 
                        help="Specific path(s) to dataset(s). Ignored if -dp is used.")
    parser.add_argument("--model_folder", type=str, default=None, 
                        help="Specific path to model folder. Ignored if -mp is used.")
    
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to load.")
    parser.add_argument("--results_folder", type=str, default="../probe_main-table_debug/5a_results/TRAIN-ckpt61_TEST-Qwen7B", help="Where to save results")
    
    args = parser.parse_args()
    results_dir = pathlib.Path(args.results_folder)
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Identify Model Folders ---
    model_folders_list = []
    if args.model_parent_folder:
        mp = pathlib.Path(args.model_parent_folder)
        candidates = [d for d in mp.iterdir() if d.is_dir()]
        for d in candidates:
            if (d / "saved_models").exists():
                model_folders_list.append(d)
        logger.info(f"Found {len(model_folders_list)} valid model folders in {mp}")
    elif args.model_folder:
        model_folders_list = [pathlib.Path(args.model_folder)]
    else:
        logger.error("Must provide either --model_parent_folder (-mp) or --model_folder")
        return

    # --- 2. Identify Test Dataset Groups ---
    dataset_groups_list = [] 
    
    if args.test_dataset_parent_folder:
        dp = pathlib.Path(args.test_dataset_parent_folder)
        candidates = [d for d in dp.iterdir() if d.is_dir()]
        valid_datasets = []
        for d in candidates:
            # Filter Logic: Must have activations/, labels/, AND "test" in name (case-insensitive)
            if (d / "activations").exists() and (d / "labels").exists():
                if "test" in d.name.lower():
                    valid_datasets.append(d)
        
        valid_datasets.sort(key=lambda x: x.name)
        dataset_groups_list = [[d] for d in valid_datasets]
        logger.info(f"Found {len(dataset_groups_list)} valid test datasets (matching 'test') in {dp}")
        
    elif args.test_dataset_folder:
        p_list = [pathlib.Path(p) for p in args.test_dataset_folder]
        dataset_groups_list = [p_list]
    else:
        logger.error("Must provide either --test_dataset_parent_folder (-dp) or --test_dataset_folder")
        return

    if not model_folders_list or not dataset_groups_list:
        logger.error("No valid models or datasets found. Exiting.")
        return

    # --- 3. Matrix Loop ---
    summary_prints = []
    
    total_tasks = len(model_folders_list) * len(dataset_groups_list)
    logger.info(f"Starting Matrix Evaluation: {len(model_folders_list)} Models x {len(dataset_groups_list)} Datasets = {total_tasks} Tasks")

    pbar = tqdm(total=total_tasks, desc="Overall Progress")
    
    for model_path in model_folders_list:
        for dataset_paths in dataset_groups_list:
            
            result_str = evaluate_single_run(model_path, dataset_paths, args.seed, results_dir)
            
            if result_str:
                summary_prints.append(result_str)
            
            pbar.update(1)
            
    pbar.close()

    # --- 4. Final Print ---
    print("\n\n" + "#"*95)
    print("FINAL BATCH EVALUATION REPORT")
    print("#"*95)
    
    for s in summary_prints:
        print(s)

if __name__ == "__main__":
    main()