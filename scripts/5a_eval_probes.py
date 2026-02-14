"""
Script: 5a_eval_probes.py

Description:
    Tests pre-trained probes on NEW datasets (Transfer/OOD Evaluation).
    
    Updates:
    - Supports multiple test datasets (merges them).
    - Supports evaluating across ALL available seeds if --seed is not specified.
    - Reports results as "Mean ± Std" when multiple seeds are evaluated.
    - Maintains the 3-stage preprocessing (Global Scaler -> PCA -> MLP Scaler).

Usage:
    # Run specific seed
    python 5a_eval_probes.py --test_dataset_folder ./processed/ds1 --model_folder ./models --seed 42

    # Run ALL seeds found in folder (Auto-Aggregate)
    python 5a_eval_probes.py --test_dataset_folder ./processed/ds1 ./processed/ds2 --model_folder ./models

python3 5a_eval_probes.py --test_dataset_folder /home/Lichen/cot-monitor-modified/processed/7b_pfc-prompted_n-lc-88_m22_qwen_1p5b --model_folder /home/Lichen/cot-monitor-modified/probe_outputs/dup4_lc-h-700_lc-n-700_qwen_1p5b
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

# ==========================================
# 1. MLP Definition (Matches 3a exactly)
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
        logger.warning(f"Skipping {input_folder}: Must contain 'activations' and 'labels' subdirectories.")
        return None, None

    xs = []
    ys = []
    
    pt_files = list(activations_path.glob("*.pt"))
    if not pt_files:
        logger.warning(f"Skipping {input_folder}: No .pt files found.")
        return None, None

    logger.info(f"Scanning {input_folder.name}: Found {len(pt_files)} activation files.")

    for activation_file in tqdm(pt_files, desc=f"Loading from {input_folder.name}", leave=False):
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
                ys.append(int(label_data["safety_label"]["score"]))
                
        except Exception as e:
            continue
            
    if len(xs) == 0:
        logger.warning(f"No valid data pairs loaded from {input_folder}.")
        return None, None

    return np.array(xs), np.array(ys)

# ==========================================
# 3. Helper: Calculate Metrics
# ==========================================
def calculate_metrics(y_true, y_pred, y_probs):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_probs)),
        "pr_auc": float(average_precision_score(y_true, y_probs))
    }

def format_stat(values):
    """Returns 'mean ± std' string."""
    if not values:
        return "N/A"
    return f"{np.mean(values):.4f} ± {np.std(values):.4f}"

# ==========================================
# 4. Main Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset_folder", type=str, nargs='+', required=True, 
                        help="Path(s) to the NEW dataset(s) to test on.")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the folder containing saved models")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to load. If None, runs ALL seeds found.")
    parser.add_argument("--results_folder", type=str, default="../probe_main-table/5a_results", help="Where to save the evaluation json")
    
    args = parser.parse_args()
    
    model_folder_dir = pathlib.Path(args.model_folder)
    model_dir = model_folder_dir/"saved_models"
    results_dir = pathlib.Path(args.results_folder)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # A. Identify Seeds
    # ---------------------------------------------------------
    seeds_to_run = []
    
    if args.seed is not None:
        seeds_to_run = [args.seed]
        logger.info(f"Running for specified seed: {args.seed}")
    else:
        # Scan for all available seeds
        logger.info(f"No seed specified. Scanning {model_dir} for models...")
        potential_files = list(model_dir.glob("logreg_seed*.joblib"))
        extracted_seeds = []
        for p in potential_files:
            match = re.search(r"seed(\d+)", p.name)
            if match:
                extracted_seeds.append(int(match.group(1)))
        
        seeds_to_run = sorted(list(set(extracted_seeds)))
        if not seeds_to_run:
            logger.error(f"No seeded models found in {model_dir}")
            return
        logger.info(f"Found {len(seeds_to_run)} seeds: {seeds_to_run}")

    # ---------------------------------------------------------
    # B. Load Raw Test Data (ONCE)
    # ---------------------------------------------------------
    test_data_dirs = [pathlib.Path(p) for p in args.test_dataset_folder]
    dataset_names = [d.name for d in test_data_dirs]
    combined_dataset_name = f"Combined_{len(dataset_names)}_Datasets" if len(dataset_names) > 3 else "+".join(dataset_names)
    
    all_xs = []
    all_ys = []

    logger.info("Loading Raw Test Data...")
    for d in test_data_dirs:
        xs, ys = load_activations(d)
        if xs is not None and ys is not None:
            all_xs.append(xs)
            all_ys.append(ys)
    
    if not all_xs:
        logger.error("No valid data found.")
        return

    X_raw_global = np.concatenate(all_xs, axis=0)
    y_test_global = np.concatenate(all_ys, axis=0)
    logger.info(f"Total Test Data: {X_raw_global.shape[0]} samples.")

    # ---------------------------------------------------------
    # C. Loop Over Seeds
    # ---------------------------------------------------------
    
    # Storage for aggregation
    agg_metrics = {
        "logreg": collections.defaultdict(list),
        "mlp": collections.defaultdict(list)
    }
    
    detailed_results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in tqdm(seeds_to_run, desc="Evaluating Seeds"):
        try:
            # 1. Define Paths for this seed
            global_scaler_path = model_dir / f"global_scaler_seed{seed}.joblib"
            mlp_scaler_path = model_dir / f"mlp_internal_scaler_seed{seed}.joblib"
            pca_path = model_dir / f"pca_seed{seed}.joblib"
            logreg_path = model_dir / f"logreg_seed{seed}.joblib"
            mlp_path = model_dir / f"mlp_seed{seed}.pth"

            # 2. Load Artifacts
            if not global_scaler_path.exists() or not mlp_scaler_path.exists() or not logreg_path.exists() or not mlp_path.exists():
                logger.warning(f"Missing artifacts for seed {seed}, skipping.")
                continue

            global_scaler = joblib.load(global_scaler_path)
            mlp_scaler = joblib.load(mlp_scaler_path)
            logreg = joblib.load(logreg_path)
            
            pca = None
            if pca_path.exists():
                pca = joblib.load(pca_path)

            # 3. Preprocess for this seed
            # (Scalers are fit on training data of this specific seed, so must transform raw test data again)
            X_stage1 = global_scaler.transform(X_raw_global)
            
            if pca:
                X_stage2 = pca.transform(X_stage1)
            else:
                X_stage2 = X_stage1
                
            X_stage3 = mlp_scaler.transform(X_stage2)

            # 4. Evaluate LogReg
            y_probs_lr = logreg.predict_proba(X_stage2)[:, 1]
            y_pred_lr = logreg.predict(X_stage2)
            metrics_lr = calculate_metrics(y_test_global, y_pred_lr, y_probs_lr)

            # 5. Evaluate MLP
            input_dim = X_stage3.shape[1]
            mlp = CustomMLP2Layer(input_size=input_dim).to(device)
            mlp.load_state_dict(torch.load(mlp_path, map_location=device))
            mlp.eval()
            
            X_mlp_tensor = torch.tensor(X_stage3, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs_mlp = mlp(X_mlp_tensor).squeeze().cpu().numpy()
                y_pred_mlp = (probs_mlp > 0.5).astype(int)
            metrics_mlp = calculate_metrics(y_test_global, y_pred_mlp, probs_mlp)

            # 6. Store
            for k, v in metrics_lr.items():
                agg_metrics["logreg"][k].append(v)
            for k, v in metrics_mlp.items():
                agg_metrics["mlp"][k].append(v)
            
            detailed_results.append({
                "seed": seed,
                "logreg": metrics_lr,
                "mlp": metrics_mlp
            })

        except Exception as e:
            logger.error(f"Error evaluating seed {seed}: {e}")
            continue

    # ---------------------------------------------------------
    # D. Aggregate & Print Results
    # ---------------------------------------------------------
    if not detailed_results:
        logger.error("No seeds were successfully evaluated.")
        return

    # Prepare Final JSON Structure
    final_output = {
        "source_model": model_folder_dir.name,
        "target_dataset": combined_dataset_name,
        "seeds_evaluated": [d['seed'] for d in detailed_results],
        "summary": {},
        "detailed_runs": detailed_results
    }
    
    # Calculate Mean ± Std
    print("\n" + "="*80)
    print(f"TRANSFER EVALUATION SUMMARY")
    print(f"Source Model: {model_folder_dir.name}")
    print(f"Target Data:  {combined_dataset_name}")
    print(f"Seeds:        {len(detailed_results)} runs")
    print("-" * 80)
    print(f"{'Model':<10} | {'Metric':<10} | {'Mean ± Std'}")
    print("-" * 80)

    for model_name in ["logreg", "mlp"]:
        final_output["summary"][model_name] = {}
        for metric in ["accuracy", "f1", "auc_roc", "pr_auc"]:
            values = agg_metrics[model_name][metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            stat_str = f"{mean_val:.4f} ± {std_val:.4f}"
            print(f"{model_name:<10} | {metric:<10} | {stat_str}")
            
            final_output["summary"][model_name][metric] = {
                "mean": mean_val,
                "std": std_val,
                "values": values
            }
        print("-" * 80)
    print("="*80 + "\n")

    # Save to file
    # If single seed, use old naming convention. If multiple, indicate summary.
    if len(detailed_results) == 1:
        seed_suffix = f"_seed{detailed_results[0]['seed']}"
    else:
        seed_suffix = "_ALL_SEEDS"
        
    out_filename = f"{model_folder_dir.name}_ON_{combined_dataset_name}{seed_suffix}.json"
    out_filename = re.sub(r'[^\w\-.+]', '_', out_filename)
    
    out_path = results_dir / out_filename
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
        
    logger.success(f"Full results saved to: {out_path}")

if __name__ == "__main__":
    main()