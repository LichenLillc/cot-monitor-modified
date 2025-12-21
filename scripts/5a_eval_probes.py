"""
Script: 5a_eval_probes.py

Description:
    Tests pre-trained probes on a NEW dataset (Transfer/OOD Evaluation).
    
    Updates:
    - Calculates and prints: Accuracy, F1, AUC-ROC, PR-AUC.
    - Corrected MLP scaler filename to 'mlp_internal_scaler_seedX.joblib'.
    - Maintains the 3-stage preprocessing (Global Scaler -> PCA -> MLP Scaler).

Usage:
    python 5a_eval_probes.py \
      --test_dataset_folder ./processed/target_dataset \
      --model_folder ./probe_outputs/source_model/saved_models
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
    input_folder = pathlib.Path(input_folder)
    activations_path = input_folder / "activations"
    labels_path = input_folder / "labels"
    
    if not activations_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Input folder must contain 'activations' and 'labels' subdirectories. Found: {input_folder}")

    xs = []
    ys = []
    
    pt_files = list(activations_path.glob("*.pt"))
    if not pt_files:
        raise ValueError(f"No .pt files found in {activations_path}")

    logger.info(f"Found {len(pt_files)} activation files. Loading...")

    for activation_file in tqdm(pt_files, desc="Loading test data"):
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
            logger.warning(f"Error loading pair {activation_file.stem}: {e}")
            continue
            
    if len(xs) == 0:
        raise ValueError("No valid data pairs found in test dataset.")

    return np.array(xs), np.array(ys)

# ==========================================
# 3. Helper: Calculate Metrics
# ==========================================
def calculate_metrics(y_true, y_pred, y_probs):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_probs)),
        "pr_auc": float(average_precision_score(y_true, y_probs)) # PR-AUC
    }

# ==========================================
# 4. Main Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset_folder", type=str, required=True, help="Path to the NEW dataset to test on")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the folder containing saved models")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to load. Auto-detected if None.")
    parser.add_argument("--results_folder", type=str, default="../5a_results", help="Where to save the evaluation json")
    
    args = parser.parse_args()
    
    model_folder_dir = pathlib.Path(args.model_folder)
    model_dir = model_folder_dir/"saved_models"
    test_data_dir = pathlib.Path(args.test_dataset_folder)
    results_dir = pathlib.Path(args.results_folder)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # A. Identify Seed
    # ---------------------------------------------------------
    target_seed = args.seed
    if target_seed is None:
        potential_files = list(model_dir.glob("logreg_seed*.joblib"))
        if not potential_files:
            logger.error(f"No 'logreg_seed*.joblib' files found in {model_dir}")
            return
        match = re.search(r"seed(\d+)", potential_files[0].name)
        if match:
            target_seed = int(match.group(1))
            logger.info(f"Auto-detected seed: {target_seed}")
        else:
            logger.error("Could not parse seed. Please specify --seed.")
            return

    # ---------------------------------------------------------
    # B. Load All Preprocessors and Models
    # ---------------------------------------------------------
    try:
        # Paths
        global_scaler_path = model_dir / f"global_scaler_seed{target_seed}.joblib"
        # [Corrected Filename] mlp_internal_scaler
        mlp_scaler_path = model_dir / f"mlp_internal_scaler_seed{target_seed}.joblib"
        pca_path = model_dir / f"pca_seed{target_seed}.joblib"
        logreg_path = model_dir / f"logreg_seed{target_seed}.joblib"
        mlp_path = model_dir / f"mlp_seed{target_seed}.pth"

        # 1. Global Scaler
        if not global_scaler_path.exists():
            raise FileNotFoundError(f"Global Scaler not found: {global_scaler_path}")
        logger.info(f"Loading Global Scaler (Seed {target_seed})...")
        global_scaler = joblib.load(global_scaler_path)

        # 2. PCA
        pca = None
        if pca_path.exists():
            logger.info("Loading PCA...")
            pca = joblib.load(pca_path)
        
        # 3. MLP Scaler
        if not mlp_scaler_path.exists():
            raise FileNotFoundError(f"MLP Internal Scaler not found: {mlp_scaler_path}")
        logger.info("Loading MLP Internal Scaler...")
        mlp_scaler = joblib.load(mlp_scaler_path)

        # 4. Models
        logger.info("Loading LogReg...")
        logreg = joblib.load(logreg_path)
        
        if not mlp_path.exists():
            raise FileNotFoundError(f"MLP weights not found: {mlp_path}")
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return

    # ---------------------------------------------------------
    # C. Load & Preprocess Test Data
    # ---------------------------------------------------------
    logger.info(f"Loading Test Data from {test_data_dir}...")
    X_raw, y_test = load_activations(test_data_dir)
    logger.info(f"Raw Input Shape: {X_raw.shape}")

    # --- Stage 1: Global Scaler ---
    X_stage1 = global_scaler.transform(X_raw)

    # --- Stage 2: PCA (Output used for LogReg) ---
    if pca:
        X_stage2 = pca.transform(X_stage1)
    else:
        X_stage2 = X_stage1

    # --- Stage 3: MLP Scaler (Output used for MLP) ---
    X_stage3 = mlp_scaler.transform(X_stage2)

    # ---------------------------------------------------------
    # D. Initialize & Load MLP
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_stage3.shape[1]
    
    mlp = CustomMLP2Layer(input_size=input_dim).to(device)
    
    try:
        state_dict = torch.load(mlp_path, map_location=device)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        logger.info("MLP weights loaded.")
    except Exception as e:
        logger.error(f"Failed to load MLP state dict: {e}")
        return

    # ---------------------------------------------------------
    # E. Evaluate
    # ---------------------------------------------------------
    results = {
        "source_model": model_folder_dir.name,
        "target_dataset": test_data_dir.name,
        "seed": target_seed,
        "metrics": {}
    }
    
    # 1. LogReg Eval
    # Need probabilities for AUC
    y_probs_lr = logreg.predict_proba(X_stage2)[:, 1] 
    y_pred_lr = logreg.predict(X_stage2)
    
    results["metrics"]["logreg"] = calculate_metrics(y_test, y_pred_lr, y_probs_lr)
    
    # 2. MLP Eval
    X_mlp_tensor = torch.tensor(X_stage3, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs_mlp = mlp(X_mlp_tensor).squeeze().cpu().numpy()
        y_pred_mlp = (probs_mlp > 0.5).astype(int)
        
    results["metrics"]["mlp"] = calculate_metrics(y_test, y_pred_mlp, probs_mlp)
    
    # ---------------------------------------------------------
    # F. Save & Print Results
    # ---------------------------------------------------------
    out_filename = f"{model_folder_dir.name}_ON_{test_data_dir.name}.json"
    out_path = results_dir / out_filename
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.success(f"Results saved to: {out_path}")
    
    # Print formatted summary
    print("\n" + "="*50)
    print(f"TRANSFER EVALUATION SUMMARY (Seed {target_seed})")
    print(f"Model:   {model_folder_dir.name}")
    print(f"Dataset: {test_data_dir.name}")
    print("-" * 50)
    
    # Helper to print row
    def print_row(name, metrics):
        print(f"{name:<10} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")

    print_row("LogReg", results["metrics"]["logreg"])
    print_row("MLP", results["metrics"]["mlp"])
    print("="*50 + "\n")

if __name__ == "__main__":
    main()