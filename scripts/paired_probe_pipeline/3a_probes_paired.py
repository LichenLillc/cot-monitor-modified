"""
Trains simple probes (logistic regression and MLPs) to predict safety alignment outcomes from CoT activations.
Features:
- Nested Funnel Architectures: V1 (Shallow), V2 (Medium), V3 (Deep)
- PCA Mode: Supports training on RAW activations, PCA-reduced activations, or BOTH in a single run.
- Clean isolation of outputs (_raw vs _pca) for strict ablation studies.
"""

import collections
from loguru import logger
import os
import sys
import json
import torch
import argparse
import pathlib
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Sampler
import joblib
import re
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import eval_pred, add_to_final_scores, calculate_metrics_stats, save_probe_outputs_tsv

# -----------------------------------------------------------------------------
# Argument Parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, nargs='+', required=True, 
                    help="input folder(s) containing activations and labels")
parser.add_argument("--N_runs", type=int, default=30, help="number of different seeded runs")
parser.add_argument("--sample_K", type=int, default=-1, help="number of training samples")

# [核心修改] 引入 PCA 模式选择
parser.add_argument("--pca_mode", type=str, choices=['both', 'pca', 'raw'], default='both', 
                    help="Whether to train on RAW data, PCA data, or BOTH (default: both).")
parser.add_argument("--pca_components", type=int, default=128, help="number of PCA components (default: 128)")

### Regularization & Training Hyperparameters
parser.add_argument('--l2', type=float, nargs='?', const=1e-3, default=1e-3,
                    help='L2 regularization strength (weight decay). Default: 1e-3')
parser.add_argument('--dropout', type=float, nargs='?', const=0.3, default=0.3,
                    help='Dropout rate applied to all hidden layers. Default: 0.3')
parser.add_argument('--patience', type=int, nargs='?', const=10, default=5,
                    help='Early stopping patience (epochs). Default: 5')
parser.add_argument('--epoch', type=int, nargs='?', const=5, default=50,
                    help='num of epochs')

### storing test prediction outputs
parser.add_argument("--store_outputs", action="store_true", help="whether to store model outputs")
parser.add_argument("--probe_output_folder", type=str, default="../probe_outputs/", help="folder to store model outputs and results")
parser.add_argument("--save_models", action="store_true", help="whether to save trained models and PCA objects")

args = parser.parse_args()

# Handle multiple input folders for output naming
input_paths = [pathlib.Path(p) for p in args.input_folder]
if len(input_paths) == 1:
    out_dir_name = input_paths[0].name
else:
    folder_names = [p.name for p in input_paths]
    if len(folder_names) > 3:
        out_dir_name = f"Combined_{len(folder_names)}_Datasets"
    else:
        out_dir_name = "+".join(folder_names)
    out_dir_name = re.sub(r'[^\w\-.+]', '_', out_dir_name)

PROBE_OUTPUT_FOLDER = pathlib.Path(args.probe_output_folder) / out_dir_name
PROBE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Auto-Skip Logic (Aware of PCA mode)
# -----------------------------------------------------------------------------
def check_if_already_done():
    summary_path = PROBE_OUTPUT_FOLDER / "training_summary.txt"
    if not summary_path.exists():
        return False
    
    if args.save_models:
        models_dir = PROBE_OUTPUT_FOLDER / "saved_models"
        if not models_dir.exists():
            return False
            
        for seed in range(args.N_runs):
            required_files = [models_dir / f"global_scaler_seed{seed}.joblib"]
            
            if args.pca_mode in ['both', 'raw']:
                required_files.extend([
                    models_dir / f"logreg_raw_seed{seed}.joblib",
                    models_dir / f"mlp_v1_raw_seed{seed}.pth",
                    models_dir / f"mlp_v1_scaler_raw_seed{seed}.joblib",
                    models_dir / f"mlp_v2_raw_seed{seed}.pth",
                    models_dir / f"mlp_v2_scaler_raw_seed{seed}.joblib",
                    models_dir / f"mlp_v3_raw_seed{seed}.pth",
                    models_dir / f"mlp_v3_scaler_raw_seed{seed}.joblib",
                ])
                
            if args.pca_mode in ['both', 'pca']:
                required_files.extend([
                    models_dir / f"pca_model_seed{seed}.joblib",
                    models_dir / f"logreg_pca_seed{seed}.joblib",
                    models_dir / f"mlp_v1_pca_seed{seed}.pth",
                    models_dir / f"mlp_v1_scaler_pca_seed{seed}.joblib",
                    models_dir / f"mlp_v2_pca_seed{seed}.pth",
                    models_dir / f"mlp_v2_scaler_pca_seed{seed}.joblib",
                    models_dir / f"mlp_v3_pca_seed{seed}.pth",
                    models_dir / f"mlp_v3_scaler_pca_seed{seed}.joblib",
                ])
            
            if not all(f.exists() for f in required_files):
                return False
                
    return True

if check_if_already_done():
    logger.success(f"Skipping {out_dir_name}: Summary and requested models already exist.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Data Loading & Preparation (Unchanged)
# -----------------------------------------------------------------------------
def load_data(input_folders):
    activations = dict()
    labels = dict()
    prompts = {}
    cots = {}

    for folder in input_folders:
        folder_path = pathlib.Path(folder)
        dataset_prefix = folder_path.name.replace("_", "-")
        logger.info(f"Loading data from: {folder_path} (Prefix: {dataset_prefix})")

        act_files = list((folder_path / "activations").glob("*.pt"))
        for act_file in tqdm(act_files, desc=f"Activations {folder_path.name}"):
            filename = os.path.basename(act_file)
            original_key = filename.split('.')[0]
            
            parts = original_key.rsplit('_', 1)
            if len(parts) >= 2:
                prompt_id = parts[0]
                cot_id = parts[1]
                new_key = f"{prompt_id}-{dataset_prefix}_{cot_id}"
                
                activation = torch.load(act_file)
                activations[new_key] = activation

        label_files = list((folder_path / "labels").rglob("*.json"))
        for label_file in tqdm(label_files, desc=f"Labels {folder_path.name}"):
            filename = os.path.basename(label_file)
            if filename.endswith("_labeled.json"):
                original_key = filename.replace('_labeled.json', '')
                parts = original_key.rsplit('_', 1)
                
                if len(parts) >= 2:
                    prompt_id = parts[0]
                    cot_id = parts[1]
                    new_key = f"{prompt_id}-{dataset_prefix}_{cot_id}"
                    
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        labels[new_key] = data["safety_label"]["score"]
                        prompts[new_key] = data.get("prompt", "")
                        cots[new_key] = data.get("cot", "")
    
    return activations, prompts, cots, labels

def prepare_data(activations, labels):
    def convert_to_numpy(tensor):
        if isinstance(tensor, np.ndarray): return tensor
        if tensor.dtype == torch.bfloat16: tensor = tensor.to(torch.float32)
        if tensor.dtype in [torch.float16, torch.int8, torch.uint8, torch.int16]: tensor = tensor.to(torch.float32)
        if tensor.requires_grad: tensor = tensor.detach()
        if tensor.device.type != 'cpu': tensor = tensor.cpu()
        return tensor.numpy()

    activations_list = []
    labels_list = []
    prompt_sent_ids = []

    common_keys = set(activations.keys()) & set(labels.keys())
    sorted_keys = sorted(list(common_keys))

    for id in sorted_keys:
        activations_list.append(convert_to_numpy(activations[id]))
        labels_list.append(labels[id]) 
        prompt_sent_ids.append(id)

    X = np.vstack(activations_list)
    labels_np = np.array(labels_list)    
    return X, labels_np, prompt_sent_ids

def apply_pca(X_train, X_val, X_test):
    pca_components = min(args.pca_components, X_train.shape[0], X_train.shape[1])
    logger.info(f"PCA:::reducing to {pca_components}")
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca, pca

class PairedBatchSampler(Sampler):
    def __init__(self, prompt_ids, batch_size):
        self.grouped_indices = collections.defaultdict(list)
        for i, pid in enumerate(prompt_ids):
            base_id = pid.rsplit('_', 1)[0]
            self.grouped_indices[base_id].append(i)
        self.base_ids = sorted(list(self.grouped_indices.keys()))
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.base_ids)
        batch = []
        for base_id in self.base_ids:
            items = self.grouped_indices[base_id]
            if len(batch) + len(items) > self.batch_size and len(batch) > 0:
                yield batch
                batch = []
            batch.extend(items)
        if batch:
            yield batch

    def __len__(self):
        return sum(len(v) for v in self.grouped_indices.values()) // self.batch_size + 1

# -----------------------------------------------------------------------------
# Models (Nested Funnel Architecture)
# -----------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob, model

class MLP_V1_Shallow(nn.Module):
    """1 Hidden Layer: D_in -> 128 -> 1"""
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        
        self.head = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        return self.sigmoid(self.head(x))

class MLP_V2_Medium(nn.Module):
    """2 Hidden Layers: D_in -> 256 -> 128 -> 1"""
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        self.head = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.relu(self.ln1(self.fc1(x))))
        x = self.drop(self.relu(self.ln2(self.fc2(x))))
        return self.sigmoid(self.head(x))

class MLP_V3_Deep(nn.Module):
    """3 Hidden Layers: D_in -> 512 -> 256 -> 128 -> 1"""
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.LayerNorm(128)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)
        
        self.head = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.drop(self.relu(self.ln1(self.fc1(x))))
        x = self.drop(self.relu(self.ln2(self.fc2(x))))
        x = self.drop(self.relu(self.ln3(self.fc3(x))))
        return self.sigmoid(self.head(x))

'''
# [ARCHIVED] DANN (Version 4)
class DomainAdversarialMLP(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, hidden_size3=128, dropout_rate=0.4, noise_std=0):
        super(DomainAdversarialMLP, self).__init__()
        # ... logic preserved but inactive ...
'''

# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------
def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, train_ids=None, version=2):
    # Local Scaling for fast Neural Net convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = 64
    if train_ids is not None:
        sampler = PairedBatchSampler(train_ids, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    input_size = X_train.shape[1]
    
    # Model Selection
    if version == 1:
        model = MLP_V1_Shallow(input_size, dropout_rate=args.dropout)
    elif version == 2:
        model = MLP_V2_Medium(input_size, dropout_rate=args.dropout)
    elif version == 3:
        model = MLP_V3_Deep(input_size, dropout_rate=args.dropout)
    else:
        raise ValueError(f"Unknown MLP version: {version}")
    
    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    if num_pos < num_neg:
        pos_weight = torch.FloatTensor([num_neg / num_pos])
        minority_class = 1
    else:
        pos_weight = torch.FloatTensor([num_pos / num_neg])
        minority_class = 0
        
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.l2)
    
    num_epochs = args.epoch
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            weights = torch.ones_like(labels)
            weights[labels == minority_class] = pos_weight
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        y_pred_binary = (y_pred >= 0.5).astype(int)
        val_f1 = f1_score(y_true, y_pred_binary, average="binary")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred_prob = y_pred_tensor.cpu().numpy().flatten()
        
    y_pred = (y_pred_prob >= 0.5).astype(int)
    return y_pred, y_pred_prob, model, scaler


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    logger.info("====== [DEBUG] 3a_probes_paired.py successfully started! ======")
    activations_dict, prompts_dict, cots_dict, labels_dict = load_data(input_paths)
    
    prompt_IDs = set([x.rsplit("_", 1)[0] for x in activations_dict.keys()])
    N = len(prompt_IDs)
    logger.debug(f"Loaded {len(activations_dict)} activations. Total unique Prompts: {N}")
    
    if N == 0:
        logger.error("[FATAL ERROR] Found 0 activations! Aborting.")
        sys.exit(1)
    
    logger.info(f"Configuration -> Mode: {args.pca_mode.upper()}, Dropout: {args.dropout}, L2: {args.l2}, Patience: {args.patience}")
    logger.info(f"Output Folder: {PROBE_OUTPUT_FOLDER}")

    # Track metrics for both modes separately
    score_dicts = {}
    for mode in ['raw', 'pca']:
        for model_name in ['logreg', 'mlp_v1', 'mlp_v2', 'mlp_v3']:
            score_dicts[f"{model_name}_{mode}"] = collections.defaultdict(list)
            
    # Baselines
    score_dicts['empirical_random'] = collections.defaultdict(list)
    score_dicts['theoretical_random'] = collections.defaultdict(list)
    score_dicts['always_ones'] = collections.defaultdict(list)
    score_dicts['always_zeros'] = collections.defaultdict(list)
    
    seed_results = []

    for seed in range(args.N_runs):
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        train_prompt_ids = set(np.random.choice(sorted(list(prompt_IDs)), int(0.7 * N), replace=False))
        test_prompt_ids = prompt_IDs - train_prompt_ids
        
        train_prompt_ids_list = list(train_prompt_ids)
        np.random.shuffle(train_prompt_ids_list)
        split_idx = int(0.9 * len(train_prompt_ids_list))
        train_prompt_ids = set(train_prompt_ids_list[:split_idx])
        val_prompt_ids = set(train_prompt_ids_list[split_idx:])
        
        X, labels_np, prompt_sent_ids  = prepare_data(activations_dict, labels_dict)
        
        train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.rsplit('_', 1)[0] in train_prompt_ids]
        val_indices = [i for i, key in enumerate(prompt_sent_ids) if key.rsplit('_', 1)[0] in val_prompt_ids]
        test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.rsplit('_', 1)[0] in test_prompt_ids]
        
        if args.sample_K > 0:
            np.random.shuffle(train_indices)
            train_indices = train_indices[:args.sample_K]

        X_train_raw = X[train_indices]
        X_val_raw = X[val_indices]
        X_test_raw = X[test_indices]

        # 1. Global Pre-Scaling
        global_scaler = StandardScaler()
        X_train_scaled = global_scaler.fit_transform(X_train_raw)
        X_val_scaled = global_scaler.transform(X_val_raw)
        X_test_scaled = global_scaler.transform(X_test_raw)

        threshold = 0.5
        y_train = (labels_np[train_indices] >= threshold).astype(int) 
        y_val = (labels_np[val_indices] >= threshold).astype(int)
        y_test = (labels_np[test_indices] >= threshold).astype(int)

        keys_train = [prompt_sent_ids[i] for i in train_indices]
        keys_test = [prompt_sent_ids[i] for i in test_indices]

        if args.save_models:
            models_dir = PROBE_OUTPUT_FOLDER / "saved_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(global_scaler, models_dir / f"global_scaler_seed{seed}.joblib")
            
        seed_record = {'seed': seed}
        test_text_prompts = [prompts_dict[key] for key in keys_test]
        test_text_cots = [cots_dict[key] for key in keys_test]
        
        # =====================================================================
        # 轨道 1: RAW Data Training
        # =====================================================================
        if args.pca_mode in ['both', 'raw']:
            logger.info(f"Seed {seed} | Starting RAW Models Training (Input Dim: {X_train_scaled.shape[1]})")
            
            # LogReg
            lr_y_pred, lr_y_prob, lr_model = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
            lr_eval = eval_pred(y_test, lr_y_pred, lr_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            # MLPs
            v1_y_pred, v1_y_prob, v1_model, v1_scl = train_mlp(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, keys_train, 1)
            v1_eval = eval_pred(y_test, v1_y_pred, v1_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            v2_y_pred, v2_y_prob, v2_model, v2_scl = train_mlp(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, keys_train, 2)
            v2_eval = eval_pred(y_test, v2_y_pred, v2_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            v3_y_pred, v3_y_prob, v3_model, v3_scl = train_mlp(X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, keys_train, 3)
            v3_eval = eval_pred(y_test, v3_y_pred, v3_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            seed_record['logreg_raw'] = lr_eval
            seed_record['mlp_v1_raw'] = v1_eval
            seed_record['mlp_v2_raw'] = v2_eval
            seed_record['mlp_v3_raw'] = v3_eval
            
            add_to_final_scores(lr_eval, score_dicts['logreg_raw'], 'logreg_raw')
            add_to_final_scores(v1_eval, score_dicts['mlp_v1_raw'], 'mlp_v1_raw')
            add_to_final_scores(v2_eval, score_dicts['mlp_v2_raw'], 'mlp_v2_raw')
            add_to_final_scores(v3_eval, score_dicts['mlp_v3_raw'], 'mlp_v3_raw')
            
            if args.save_models:
                joblib.dump(lr_model, models_dir / f"logreg_raw_seed{seed}.joblib")
                torch.save(v1_model.state_dict(), models_dir / f"mlp_v1_raw_seed{seed}.pth")
                joblib.dump(v1_scl, models_dir / f"mlp_v1_scaler_raw_seed{seed}.joblib")
                torch.save(v2_model.state_dict(), models_dir / f"mlp_v2_raw_seed{seed}.pth")
                joblib.dump(v2_scl, models_dir / f"mlp_v2_scaler_raw_seed{seed}.joblib")
                torch.save(v3_model.state_dict(), models_dir / f"mlp_v3_raw_seed{seed}.pth")
                joblib.dump(v3_scl, models_dir / f"mlp_v3_scaler_raw_seed{seed}.joblib")
                
            if args.store_outputs:
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"logreg_raw_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, lr_y_pred, lr_y_prob)
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_v1_raw_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, v1_y_pred, v1_y_prob)
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_v2_raw_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, v2_y_pred, v2_y_prob)
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_v3_raw_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, v3_y_pred, v3_y_prob)

        # =====================================================================
        # 轨道 2: PCA Data Training
        # =====================================================================
        if args.pca_mode in ['both', 'pca']:
            X_train_pca, X_val_pca, X_test_pca, pca_model = apply_pca(X_train_scaled, X_val_scaled, X_test_scaled)
            logger.info(f"Seed {seed} | Starting PCA Models Training (Input Dim: {X_train_pca.shape[1]})")
            
            # LogReg
            lr_y_pred, lr_y_prob, lr_model = train_logistic_regression(X_train_pca, y_train, X_test_pca, y_test)
            lr_eval = eval_pred(y_test, lr_y_pred, lr_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            # MLPs
            v1_y_pred, v1_y_prob, v1_model, v1_scl = train_mlp(X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, keys_train, 1)
            v1_eval = eval_pred(y_test, v1_y_pred, v1_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            v2_y_pred, v2_y_prob, v2_model, v2_scl = train_mlp(X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, keys_train, 2)
            v2_eval = eval_pred(y_test, v2_y_pred, v2_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            v3_y_pred, v3_y_prob, v3_model, v3_scl = train_mlp(X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, keys_train, 3)
            v3_eval = eval_pred(y_test, v3_y_pred, v3_y_prob, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
            
            seed_record['logreg_pca'] = lr_eval
            seed_record['mlp_v1_pca'] = v1_eval
            seed_record['mlp_v2_pca'] = v2_eval
            seed_record['mlp_v3_pca'] = v3_eval
            
            add_to_final_scores(lr_eval, score_dicts['logreg_pca'], 'logreg_pca')
            add_to_final_scores(v1_eval, score_dicts['mlp_v1_pca'], 'mlp_v1_pca')
            add_to_final_scores(v2_eval, score_dicts['mlp_v2_pca'], 'mlp_v2_pca')
            add_to_final_scores(v3_eval, score_dicts['mlp_v3_pca'], 'mlp_v3_pca')
            
            if args.save_models:
                joblib.dump(pca_model, models_dir / f"pca_model_seed{seed}.joblib")
                joblib.dump(lr_model, models_dir / f"logreg_pca_seed{seed}.joblib")
                torch.save(v1_model.state_dict(), models_dir / f"mlp_v1_pca_seed{seed}.pth")
                joblib.dump(v1_scl, models_dir / f"mlp_v1_scaler_pca_seed{seed}.joblib")
                torch.save(v2_model.state_dict(), models_dir / f"mlp_v2_pca_seed{seed}.pth")
                joblib.dump(v2_scl, models_dir / f"mlp_v2_scaler_pca_seed{seed}.joblib")
                torch.save(v3_model.state_dict(), models_dir / f"mlp_v3_pca_seed{seed}.pth")
                joblib.dump(v3_scl, models_dir / f"mlp_v3_scaler_pca_seed{seed}.joblib")
                
            if args.store_outputs:
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"logreg_pca_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, lr_y_pred, lr_y_prob)
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_v1_pca_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, v1_y_pred, v1_y_prob)
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_v2_pca_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, v2_y_pred, v2_y_prob)
                save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_v3_pca_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, v3_y_pred, v3_y_prob)

        seed_results.append(seed_record)

        # Baseline evaluations (Random / All Ones / All Zeros)
        positive_prior = np.sum(y_train == 1) / len(y_train)
        random_probs = np.random.uniform(0, 1, size=len(X_test_raw))
        random_y_pred = (random_probs < positive_prior).astype(int)
        
        random_eval = eval_pred(y_test, random_y_pred, random_probs, metrics=["f1", "accuracy", "pr_auc", "auc_roc"])
        theory_random_eval = {"f1": positive_prior, "pr_auc": positive_prior, "auc_roc": 0.5} 
        
        always_ones_eval = eval_pred(y_test, np.ones(len(y_test)), metrics=["f1", "accuracy"])
        always_ones_eval.update({"pr_auc": positive_prior, "auc_roc": 0.5})
        
        always_zeros_eval = eval_pred(y_test, np.zeros(len(y_test)), metrics=["f1", "accuracy"])
        always_zeros_eval.update({"pr_auc": 0, "auc_roc": 0.5})
        
        add_to_final_scores(random_eval, score_dicts['empirical_random'], 'empirical_random')
        add_to_final_scores(theory_random_eval, score_dicts['theoretical_random'], 'theoretical_random')
        add_to_final_scores(always_ones_eval, score_dicts['always_ones'], "always_ones")
        add_to_final_scores(always_zeros_eval, score_dicts['always_zeros'], "always_zeros")


    # -----------------------------------------------------------------------------
    # Summary & Printing
    # -----------------------------------------------------------------------------
    output_lines = []
    def record(text):
        print(text)
        output_lines.append(str(text))

    record("\n" + "="*85)
    record(f"PER-SEED PERFORMANCE SUMMARY ({args.pca_mode.upper()} MODE)")
    record("="*85)
    record(f"{'Seed':<5} | {'Model_Mode':<13} | {'F1':<8} | {'Acc':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}")
    record("-" * 85)
    
    for res in seed_results:
        s = res['seed']
        for mode in ['raw', 'pca']:
            if args.pca_mode in ['both', mode]:
                lr = res[f'logreg_{mode}']
                record(f"{s:<5} | {'LogReg_'+mode.upper():<13} | {lr['f1']:.4f}   | {lr['accuracy']:.4f}   | {lr['pr_auc']:.4f}   | {lr['auc_roc']:.4f}")
                m1 = res[f'mlp_v1_{mode}']
                record(f"{'':<5} | {'MLP_V1_'+mode.upper():<13} | {m1['f1']:.4f}   | {m1['accuracy']:.4f}   | {m1['pr_auc']:.4f}   | {m1['auc_roc']:.4f}")
                m2 = res[f'mlp_v2_{mode}']
                record(f"{'':<5} | {'MLP_V2_'+mode.upper():<13} | {m2['f1']:.4f}   | {m2['accuracy']:.4f}   | {m2['pr_auc']:.4f}   | {m2['auc_roc']:.4f}")
                m3 = res[f'mlp_v3_{mode}']
                record(f"{'':<5} | {'MLP_V3_'+mode.upper():<13} | {m3['f1']:.4f}   | {m3['accuracy']:.4f}   | {m3['pr_auc']:.4f}   | {m3['auc_roc']:.4f}")
        record("-" * 85)
    record("="*85 + "\n")

    # Aggregate active dictionaries
    active_dicts = [d for d in score_dicts.values() if len(d) > 0]
    stats_output = calculate_metrics_stats(active_dicts)
    record(stats_output)

    summary_path = PROBE_OUTPUT_FOLDER / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(output_lines))
    
    logger.info(f"Full training summary saved to: {summary_path}")

if __name__ == "__main__":
    main()