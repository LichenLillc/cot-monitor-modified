"""
Trains simple probes (logistic regression and MLPs) to predict safety alignment outcomes from CoT activations.
Now includes adjustable Regularization (L2, Dropout, Noise) and Early Stopping via CLI arguments.
Now supports merging multiple input folders for training.
Now supports dynamic MLP architecture selection (v1, v2, v3).
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
# Argument Parsing with Smart Defaults
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, nargs='+', required=True, 
                    help="input folder(s) containing activations and labels")
parser.add_argument("--N_runs", type=int, default=5, help="number of different seeded runs")
parser.add_argument("--sample_K", type=int, default=-1, help="number of training samples")
parser.add_argument("--pca", action="store_true", help="run PCA")
parser.add_argument("--pca_components", type=int, default=50, help="number of PCA components")

# [新增] 模型架构版本选择参数
parser.add_argument("--version", "-v", type=int, choices=[1, 2, 3], default=2,
                    help="MLP architecture version: 1 (CustomMLP), 2 (RobustMLP), 3 (DANN)")

### Regularization & Training Hyperparameters
parser.add_argument('--l2', type=float, nargs='?', const=1e-3, default=0.0,
                    help='L2 regularization strength (weight decay). Default if flag used: 1e-3')
parser.add_argument('--dropout', type=float, nargs='?', const=0.3, default=0.0,
                    help='Dropout rate. Default if flag used: 0.3')
parser.add_argument('--noise', type=float, nargs='?', const=0.05, default=0.0,
                    help='Gaussian noise std dev added to inputs. Default if flag used: 0.05')
parser.add_argument('--patience', type=int, nargs='?', const=5, default=3,
                    help='Early stopping patience (epochs). Default if flag used: 5. Default if no flag: 2')
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
# Auto-Skip Logic
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
            required_files = [
                models_dir / f"logreg_seed{seed}.joblib",
                models_dir / f"mlp_seed{seed}.pth",
                models_dir / f"global_scaler_seed{seed}.joblib",
                models_dir / f"mlp_internal_scaler_seed{seed}.joblib"
            ]
            if args.pca:
                required_files.append(models_dir / f"pca_seed{seed}.joblib")
            
            if not all(f.exists() for f in required_files):
                return False
                
    return True

if check_if_already_done():
    logger.success(f"Skipping {out_dir_name}: Summary and {args.N_runs} model runs already exist.")
    sys.exit(0)

# -----------------------------------------------------------------------------
# Data Loading & Preparation
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
        if isinstance(tensor, np.ndarray):
            return tensor
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float32)
        if tensor.dtype in [torch.float16, torch.int8, torch.uint8, torch.int16]:
            tensor = tensor.to(torch.float32)
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        return tensor.numpy()

    activations_list = []
    labels_list = []
    prompt_sent_ids = []

    common_keys = set(activations.keys()) & set(labels.keys())
    if len(common_keys) != len(activations):
        logger.warning(f"Mismatch: {len(activations)} activations, {len(labels)} labels. Using intersection {len(common_keys)}.")
    
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
# Models
# -----------------------------------------------------------------------------
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob, model

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
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x) 
        x = self.sigmoid(x)
        return x

class RobustMLP(nn.Module):
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, hidden_size3=128, dropout_rate=0.4, noise_std=0):
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
    def __init__(self, input_size, hidden_size1=512, hidden_size2=256, hidden_size3=128, dropout_rate=0.4, noise_std=0):
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

# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------
def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, train_ids=None):
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
    
    # [修改] 根据 version 动态选择 V1 或 V2 模型
    if args.version == 1:
        model = CustomMLP2Layer(input_size, dropout_rate=args.dropout, noise_std=args.noise)
    else:
        model = RobustMLP(input_size, dropout_rate=args.dropout, noise_std=args.noise)
    
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
    
    patience = args.patience
    patience_counter = 0
    
    print(f"\n--- Starting Standard MLP Training (Version {args.version}) ---")
    
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
            
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (Patience: {patience})")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred_prob = y_pred_tensor.cpu().numpy().flatten()
        
    y_pred = (y_pred_prob >= 0.5).astype(int)
    return y_pred, y_pred_prob, model, scaler


def train_dann_mlp(X_train, y_train, X_val, y_val, X_test, y_test, keys_train, keys_val, keys_test, log_path=None):
    scaler = StandardScaler()
    X_train_tensor = torch.FloatTensor(scaler.fit_transform(X_train))
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
    X_val_tensor = torch.FloatTensor(scaler.transform(X_val))
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    X_test_tensor = torch.FloatTensor(scaler.transform(X_test))
    
    def get_domain_labels(keys):
        return np.array([1.0 if 'taco' in k.lower() else 0.0 for k in keys])
        
    d_train_tensor = torch.FloatTensor(get_domain_labels(keys_train)).unsqueeze(1)
    d_val_tensor = torch.FloatTensor(get_domain_labels(keys_val)).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, d_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor, d_val_tensor)
    
    batch_size = 64
    sampler = PairedBatchSampler(keys_train, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = DomainAdversarialMLP(input_size=X_train.shape[1], dropout_rate=args.dropout, noise_std=args.noise)
    
    opt_domain = optim.Adam(
        list(model.domain_fc.parameters()) + list(model.domain_ln.parameters()) + list(model.domain_head.parameters()), 
        lr=0.001, weight_decay=args.l2
    )
    opt_shared_task = optim.Adam(
        list(model.shared1.parameters()) + list(model.ln1.parameters()) +
        list(model.shared2.parameters()) + list(model.ln2.parameters()) +
        list(model.task_fc.parameters()) + list(model.task_ln.parameters()) + list(model.task_head.parameters()), 
        lr=0.001, weight_decay=args.l2
    )
    
    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    pos_weight = torch.FloatTensor([num_neg / num_pos]) if num_pos < num_neg else torch.FloatTensor([num_pos / num_neg])
    minority_class = 1 if num_pos < num_neg else 0
    
    criterion_bce = nn.BCELoss(reduction='none')
    criterion_domain = nn.BCELoss()
    
    num_epochs = 50
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    lambda_entropy_max = 1.0 
    
    print(f"\n--- Starting DANN Training (Version 3) ---")
    
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 73 + "\n")
            f.write(f"{'Epoch':<7} | {'Task Val F1':<13} | {'Dom Acc':<9} | {'Dom Entropy':<13} | {'Lambda':<8}\n")
            f.write("-" * 73 + "\n")
    
    for epoch in range(num_epochs):
        model.train()
        
        p = float(epoch) / num_epochs
        current_lambda = lambda_entropy_max * (2. / (1. + np.exp(-10. * p)) - 1)
        
        epoch_dom_loss = 0.0
        dom_correct = 0
        total_samples = 0
        epoch_entropy = 0.0
        
        for inputs, t_labels, d_labels in train_loader:
            b_size = inputs.size(0)
            total_samples += b_size
            
            feat_detached = model.extract_features(inputs).detach() 
            d_pred = model.predict_domain(feat_detached)
            loss_domain = criterion_domain(d_pred, d_labels)
            
            opt_domain.zero_grad()
            loss_domain.backward()
            opt_domain.step()
            
            epoch_dom_loss += loss_domain.item() * b_size
            dom_correct += ((d_pred >= 0.5).float() == d_labels).sum().item()
            
            feat = model.extract_features(inputs)
            t_pred = model.predict_task(feat)
            d_pred_for_shared = model.predict_domain(feat) 
            
            loss_task = criterion_bce(t_pred, t_labels)
            weights = torch.ones_like(t_labels)
            weights[t_labels == minority_class] = pos_weight
            loss_task = (loss_task * weights).mean()
            
            loss_entropy = torch.mean((d_pred_for_shared - 0.5)**2)
            loss_shared_task = loss_task + current_lambda * loss_entropy
            
            opt_shared_task.zero_grad()
            loss_shared_task.backward()
            opt_shared_task.step()
            
            p_det = torch.clamp(d_pred_for_shared.detach(), 1e-7, 1.0 - 1e-7)
            batch_entropy = - (p_det * torch.log2(p_det) + (1.0 - p_det) * torch.log2(1.0 - p_det)).mean().item()
            epoch_entropy += batch_entropy * b_size

        avg_dom_acc = dom_correct / total_samples
        avg_entropy = epoch_entropy / total_samples

        model.eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for inputs, t_labels, _ in val_loader:
                feat = model.extract_features(inputs)
                outputs = model.predict_task(feat)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(t_labels.cpu().numpy())
                
        y_pred = np.array(y_pred).flatten()
        y_pred_binary = (y_pred >= 0.5).astype(int)
        val_f1 = f1_score(np.array(y_true).flatten(), y_pred_binary, average="binary")
        
        logger.info(f"Ep {epoch+1:02d} | Task Val F1: {val_f1:.4f} | Dom Acc: {avg_dom_acc:.4f} | Dom Entropy: {avg_entropy:.4f} | λ: {current_lambda:.3f}")
        
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1:<7} | {val_f1:<13.4f} | {avg_dom_acc:<9.4f} | {avg_entropy:<13.4f} | {current_lambda:<8.4f}\n")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            msg = f"Early stopping at epoch {epoch+1} (Patience: {args.patience})"
            logger.info(msg)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write("-" * 73 + "\n")
                    f.write(msg + "\n")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        feat_test = model.extract_features(X_test_tensor)
        y_pred_tensor = model.predict_task(feat_test)
        y_pred_prob = y_pred_tensor.cpu().numpy().flatten()
        
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    return y_pred, y_pred_prob, model, scaler

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    print("====== [DEBUG] 3a_probes_paired.py successfully started! ======", flush=True)
    logger.info("====== [DEBUG] 3a_probes_paired.py successfully started! ======")
    print(f"====== [DEBUG] Loading data from {input_paths} ======", flush=True)
    
    activations_dict, prompts_dict, cots_dict, labels_dict = load_data(input_paths)
    
    prompt_IDs = set([x.rsplit("_", 1)[0] for x in activations_dict.keys()])
    N = len(prompt_IDs)
    
    print(f"====== [DEBUG] Data loaded! Unique Prompt-Dataset pairs: {N} (Total activations: {len(activations_dict)}) ======", flush=True)
    logger.debug(f"Loaded {len(activations_dict)=} activations. Total unique Prompt-Dataset pairs: {N}")
    
    if N == 0:
        print("🚨🚨🚨 [FATAL ERROR] Found 0 activations! Check your input paths or file naming. Aborting 3a script.", flush=True)
        logger.error("[FATAL ERROR] Found 0 activations! Aborting.")
        sys.exit(1)
    
    logger.info(f"Configuration -> L2: {args.l2}, Dropout: {args.dropout}, Noise: {args.noise}, Patience: {args.patience}, Version: {args.version}")
    logger.info(f"Output Folder: {PROBE_OUTPUT_FOLDER}")

    D_final_logreg_scores = collections.defaultdict(list)
    D_final_mlp_scores = collections.defaultdict(list)
    D_final_rand_lr_scores = collections.defaultdict(list)
    D_final_random_scores = collections.defaultdict(list)
    D_final_theoretical_random_scores = collections.defaultdict(list)
    D_final_always_ones_scores = collections.defaultdict(list)
    D_final_always_zeros_scores = collections.defaultdict(list)
    total_disagreement_percentage = 0
    
    seed_results = []

    for seed in range(args.N_runs):
        np.random.seed(seed)
        torch.manual_seed(seed) 
        
        train_prompt_ids = set(np.random.choice(sorted(list(prompt_IDs)), int(0.7 * N), replace=False))
        test_prompt_ids = prompt_IDs - train_prompt_ids
        print(f"Seed {seed}: Selected {len(train_prompt_ids)} prompts for training and {len(test_prompt_ids)} for testing")
        
        train_prompt_ids_list = list(train_prompt_ids)
        np.random.shuffle(train_prompt_ids_list)
        split_idx = int(0.9 * len(train_prompt_ids_list))
        train_prompt_ids = set(train_prompt_ids_list[:split_idx])
        val_prompt_ids = set(train_prompt_ids_list[split_idx:])
        
        X, labels_np, prompt_sent_ids  = prepare_data(activations_dict, labels_dict)
        
        train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.rsplit('_', 1)[0] in train_prompt_ids]
        val_indices = [i for i, key in enumerate(prompt_sent_ids) if key.rsplit('_', 1)[0] in val_prompt_ids]
        test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.rsplit('_', 1)[0] in test_prompt_ids]
        
        if args.sample_K and args.sample_K > 0:
            assert len(train_indices) >= args.sample_K, f"Not enough training samples. Required: {args.sample_K}, Available: {len(train_indices)}"
            np.random.shuffle(train_indices)
            train_indices = train_indices[:args.sample_K]
            logger.info(f">>>> use {args.sample_K} data")
        else:
            logger.info("use all data")

        X_train = X[train_indices]
        X_val = X[val_indices]
        X_test = X[test_indices]

        global_scaler = StandardScaler()
        X_train = global_scaler.fit_transform(X_train)
        X_val = global_scaler.transform(X_val)
        X_test = global_scaler.transform(X_test)

        threshold = 0.5
        y_train = (labels_np[train_indices] >= threshold).astype(int) 
        y_val = (labels_np[val_indices] >= threshold).astype(int)
        y_test = (labels_np[test_indices] >= threshold).astype(int)

        keys_test = [prompt_sent_ids[i] for i in test_indices]

        pca_model = None
        if args.pca:
            X_train, X_val, X_test, pca_model = apply_pca(X_train, X_val, X_test)
        
        keys_train = [prompt_sent_ids[i] for i in train_indices]
        keys_val = [prompt_sent_ids[i] for i in val_indices]
        
        logreg_y_pred, logreg_y_pred_prob, logreg_model = train_logistic_regression(X_train, y_train, X_test, y_test)
        
        # [修改] 动态根据 version 选择调用的模型训练器
        if args.version == 3:
            curves_dir = PROBE_OUTPUT_FOLDER / "training_curves"
            curves_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = curves_dir / f"dann_metrics_seed{seed}.txt"
            
            mlp_y_pred, mlp_y_pred_prob, mlp_model, mlp_internal_scaler = train_dann_mlp(
                X_train, y_train, X_val, y_val, X_test, y_test, 
                keys_train=keys_train, keys_val=keys_val, keys_test=keys_test,
                log_path=log_file_path
            )
        else:
            mlp_y_pred, mlp_y_pred_prob, mlp_model, mlp_internal_scaler = train_mlp(
                X_train, y_train, X_val, y_val, X_test, y_test, 
                train_ids=keys_train
            )

        np.random.seed(seed)
        positive_prior = np.sum(y_train == 1)/len(y_train)
        random_probs = np.random.uniform(0, 1, size=len(X_test))
        random_y_pred = (random_probs < positive_prior).astype(int)
        always_ones_pred = np.ones(len(X_test))
        always_zeros_pred = np.zeros(len(X_test))

        np.random.seed(seed)
        shuffled_indices = np.arange(len(y_train))
        np.random.shuffle(shuffled_indices)
        y_train_shuffled = y_train[shuffled_indices]
        
        disagreement = np.sum(y_train != y_train_shuffled)
        disagreement_percentage = (disagreement / len(y_train)) * 100
        total_disagreement_percentage += disagreement_percentage
        rand_lr_pred, rand_lr_pred_prob, _ = train_logistic_regression(X_train, y_train_shuffled, X_test, y_test)

        eval_metrics = ["f1", "accuracy", "pr_auc", "auc_roc"]
        logreg_eval = eval_pred(y_test, logreg_y_pred, logreg_y_pred_prob, metrics=eval_metrics)
        mlp_eval = eval_pred(y_test, mlp_y_pred, mlp_y_pred_prob, metrics=eval_metrics)
        rand_lr_eval = eval_pred(y_test, rand_lr_pred, rand_lr_pred_prob, metrics=eval_metrics)
        random_eval = eval_pred(y_test, random_y_pred, random_probs, metrics=eval_metrics)
        theory_random_eval = {"f1": positive_prior, "pr_auc": positive_prior, "auc_roc": 0.5} 
        always_ones_eval = eval_pred(y_test, always_ones_pred, metrics=["f1", "accuracy"])
        always_ones_eval["pr_auc"] = positive_prior 
        always_ones_eval["auc_roc"] = 0.5
        always_zeros_eval = eval_pred(y_test, always_zeros_pred, metrics=["f1", "accuracy"])
        always_zeros_eval["pr_auc"] = 0 
        always_zeros_eval["auc_roc"] = 0.5

        seed_results.append({
            "seed": seed,
            "logreg": logreg_eval,
            "mlp": mlp_eval
        })

        if args.save_models:
            models_dir = PROBE_OUTPUT_FOLDER / "saved_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(logreg_model, models_dir / f"logreg_seed{seed}.joblib")
            torch.save(mlp_model.state_dict(), models_dir / f"mlp_seed{seed}.pth")
            joblib.dump(global_scaler, models_dir / f"global_scaler_seed{seed}.joblib")
            joblib.dump(mlp_internal_scaler, models_dir / f"mlp_internal_scaler_seed{seed}.joblib")
            if pca_model is not None:
                joblib.dump(pca_model, models_dir / f"pca_seed{seed}.joblib")
            logger.info(f"Models and Scalers saved to {models_dir} (seed {seed})")

        add_to_final_scores(logreg_eval, D_final_logreg_scores, 'logreg')
        add_to_final_scores(mlp_eval, D_final_mlp_scores, 'mlp')
        add_to_final_scores(rand_lr_eval, D_final_rand_lr_scores, 'random_logreg')
        add_to_final_scores(random_eval, D_final_random_scores, 'empirical_random')
        add_to_final_scores(theory_random_eval, D_final_theoretical_random_scores, 'theoretical_random')
        add_to_final_scores(always_ones_eval, D_final_always_ones_scores, "always_ones")
        add_to_final_scores(always_zeros_eval, D_final_always_zeros_scores, "always_zeros")
        
        if args.store_outputs:
            test_text_prompts = [prompts_dict[key] for key in keys_test]
            test_text_cots = [cots_dict[key] for key in keys_test]
            save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"logreg_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, logreg_y_pred, logreg_y_pred_prob)
            save_probe_outputs_tsv(PROBE_OUTPUT_FOLDER, f"mlp_seed{seed}", keys_test, test_text_prompts, test_text_cots, y_test, mlp_y_pred, mlp_y_pred_prob)


    output_lines = []

    def record(text):
        print(text)
        output_lines.append(str(text))

    record(f"---- mean disagreement after shuffling: {total_disagreement_percentage/args.N_runs:.2f}%")
    record("\n" + "="*85)
    record(f"PER-SEED PERFORMANCE SUMMARY")
    record("="*85)
    record(f"{'Seed':<5} | {'Model':<8} | {'F1':<8} | {'Acc':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}")
    record("-" * 85)
    
    for res in seed_results:
        s = res['seed']
        lr = res['logreg']
        record(f"{s:<5} | {'LogReg':<8} | {lr['f1']:.4f}   | {lr['accuracy']:.4f}   | {lr['pr_auc']:.4f}   | {lr['auc_roc']:.4f}")
        mlp = res['mlp']
        record(f"{'':<5} | {'MLP':<8} | {mlp['f1']:.4f}   | {mlp['accuracy']:.4f}   | {mlp['pr_auc']:.4f}   | {mlp['auc_roc']:.4f}")
        record("-" * 85)
    record("="*85 + "\n")

    stats_output = calculate_metrics_stats([
        D_final_logreg_scores,
        D_final_mlp_scores,
        D_final_rand_lr_scores,
        D_final_random_scores,
        D_final_theoretical_random_scores,
        D_final_always_ones_scores,
        D_final_always_zeros_scores
    ])
    record(stats_output)

    summary_path = PROBE_OUTPUT_FOLDER / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(output_lines))
    
    logger.info(f"Full training summary saved to: {summary_path}")

if __name__ == "__main__":
    main()