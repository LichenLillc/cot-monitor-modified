"""
Trains simple probes (logistic regression and MLPs) to predict safety alignment outcomes from CoT activations.
Now includes adjustable Regularization (L2, Dropout, Noise) and Early Stopping via CLI arguments.
"""

import collections
from loguru import logger
import os
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
from torch.utils.data import TensorDataset, DataLoader
import joblib

from utils import eval_pred, add_to_final_scores, calculate_metrics_stats, save_probe_outputs_tsv

# -----------------------------------------------------------------------------
# Argument Parsing with Smart Defaults
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, required=True, help="input folder containing activations and labels")
parser.add_argument("--N_runs", type=int, default=1, help="number of different seeded runs")
parser.add_argument("--sample_K", type=int, default=-1, help="number of training samples")
parser.add_argument("--pca", action="store_true", help="run PCA")
parser.add_argument("--pca_components", type=int, default=50, help="number of PCA components")

### Regularization & Training Hyperparameters
# logic: 
# 1. flag not set -> value is default (usually 0/off)
# 2. flag set without value -> value is const (recommended for small data)
# 3. flag set with value -> value is user provided

# L2 Regularization (Weight Decay)
parser.add_argument('--l2', type=float, nargs='?', const=1e-3, default=0.0,
                    help='L2 regularization strength (weight decay). Default if flag used: 1e-3')

# Dropout
parser.add_argument('--dropout', type=float, nargs='?', const=0.3, default=0.0,
                    help='Dropout rate. Default if flag used: 0.3')

# Gaussian Noise
parser.add_argument('--noise', type=float, nargs='?', const=0.05, default=0.0,
                    help='Gaussian noise std dev added to inputs. Default if flag used: 0.05')

# Early Stopping Patience
parser.add_argument('--patience', type=int, nargs='?', const=5, default=3,
                    help='Early stopping patience (epochs). Default if flag used: 5. Default if no flag: 2')

### storing test prediction outputs
parser.add_argument("--store_outputs", action="store_true", help="whether to store model outputs")
parser.add_argument("--probe_output_folder", type=str, default="../probe_outputs/", help="folder to store model outputs and results")
parser.add_argument("--save_models", action="store_true", help="whether to save trained models and PCA objects")

args = parser.parse_args()
INPUT_FOLDER = pathlib.Path(args.input_folder)
PROBE_OUTPUT_FOLDER = pathlib.Path(args.probe_output_folder) / INPUT_FOLDER.name
PROBE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Loading & Preparation
# -----------------------------------------------------------------------------
def load_data():
    """
    Load activations and labels from the input folder.
    """
    activations = dict()
    labels = dict()
    prompts = {}
    cots = {}

    for act_file in tqdm((INPUT_FOLDER / "activations").glob("*.pt"), desc="Loading activations"):
        filename = os.path.basename(act_file)
        key = filename.split('.')[0]
        activation = torch.load(act_file)
        activations[key] = activation

    for label_file in tqdm((INPUT_FOLDER / "labels").rglob("*.json"), desc="Loading labels and texts"):
        filename = os.path.basename(label_file)
        if filename.endswith("_labeled.json"):
            key = filename.split('.')[0].split('_')[0:2]
            key = '_'.join(key)
            with open(label_file, 'r') as f:
                data = json.load(f)
                labels[key] = data["safety_label"]["score"]
                prompts[key] = data.get("prompt", "")
                cots[key] = data.get("cot", "")
    return activations, prompts, cots, labels

def prepare_data(activations, labels):
    """
    Reformat activations and labels into a single numpy array.
    """
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

    assert set(activations.keys()) ==  set(labels.keys()), f"difference: {set(activations.keys()) - set(labels.keys())}"
    for id in activations.keys():
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

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

### Logistic Regression
def train_logistic_regression(X_train, y_train, X_test, y_test):
    # LogReg supports L2 via 'C' parameter (inverse of lambda), but keeping it simple here.
    # If you wanted to link args.l2 to LogReg, you would set C = 1 / (args.l2 + 1e-9)
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_prob, model

### MLP Modules

class GaussianNoise(nn.Module):
    """Adds Gaussian noise to the input during training only."""
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
        
        # Regularization 1: Gaussian Noise on Input
        self.noise = GaussianNoise(std=noise_std)
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        
        # Regularization 2: Dropout after activation
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
    
def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
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
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    input_size = X_train.shape[1]
    
    # Initialize model with arguments for regularization
    model = CustomMLP2Layer(
        input_size, 
        dropout_rate=args.dropout, 
        noise_std=args.noise
    )
    
    # Calculate class weights
    num_pos = np.sum(y_train)
    num_neg = len(y_train) - num_pos
    if num_pos < num_neg:
        pos_weight = torch.FloatTensor([num_neg / num_pos])
        minority_class = 1
    else:
        pos_weight = torch.FloatTensor([num_pos / num_neg])
        minority_class = 0
        
    criterion = nn.BCELoss(reduction='none')
    
    # Regularization 3: L2 (Weight Decay) in Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.l2)
    
    num_epochs = 50
    best_f1 = 0
    best_model_state = None
    
    # Regularization 4: Configurable Early Stopping
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            weights = torch.ones_like(labels)
            weights[labels == minority_class] = pos_weight
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
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
            print(f"Early stopping at epoch {epoch+1} (Patience: {patience})")
            break
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val F1: {val_f1:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test
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
    activations_dict, prompts_dict, cots_dict, labels_dict = load_data()
    
    prompt_IDs = set([x.split("_")[0] for x in activations_dict.keys()])
    N = len(prompt_IDs)
    logger.debug(f"Loaded {len(activations_dict)=} activations of last-token CoTs for {N=} prompts.")
    
    # Report configuration
    logger.info(f"Configuration -> L2: {args.l2}, Dropout: {args.dropout}, Noise: {args.noise}, Patience: {args.patience}")

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
        torch.manual_seed(seed) # Ensure torch is also seeded
        
        train_prompt_ids = set(np.random.choice(sorted(list(prompt_IDs)), int(0.7 * N), replace=False))
        test_prompt_ids = prompt_IDs - train_prompt_ids
        print(f"Selected {len(train_prompt_ids)} prompts for training and {len(test_prompt_ids)} for testing")
        
        train_prompt_ids_list = list(train_prompt_ids)
        np.random.shuffle(train_prompt_ids_list)
        split_idx = int(0.9 * len(train_prompt_ids_list))
        train_prompt_ids = set(train_prompt_ids_list[:split_idx])
        val_prompt_ids = set(train_prompt_ids_list[split_idx:])
        
        X, labels_np, prompt_sent_ids  = prepare_data(activations_dict, labels_dict)
        
        train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in train_prompt_ids]
        val_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in val_prompt_ids]
        test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in test_prompt_ids]
        
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

        if (y_test == 0).sum() < (y_test == 1).sum():
            logger.info("Flipping labels (0->1, 1->0) so unsafe -> 0, safe (rarer) -> 1")
            y_train = 1 - y_train
            y_val = 1 - y_val
            y_test = 1 - y_test

        keys_test = [prompt_sent_ids[i] for i in test_indices]

        pca_model = None
        if args.pca:
            X_train, X_val, X_test, pca_model = apply_pca(X_train, X_val, X_test)
        
        # Train Probes
        logreg_y_pred, logreg_y_pred_prob, logreg_model = train_logistic_regression(X_train, y_train, X_test, y_test)
        mlp_y_pred, mlp_y_pred_prob, mlp_model, mlp_internal_scaler = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)

        # Random Baseline
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

        # Eval
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

    print(f"---- mean disagreement after shuffling: {total_disagreement_percentage/args.N_runs:.2f}%")
    print("\n" + "="*85)
    print(f"PER-SEED PERFORMANCE SUMMARY")
    print("="*85)
    print(f"{'Seed':<5} | {'Model':<8} | {'F1':<8} | {'Acc':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}")
    print("-" * 85)
    for res in seed_results:
        s = res['seed']
        lr = res['logreg']
        print(f"{s:<5} | {'LogReg':<8} | {lr['f1']:.4f}   | {lr['accuracy']:.4f}   | {lr['pr_auc']:.4f}   | {lr['auc_roc']:.4f}")
        mlp = res['mlp']
        print(f"{'':<5} | {'MLP':<8} | {mlp['f1']:.4f}   | {mlp['accuracy']:.4f}   | {mlp['pr_auc']:.4f}   | {mlp['auc_roc']:.4f}")
        print("-" * 85)
    print("="*85 + "\n")

    print(calculate_metrics_stats([
        D_final_logreg_scores,
        D_final_mlp_scores,
        D_final_rand_lr_scores,
        D_final_random_scores,
        D_final_theoretical_random_scores,
        D_final_always_ones_scores,
        D_final_always_zeros_scores
    ]))

if __name__ == "__main__":
    main()