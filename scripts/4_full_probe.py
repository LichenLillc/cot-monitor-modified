"""
Full code for training probes, including in-distribution, out-of-distribution, 
and predicting future answer safety.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve,auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils import eval_pred, add_to_final_scores, calculate_metrics_stats

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=['in_distribution', 'out_of_distribution', 'future_predict'], help="type of probe to run")
parser.add_argument("--train_input", type=str, required=True, help="input folder containing activations and labels for training")
parser.add_argument("--test_input", type=str, help="input folder containing activations and labels for testing (only for OOD mode)")
parser.add_argument("--N_runs", type=int, default=5, help="number of different seeded runs")
parser.add_argument("--sample_K", type=int, default=1000, help="number of training samples")
parser.add_argument("--pca", action="store_true", help="run PCA")
parser.add_argument("--pca_components", type=int, default=50, help="number of different seeded runs")
# for running future_predict mode
parser.add_argument("--prior_cot",  type=int, nargs='+', default=[0, 1, 5, 10, 20], help="List of prior COT sentences to use for prediction.")
parser.add_argument("--future_offsets", type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 10, 15, 20], help="List of future offsets to predict for.")

args = parser.parse_args()

if args.mode == 'out_of_distribution' and not args.test_input:
    parser.error("--test_input is required for out-of-distribution mode")

TRAIN_INPUT = pathlib.Path(args.train_input)
if args.test_input:
    TEST_INPUT = pathlib.Path(args.test_input)
else:
    TEST_INPUT = None


def _load_from_folder(folder_path):
    """
    Load activations and labels from a folder.
    """
    activations = dict()
    labels = dict()

    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return activations, labels

    for act_file in tqdm((folder_path / "activations").glob("*.pt"), desc=f"Loading activations from {folder_path}"):
        filename = os.path.basename(act_file)
        # key is in the format of "{prompt_id}_{sentence_id}"
        key = filename.split('.')[0]
        activation = torch.load(act_file)
        activations[key] = activation

    for label_file in tqdm((folder_path / "labels").rglob("*.json"), desc=f"Loading labels from {folder_path}"):
        # filename is in the format of "{prompt_id}_{sentence_id}_labeled.json"
        filename = os.path.basename(label_file)
        if filename.endswith("_labeled.json"):
            key = filename.split('.')[0].split('_')[0:2]
            key = '_'.join(key)
            with open(label_file, 'r') as f:
                data = json.load(f)
                labels[key] = data["safety_label"]["score"]
    return activations, labels

def load_data():
    """
    Load data based on the test mode.
    """
    if args.mode == 'in_distribution':
        activations, labels = _load_from_folder(TRAIN_INPUT)
        return activations, labels, None, None
    elif args.mode == 'out_of_distribution':
        train_activations, train_labels = _load_from_folder(TRAIN_INPUT)
        test_activations, test_labels = _load_from_folder(TEST_INPUT)
        return train_activations, train_labels, test_activations, test_labels
    elif args.mode == 'future_predict':
        activations, labels = _load_from_folder(TRAIN_INPUT)
        return activations, labels, None, None

def prepare_data(activations, labels, offset=0):
    """
    Reformat activations and labels into a single numpy array.
    """
    def convert_to_numpy(tensor):
        # convert a tensor to numpy, handle bfloat16
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
    
    # make sure we only process ids that exist in both activations and labels
    if offset > 0:
        logger.info(f"Preparing data with future offset: {offset}")
        for key, act in tqdm(activations.items(), desc="Aligning activations and future labels"):
            prompt_id, sent_id_str = key.split("_")
            try:
                sent_id = int(sent_id_str)
            except ValueError:
                continue # skip if sentence id is not an integer
            
            future_sent_id = sent_id + offset
            future_key = f"{prompt_id}_{future_sent_id}"

            if future_key in labels:
                activations_list.append(convert_to_numpy(act))
                labels_list.append(labels[future_key])
                prompt_sent_ids.append(key)
    else:
        common_ids = set(activations.keys()) & set(labels.keys())
        print(f"common_ids: {len(common_ids)}")
        for id in common_ids:
            # convert to numpy first
            activations_list.append(convert_to_numpy(activations[id]))
            labels_list.append(labels[id]) 
            prompt_sent_ids.append(id)
    
    if not activations_list:
        logger.warning(f"No matching activation-label pairs found for offset {offset}")
        return None, None, None
    
    X = np.vstack(activations_list)
    labels_np = np.array(labels_list)    
    return X, labels_np, prompt_sent_ids

def apply_pca(X_train, X_val, X_test):
    # Fit PCA on training data
    pca_components = min(args.pca_components, X_train.shape[0], X_train.shape[1])
    logger.info(f"PCA:::reducing to {pca_components}")
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train)
    # Transform test data using the same PCA
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca


### logistic regression
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_prob

### MLP
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_size = X_train.shape[1]
    model = CustomMLP2Layer(input_size)
    
    # calculate class weights for weighted BCE loss
    pos_weight = torch.FloatTensor([len(y_train) / (np.sum(y_train) + 1e-8)])
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # training loop
    num_epochs = 50
    best_f1 = 0
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            weights = torch.ones_like(labels)
            weights[labels == 0] = pos_weight
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # validation
        model.eval()
        y_pred_val = []
        y_true_val = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                y_pred_val.extend(outputs.cpu().numpy())
                y_true_val.extend(labels.cpu().numpy())
        
        y_pred_val = np.array(y_pred_val).flatten()
        y_true_val = np.array(y_true_val).flatten()
        
        # convert to binary
        y_pred_binary = (y_pred_val >= 0.5).astype(int)
        val_f1 = f1_score(y_true_val, y_pred_binary, average="binary")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val F1: {val_f1:.4f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # test
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred_prob = y_pred_tensor.cpu().numpy().flatten()
        
    y_pred = (y_pred_prob >= 0.5).astype(int)
    return y_pred, y_pred_prob

def calculate_metrics_stats(final_lr_aucs, final_mlp_aucs, rand_lr_aucs, rand_mlp_aucs,
                        final_lr_f1s, final_mlp_f1s, rand_lr_f1s, rand_mlp_f1s):
    stats = {}
    # calculate AUC mean and std
    stats['lr_mean'] = sum(final_lr_aucs)/len(final_lr_aucs)*100
    stats['lr_std'] = np.std(final_lr_aucs)*100
    stats['mlp_mean'] = sum(final_mlp_aucs)/len(final_mlp_aucs)*100
    stats['mlp_std'] = np.std(final_mlp_aucs)*100
    stats['rand_lr_mean'] = sum(rand_lr_aucs)/len(rand_lr_aucs)*100
    stats['rand_lr_std'] = np.std(rand_lr_aucs)*100
    stats['rand_mlp_mean'] = sum(rand_mlp_aucs)/len(rand_mlp_aucs)*100
    stats['rand_mlp_std'] = np.std(rand_mlp_aucs)*100

    # calculate F1 mean and std
    stats['lr_f1_mean'] = sum(final_lr_f1s)/len(final_lr_f1s)*100
    stats['lr_f1_std'] = np.std(final_lr_f1s)*100
    stats['mlp_f1_mean'] = sum(final_mlp_f1s)/len(final_mlp_f1s)*100
    stats['mlp_f1_std'] = np.std(final_mlp_f1s)*100
    stats['rand_lr_f1_mean'] = sum(rand_lr_f1s)/len(rand_lr_f1s)*100
    stats['rand_lr_f1_std'] = np.std(rand_lr_f1s)*100
    stats['rand_mlp_f1_mean'] = sum(rand_mlp_f1s)/len(rand_mlp_f1s)*100
    stats['rand_mlp_f1_std'] = np.std(rand_mlp_f1s)*100
    
    return stats

def run_future_predict():
    """
    Trains a single probe on offset=0 data (predicting current safety)
    and tests its ability to predict future safety (offset>0) on held-out data.
    """
    activations_dict, labels_dict, _, _ = load_data()
    all_results_by_prior = {}

    # get all unique prompt IDs and split them once for all runs
    all_prompt_ids = list(set([x.split("_")[0] for x in activations_dict.keys()]))
    np.random.shuffle(all_prompt_ids)
    train_val_split_idx = int(0.7 * len(all_prompt_ids))
    train_val_prompt_ids = all_prompt_ids[:train_val_split_idx]
    test_prompt_ids = set(all_prompt_ids[train_val_split_idx:])
    
    # get training data
    train_val_activations = {k: v for k, v in activations_dict.items() if k.split('_')[0] in train_val_prompt_ids}
    X_train_val_all, labels_train_val_all, prompt_sent_ids_train_val = prepare_data(train_val_activations, labels_dict, offset=0)

    logger.info(f"Created train/val data with {len(X_train_val_all)} samples from {len(train_val_prompt_ids)} prompts.")

    for prior_cot_sent_to_test in args.prior_cot:
        print(f"\n\n{'='*20} Using CoT sentence {prior_cot_sent_to_test} as test {'='*20}")

        # filter test activations to only the specific CoT sentence
        test_activations_for_prior = {
            k: v for k, v in activations_dict.items()
            if k.split('_')[0] in test_prompt_ids and k.split('_')[-1] == str(prior_cot_sent_to_test)
        }
            
        logger.info(f" {len(test_activations_for_prior)} test samples for CoT sentence {prior_cot_sent_to_test}.")

        results_for_prior = {}
        for offset in args.future_offsets:
            print(f"\n===== Offset {offset}: predicting label for sent {prior_cot_sent_to_test + offset} =====\n")
            
            X_test_all, labels_test_all, _ = prepare_data(test_activations_for_prior, labels_dict, offset=offset)
            y_test = (labels_test_all >= 0.5).astype(int)
            
            logger.info("Flipping labels (0->1, 1->0) so unsafe: 0, safe (rarer): 1")
            y_test = 1 - y_test

            final_lr_f1s, final_mlp_f1s, rand_lr_f1s, rand_mlp_f1s = [], [], [], []
            random_f1s, always_ones_f1s, always_zeros_f1s = [], [], []
            n_train_samples_per_run, n_test_samples_per_run = [], []
            n_test_samples_per_run.append(len(X_test_all))

            for seed in range(args.N_runs):
                np.random.seed(seed)
                
                train_prompt_ids_list = list(train_val_prompt_ids)
                np.random.shuffle(train_prompt_ids_list)
                split_idx = int(0.9 * len(train_prompt_ids_list))
                train_ids_for_split = set(train_prompt_ids_list[:split_idx])
                val_ids_for_split = set(train_prompt_ids_list[split_idx:])

                train_indices = [i for i, key in enumerate(prompt_sent_ids_train_val) if key.split('_')[0] in train_ids_for_split]
                val_indices = [i for i, key in enumerate(prompt_sent_ids_train_val) if key.split('_')[0] in val_ids_for_split]

                if args.sample_K and args.sample_K > 0 and len(train_indices) > args.sample_K:
                    np.random.shuffle(train_indices)
                    train_indices = train_indices[:args.sample_K]

                X_train, y_train_orig = X_train_val_all[train_indices], (labels_train_val_all[train_indices] >= 0.5).astype(int)
                X_val, y_val_orig = X_train_val_all[val_indices], (labels_train_val_all[val_indices] >= 0.5).astype(int)
                
                y_train = 1 - y_train_orig
                y_val = 1 - y_val_orig

                logger.debug(f"Offset {offset} Seed {seed}: Train={len(X_train)} (Safe: {np.sum(y_train==1)}, Unsafe: {np.sum(y_train==0)}), Val={len(X_val)}, Test={len(X_test_all)}")
                n_train_samples_per_run.append(len(X_train))
                
                X_train_final, X_val_final, X_test_final = X_train, X_val, X_test_all
                if args.pca:
                    X_train_final, X_val_final, X_test_final = apply_pca(X_train, X_val, X_test_all)

                # probe training
                lr_y_pred, _ = train_logistic_regression(X_train_final, y_train, X_test_final, y_test)
                lr_f1 = f1_score(y_test, lr_y_pred, average="binary")
                
                mlp_y_pred, _ = train_mlp(X_train_final, y_train, X_val_final, y_val, X_test_final, y_test)
                mlp_f1 = f1_score(y_test, mlp_y_pred, average="binary")
                
                # random baseline
                y_train_shuffled = np.copy(y_train)
                np.random.shuffle(y_train_shuffled)
                rand_lr_y_pred, _ = train_logistic_regression(X_train_final, y_train_shuffled, X_test_final, y_test)
                rand_lr_f1 = f1_score(y_test, rand_lr_y_pred, average="binary")

                # other baselines
                positive_prior = np.sum(y_train == 1)/len(y_train)
                random_y_pred = np.random.choice([1,0], size=len(y_test), p=[positive_prior, 1-positive_prior])
                always_ones_pred = np.ones(len(y_test))
                always_zeros_pred = np.zeros(len(y_test))
                random_f1 = f1_score(y_test, random_y_pred, average="binary", zero_division=0)
                always_ones_f1 = f1_score(y_test, always_ones_pred, average="binary", zero_division=0)
                always_zeros_f1 = f1_score(y_test, always_zeros_pred, average="binary", zero_division=0)

                final_lr_f1s.append(lr_f1)
                final_mlp_f1s.append(mlp_f1)
                rand_lr_f1s.append(rand_lr_f1)
                random_f1s.append(random_f1)
                always_ones_f1s.append(always_ones_f1)
                always_zeros_f1s.append(always_zeros_f1)
            
            if final_lr_f1s:
                results_for_prior[offset] = {
                    'n_train': int(np.mean(n_train_samples_per_run)),
                    'n_test': int(np.mean(n_test_samples_per_run)),
                    'lr_f1_mean': np.mean(final_lr_f1s) * 100, 'lr_f1_std': np.std(final_lr_f1s) * 100,
                    'mlp_f1_mean': np.mean(final_mlp_f1s) * 100, 'mlp_f1_std': np.std(final_mlp_f1s) * 100,
                    'rand_lr_f1_mean': np.mean(rand_lr_f1s) * 100, 'rand_lr_f1_std': np.std(rand_lr_f1s) * 100,
                    'random_f1_mean': np.mean(random_f1s) * 100, 'random_f1_std': np.std(random_f1s) * 100,
                    'always_ones_f1_mean': np.mean(always_ones_f1s) * 100, 'always_ones_f1_std': np.std(always_ones_f1s) * 100,
                    'always_zeros_f1_mean': np.mean(always_zeros_f1s) * 100, 'always_zeros_f1_std': np.std(always_zeros_f1s) * 100,
                }
        all_results_by_prior[prior_cot_sent_to_test] = results_for_prior

    print("\n\n===== Future Predict Results =====")
    for prior_cot_sent, results in sorted(all_results_by_prior.items()):
        print(f"\n--- Test on CoT sentence {prior_cot_sent} ---")
        header = f"{'Offset':<8} | {'Target':<8} | {'N Trn':<8} | {'N Tst':<8} | {'LR F1':<18} | {'MLP F1':<18} | {'Rand LR F1':<18} | {'Rand F1':<18} | {'Ones F1':<18} | {'Zeros F1':<18}"
        print(header)
        print("-" * len(header))
        for offset, res in sorted(results.items()):
            target_sent = prior_cot_sent + offset
            lr_str = f"{res['lr_f1_mean']:.1f}% ± {res['lr_f1_std']:.1f}"
            mlp_str = f"{res['mlp_f1_mean']:.1f}% ± {res['mlp_f1_std']:.1f}"
            rand_lr_str = f"{res['rand_lr_f1_mean']:.1f}% ± {res['rand_lr_f1_std']:.1f}"
            random_str = f"{res['random_f1_mean']:.1f}% ± {res['random_f1_std']:.1f}"
            always_ones_str = f"{res['always_ones_f1_mean']:.1f}% ± {res['always_ones_f1_std']:.1f}"
            always_zeros_str = f"{res['always_zeros_f1_mean']:.1f}% ± {res['always_zeros_f1_std']:.1f}"
            print(f"{offset:<8} | {target_sent:<8} | {res['n_train']:<8} | {res['n_test']:<8} | {lr_str:<18} | {mlp_str:<18} | {rand_lr_str:<18} | {random_str:<18} | {always_ones_str:<18} | {always_zeros_str:<18}")
        print("=" * len(header))


def run_standard_probe():
    # Initialize lists to store results across all runs
    D_final_logreg_scores = collections.defaultdict(list)
    D_final_mlp_scores = collections.defaultdict(list)
    D_final_rand_lr_scores = collections.defaultdict(list)
    D_final_random_scores = collections.defaultdict(list)
    D_final_always_ones_scores = collections.defaultdict(list)
    D_final_always_zeros_scores = collections.defaultdict(list)
    total_disagreement_percentage = 0

    # select data based on mode
    if args.mode == "in_distribution":
        activations_dict, labels_dict, _, _ = load_data()
        prompt_IDs = set([x.split("_")[0] for x in activations_dict.keys()])
        N = len(prompt_IDs)
        logger.debug(f"Loaded {len(activations_dict)} activations for {N} prompts.")
        X_all, labels_all, prompt_sent_ids_all = prepare_data(activations_dict, labels_dict)
    
    elif args.mode == "out_of_distribution":
        train_activations_dict, train_labels_dict, test_activations_dict, test_labels_dict = load_data()
        train_prompt_IDs = set([x.split("_")[0] for x in train_activations_dict.keys()])
        test_prompt_IDs = set([x.split("_")[0] for x in test_activations_dict.keys()])
        logger.debug(f"Loaded {len(train_activations_dict)} train activations and {len(test_activations_dict)} test activations.")
        X_train_all, y_train_all, train_prompt_sent_ids = prepare_data(train_activations_dict, train_labels_dict)
        X_test_all, y_test_all, test_prompt_sent_ids = prepare_data(test_activations_dict, test_labels_dict)

    for seed in range(0, args.N_runs):
        np.random.seed(seed)  # for reproducibility

        if args.mode == "in_distribution":
            train_prompt_ids = set(np.random.choice(list(prompt_IDs), int(0.7 * N), replace=False))
            test_prompt_ids = prompt_IDs - train_prompt_ids
            
            train_prompt_ids_list = list(train_prompt_ids)
            np.random.shuffle(train_prompt_ids_list)
            split_idx = int(0.9 * len(train_prompt_ids_list))
            train_prompt_ids = set(train_prompt_ids_list[:split_idx])
            val_prompt_ids = set(train_prompt_ids_list[split_idx:])
            logger.debug(f"Run {seed+1}/{args.N_runs} - Train prompts: {len(train_prompt_ids)}, Val prompts: {len(val_prompt_ids)}, Test prompts: {len(test_prompt_ids)}")

            train_indices = [i for i, key in enumerate(prompt_sent_ids_all) if key.split('_')[0] in train_prompt_ids]
            val_indices = [i for i, key in enumerate(prompt_sent_ids_all) if key.split('_')[0] in val_prompt_ids]
            test_indices = [i for i, key in enumerate(prompt_sent_ids_all) if key.split('_')[0] in test_prompt_ids]

            if args.sample_K and args.sample_K > 0:
                assert len(train_indices) >= args.sample_K, f"Not enough training samples. Required: {args.sample_K}, Available: {len(train_indices)}"
                np.random.shuffle(train_indices)
                train_indices = train_indices[:args.sample_K]
                logger.info(f"Using {args.sample_K} training samples.")
            
            X_train = X_all[train_indices]
            X_val = X_all[val_indices]
            X_test = X_all[test_indices]
            
            y_train = (labels_all[train_indices] >= 0.5).astype(int)
            y_val = (labels_all[val_indices] >= 0.5).astype(int)
            y_test = (labels_all[test_indices] >= 0.5).astype(int)

        elif args.mode == "out_of_distribution":
            train_prompt_ids_list = list(train_prompt_IDs)
            np.random.shuffle(train_prompt_ids_list)
            split_idx = int(0.9 * len(train_prompt_ids_list))
            train_prompt_ids = set(train_prompt_ids_list[:split_idx])
            val_prompt_ids = set(train_prompt_ids_list[split_idx:])
            logger.debug(f"Run {seed+1}/{args.N_runs} - Train prompts: {len(train_prompt_ids)}, Val prompts: {len(val_prompt_ids)}, Test prompts: {len(test_prompt_IDs)}")

            train_indices = [i for i, key in enumerate(train_prompt_sent_ids) if key.split('_')[0] in train_prompt_ids]
            val_indices = [i for i, key in enumerate(train_prompt_sent_ids) if key.split('_')[0] in val_prompt_ids]
            
            if args.sample_K and args.sample_K > 0:
                assert len(train_indices) >= args.sample_K, f"Not enough training samples. Required: {args.sample_K}, Available: {len(train_indices)}"
                np.random.shuffle(train_indices)
                train_indices = train_indices[:args.sample_K]
                logger.info(f"Using {args.sample_K} training samples.")
            
            X_train, y_train_orig = X_train_all[train_indices], (y_train_all[train_indices] >= 0.5).astype(int)
            X_val, y_val_orig = X_train_all[val_indices], (y_train_all[val_indices] >= 0.5).astype(int)
            X_test, y_test_orig = X_test_all, (y_test_all >= 0.5).astype(int)
            
            y_train, y_val, y_test = y_train_orig, y_val_orig, y_test_orig
        
        # LABELS => unsafe: 0, safe (rarer): 1
        logger.info("Flipping labels (0->1, 1->0) so unsafe: 0, safe (rarer): 1")
        y_train = 1 - y_train
        y_val = 1 - y_val
        y_test = 1 - y_test

        if args.pca:
            X_train, X_val, X_test = apply_pca(X_train, X_val, X_test)
        
        logger.debug(f"Training set: {len(X_train)} latents (Safe: {np.sum(y_train==1)}, Unsafe: {np.sum(y_train==0)})")
        logger.debug(f"Validation set: {len(X_val)} latents (Safe: {np.sum(y_val==1)}, Unsafe: {np.sum(y_val==0)})")
        logger.debug(f"Testing set: {len(X_test)} latents (Safe: {np.sum(y_test==1)}, Unsafe: {np.sum(y_test==0)})")

        ##############################
        ### train safety probes
        ##############################
        logreg_y_pred, logreg_y_pred_prob = train_logistic_regression(X_train, y_train, X_test, y_test)
        mlp_y_pred, mlp_y_pred_prob = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)

        ##############################
        ### random baseline
        ##############################
        np.random.seed(seed)  # use same seed as outer loop for reproducibility
        positive_prior = np.sum(y_train == 1)/len(y_train)
        random_y_pred = np.random.choice([1,0], size=len(X_test), p=[positive_prior, 1-positive_prior])
        always_ones_pred = np.ones(len(X_test))
        always_zeros_pred = np.zeros(len(X_test))


        # shuffle
        np.random.seed(seed)  # use same seed as outer loop for reproducibility
        y_train_shuffled = np.copy(y_train)
        shuffled_indices = np.arange(len(y_train))
        np.random.shuffle(shuffled_indices)
        y_train_shuffled = y_train[shuffled_indices]
        
        disagreement = np.sum(y_train != y_train_shuffled)
        disagreement_percentage = (disagreement / len(y_train)) * 100
        total_disagreement_percentage += disagreement_percentage
        rand_lr_pred, rand_lr_pred_prob = train_logistic_regression(X_train, y_train_shuffled, X_test, y_test)

        ### eval
        logreg_eval = eval_pred(y_test, logreg_y_pred, logreg_y_pred_prob, metrics=["f1", "accuracy", "pr_auc"])
        mlp_eval = eval_pred(y_test, mlp_y_pred, mlp_y_pred_prob, metrics=["f1", "accuracy", "pr_auc"])
        rand_lr_eval = eval_pred(y_test, rand_lr_pred, rand_lr_pred_prob, metrics=["f1", "accuracy", "pr_auc"])
        random_eval = eval_pred(y_test, random_y_pred, metrics=["f1", "accuracy"])
        always_ones_eval = eval_pred(y_test, always_ones_pred, metrics=["f1", "accuracy"])
        always_zeros_eval = eval_pred(y_test, always_zeros_pred, metrics=["f1", "accuracy"])

        add_to_final_scores(logreg_eval, D_final_logreg_scores, 'logreg')
        add_to_final_scores(mlp_eval, D_final_mlp_scores, 'mlp')
        add_to_final_scores(rand_lr_eval, D_final_rand_lr_scores, 'random_logreg')
        add_to_final_scores(random_eval, D_final_random_scores, 'expected_random')
        add_to_final_scores(always_ones_eval, D_final_always_ones_scores, "always_ones")
        add_to_final_scores(always_zeros_eval, D_final_always_zeros_scores, "always_zeros")

    print(f"---- mean disagreement after shuffling: {total_disagreement_percentage/args.N_runs:.2f}%")
    if args.mode == "in_distribution":
        print(f"(N_train: {len(train_indices)}. N_test: {len(test_indices)})")

    print(calculate_metrics_stats([
        D_final_logreg_scores,
        D_final_mlp_scores,
        D_final_rand_lr_scores,
        D_final_random_scores,
        D_final_always_ones_scores,
        D_final_always_zeros_scores
    ]))

def main():
    if args.mode == 'future_predict':
        run_future_predict()
    else:
        run_standard_probe()

if __name__ == "__main__":
    main()