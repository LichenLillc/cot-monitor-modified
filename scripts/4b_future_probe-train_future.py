"""
Train probes on current CoT activations and future labels to predict future answer safety.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from utils import eval_pred, add_to_final_scores, calculate_metrics_stats, save_probe_outputs_tsv

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, required=True, help="input folder containing activations and labels")
parser.add_argument("--N_runs", type=int, default=5, help="number of different seeded runs")
parser.add_argument("--sample_K", type=int, default=-1, help="number of training samples")
parser.add_argument("--pca", action="store_true", help="run PCA")
parser.add_argument("--pca_components", type=int, default=50, help="number of different seeded runs")

### future predictions
parser.add_argument("--prior_cot", type=int, default=-1, help="how many CoT sentences already generated")
parser.add_argument("--left_shift", type=int, default=0, help="left shift = 1 means predicting alignment for 1 CoT sentence in advance")
parser.add_argument("--left_shift_mult", type=float, default=1.0, help="left shift mult = 2x means predicting two times prior sentences. (1x is base)")
logger.level("FUTURE", no=15, color="<red>", icon="@")

### storing test prediction outputs
parser.add_argument("--store_outputs", action="store_true", help="whether to store model outputs")
parser.add_argument("--probe_output_folder", type=str, default="../probe_outputs/", help="folder to store model outputs and results")

args = parser.parse_args()
INPUT_FOLDER = pathlib.Path(args.input_folder)
if args.store_outputs:
    PROBE_OUTPUT_FOLDER = pathlib.Path(args.probe_output_folder) / INPUT_FOLDER.name
    PROBE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# assert bool(args.left_shift_mult - 1.0) != bool(args.left_shift) #xor relationship
if args.left_shift > 0 or args.left_shift_mult > 0:
    # running FUTURE prediction experiments
    assert args.prior_cot >= 0

### load and engineer data
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
        # key is in the format of "{prompt_id}_{sentence_id}"
        key = filename.split('.')[0]
        activation = torch.load(act_file)
        activations[key] = activation

    for label_file in tqdm((INPUT_FOLDER / "labels").rglob("*.json"), desc="Loading labels and texts"):
        # filename is in the format of "{prompt_id}_{sentence_id}_labeled.json"
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

    # sanity check: make sure we only process ids that exist in both activations and labels
    assert set(activations.keys()) ==  set(labels.keys()), f"difference: {set(activations.keys()) - set(labels.keys())}"
    for id in activations.keys():
        # convert to numpy first
        activations_list.append(convert_to_numpy(activations[id]))
        labels_list.append(labels[id]) 
        prompt_sent_ids.append(id)

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
    pos_weight = torch.FloatTensor([len(y_train) / np.sum(y_train)])
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
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                y_pred.extend(outputs.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        
        # convert to binary
        y_pred_binary = (y_pred >= 0.5).astype(int)
        val_f1 = f1_score(y_true, y_pred_binary, average="binary")
        
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


########################
######## main ##########
########################
def main():
    activations_dict, prompts_dict, cots_dict, labels_dict = load_data()
    
    # get train-test split based on prompt IDs
    prompt_IDs = set([x.split("_")[0] for x in activations_dict.keys()])
    N = len(prompt_IDs)
    logger.debug(f"Loaded {len(activations_dict)=} activations of last-token CoTs for {N=} prompts.")

    # Initialize lists to store results across all runs
    D_final_logreg_scores = collections.defaultdict(list)

    for seed in range(args.N_runs):
        np.random.seed(seed)  # for reproducibility
        train_prompt_ids = set(np.random.choice(sorted(list(prompt_IDs)), int(0.7 * N), replace=False))
        test_prompt_ids = prompt_IDs - train_prompt_ids
        print(f"Selected {len(train_prompt_ids)} prompts for training and {len(test_prompt_ids)} for testing")
        
        # split train_prompt_ids into train and validation sets (90:10 split)
        train_prompt_ids_list = list(train_prompt_ids)
        np.random.shuffle(train_prompt_ids_list)
        split_idx = int(0.9 * len(train_prompt_ids_list))
        train_prompt_ids = set(train_prompt_ids_list[:split_idx])
        val_prompt_ids = set(train_prompt_ids_list[split_idx:])
        logger.debug(f"Train prompts: {len(train_prompt_ids)}, Val prompts: {len(val_prompt_ids)}, Test prompts: {len(test_prompt_ids)}")
        
        # prepare data for this iteration
        X, labels_np, prompt_sent_ids  = prepare_data(activations_dict, labels_dict)
        # train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in train_prompt_ids]
        val_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in val_prompt_ids]
        # 
        if args.prior_cot >= 0:
            logger.log("FUTURE", f"applying {args.prior_cot=} to train set")
            train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in train_prompt_ids and int(key.split('_')[1]) == args.prior_cot]
            logger.log("FUTURE", f"applying {args.prior_cot=} to test set")
            test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in test_prompt_ids and int(key.split('_')[1]) == args.prior_cot]
        else:
            train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in train_prompt_ids]
            test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in test_prompt_ids]

        ##############################
        ### FUTURE
        ##############################
        if args.left_shift:
            absolute_left_shift = args.left_shift
        elif args.left_shift_mult:
            absolute_left_shift = int(args.prior_cot * (args.left_shift_mult - 1.0))
        else:
            absolute_left_shift = 0

        key2indices = {key: i for i, key in enumerate(prompt_sent_ids)}
        indices2key = {i: key for i, key in enumerate(prompt_sent_ids)}

        ### sanity check: is the left_shift the right config? (aka would any of the existing test prompts not have enough CoTs?)
        shortest_cots = float('inf')
        seen_test_key = set()
        for test_i in test_indices:
            test_key = indices2key[test_i]
            prompt_id, sent_id = test_key.split("_")
            if prompt_id in seen_test_key:
                continue
            max_len = 0
            for future_sent_i in range(5000):
                if f"{prompt_id}_{future_sent_i}" in key2indices:
                    continue
                max_len = max(future_sent_i, max_len)
                break
            seen_test_key.add(prompt_id)
            shortest_cots = min(shortest_cots, max_len)
        logger.log("FUTURE", f"analysis --- minimum #CoT in the test sets is {shortest_cots}.")
        assert args.prior_cot + absolute_left_shift < shortest_cots, f"{args.prior_cot=} {absolute_left_shift=} {shortest_cots=}"
        
        #### TODO: write a function for both below.
        ### left shift the labels for test_indices
        left_shift_test_label_indices = []
        original_test_indices = [] # create this because not all `test_indices` are added to `left_shift_test_label_indices`, and we want X_test to match
        
        for test_i in test_indices:
            test_key = indices2key[test_i]
            prompt_id, sent_id = test_key.split("_")
            left_shift_test_key = f"{prompt_id}_{int(sent_id) + absolute_left_shift}"
            if left_shift_test_key in key2indices:
                left_shift_test_i = key2indices[left_shift_test_key]
                left_shift_test_label_indices.append(left_shift_test_i)
                original_test_indices.append(test_i)

                if absolute_left_shift != 0: # sanity check
                    assert left_shift_test_i != test_i
                else:
                    assert left_shift_test_i == test_i
        
        test_indices = original_test_indices

        ### left shift the labels for train_indices
        left_shift_train_label_indices = []
        original_train_indices = [] 
        
        for train_i in train_indices:
            train_key = indices2key[train_i]
            prompt_id, sent_id = train_key.split("_")
            left_shift_train_key = f"{prompt_id}_{int(sent_id) + absolute_left_shift}"
            if left_shift_train_key in key2indices:
                left_shift_train_i = key2indices[left_shift_train_key]
                left_shift_train_label_indices.append(left_shift_train_i)
                original_train_indices.append(train_i)

                if absolute_left_shift != 0: # sanity check
                    assert left_shift_train_i != train_i
                else:
                    assert left_shift_train_i == train_i
        
        train_indices = original_train_indices
        ############################################################
        ############################################################
        
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

        # convert scores to binary classes
        threshold = 0.5
        y_train = (labels_np[left_shift_train_label_indices] >= threshold).astype(int) 
        _current_y_train = (labels_np[train_indices] >= threshold).astype(int)
        _label_changes = _current_y_train != y_train
        logger.log("FUTURE", f"applying left-shift = {absolute_left_shift} leads to {sum(_label_changes)/len(_label_changes):.1f}% out of N={len(_label_changes)} labels changed.")
        y_val = (labels_np[val_indices] >= threshold).astype(int)

        y_test = (labels_np[left_shift_test_label_indices] >= threshold).astype(int) # NOTE: apply FUTURE thinking
        _current_y_test = (labels_np[test_indices] >= threshold).astype(int)
        _label_changes = _current_y_test != y_test
        logger.log("FUTURE", f"applying left-shift = {absolute_left_shift} leads to {sum(_label_changes)/len(_label_changes):.1f}% out of N={len(_label_changes)} labels changed.")

        # LABELS => unsafe: 0, safe (rarer): 1
        if (y_val == 0).sum() < (y_val == 1).sum():
            logger.info("Flipping labels (0->1, 1->0) so unsafe -> 0, safe (rarer) -> 1")
            y_train = 1 - y_train
            y_val = 1 - y_val
            y_test = 1 - y_test

        keys_train = [prompt_sent_ids[i] for i in train_indices]
        keys_val = [prompt_sent_ids[i] for i in val_indices]
        keys_test = [prompt_sent_ids[i] for i in test_indices]

        if args.pca:
            X_train, X_val, X_test = apply_pca(X_train, X_val, X_test)
        
        logger.debug(f"Training set: {len(X_train)} latents (Safe: {np.sum(y_train==1)}, Unsafe: {np.sum(y_train==0)})")
        logger.debug(f"Validation set: {len(X_val)} latents (Safe: {np.sum(y_val==1)}, Unsafe: {np.sum(y_val==0)})")
        logger.debug(f"Testing set: {len(X_test)} latents (Safe: {np.sum(y_test==1)}, Unsafe: {np.sum(y_test==0)})")

        
        ##############################
        ### train safety probes
        ##############################
        logreg_y_pred, logreg_y_pred_prob = train_logistic_regression(X_train, y_train, X_test, y_test)

        ### eval
        logreg_eval = eval_pred(y_test, logreg_y_pred, logreg_y_pred_prob, metrics=["f1", "accuracy", "pr_auc"])
        add_to_final_scores(logreg_eval, D_final_logreg_scores, 'logreg')
        
        ##############################
        #### save test outputs
        ##############################
        if args.store_outputs:
            test_text_prompts = [prompts_dict[key] for key in keys_test]
            test_text_cots = [cots_dict[key] for key in keys_test]

            save_probe_outputs_tsv(
                output_dir=PROBE_OUTPUT_FOLDER,
                probe_name=f"logreg_seed{seed}",
                prompt_sent_ids=keys_test,
                prompts=test_text_prompts,
                cots=test_text_cots,
                true_labels=y_test,
                pred_labels=logreg_y_pred,
                pred_probs=logreg_y_pred_prob
            )

    logger.log("FUTURE", f"{args.prior_cot=}, {absolute_left_shift=} ({args.left_shift=}, {args.left_shift_mult=})")
    print(f"(N_train: {len(train_indices)}. N_test: {len(test_indices)})")

    print(calculate_metrics_stats([
        D_final_logreg_scores,
    ]))

if __name__ == "__main__":
    main()