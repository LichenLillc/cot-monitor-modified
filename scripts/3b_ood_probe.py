"""Train logistic regression and MLP probes on out-of-distribution datasets."""
from loguru import logger
import os
import json
import torch
import argparse
import pathlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve,auc, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--train_input", type=str, required=True, help="input folder containing activations and labels for training")
parser.add_argument("--test_input", type=str, required=True, help="input folder containing activations and labels for testing")
parser.add_argument("--N_runs", type=int, default=5, help="number of different seeded runs")
parser.add_argument("--sample_K", type=int, default=1000, help="number of training samples")
parser.add_argument("--pca", action="store_true", help="run PCA")
parser.add_argument("--pca_components", type=int, default=50, help="number of different seeded runs")
args = parser.parse_args()
TRAIN_INPUT = pathlib.Path(args.train_input)
TEST_INPUT = pathlib.Path(args.test_input)

##########################
### load and engineer data
##########################
def load_data():
    train_activations = dict()
    train_labels = dict()
    test_activations = dict()
    test_labels = dict()

    # get training data
    for act_file in (TRAIN_INPUT / "activations").glob("*.pt"):
        filename = os.path.basename(act_file)
        # key is in the format of "{prompt_id}_{sentence_id}"
        key = filename.split('.')[0]
        activation = torch.load(act_file)
        train_activations[key] = activation

    for label_file in (TRAIN_INPUT / "labels").rglob("*.json"):
        # filename is in the format of "{prompt_id}_{sentence_id}_labeled.json"
        filename = os.path.basename(label_file)
        if filename.endswith("_labeled.json"):
            key = filename.split('.')[0].split('_')[0:2]
            key = '_'.join(key)
            with open(label_file, 'r') as f:
                data = json.load(f)
                train_labels[key] = data["safety_label"]["score"]

    # get test data
    for act_file in (TEST_INPUT / "activations").glob("*.pt"):
        filename = os.path.basename(act_file)
        # key is in the format of "{prompt_id}_{sentence_id}"
        key = filename.split('.')[0]
        activation = torch.load(act_file)
        test_activations[key] = activation

    for label_file in (TEST_INPUT / "labels").rglob("*.json"):
        # filename is in the format of "{prompt_id}_{sentence_id}_labeled.json"
        filename = os.path.basename(label_file)
        if filename.endswith("_labeled.json"):
            key = filename.split('.')[0].split('_')[0:2]
            key = '_'.join(key)
            with open(label_file, 'r') as f:
                data = json.load(f)
                test_labels[key] = data["safety_label"]["score"]

    return train_activations, train_labels, test_activations, test_labels

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
    
    # make sure we only process ids that exist in both activations and labels
    common_ids = set(activations.keys()) & set(labels.keys())
    print(f"common_ids: {len(common_ids)}")
    for id in common_ids:
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


#######################
### logistic regression
#######################
def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="macro")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)

    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"])

    # print results
    print(f"\n\nLogistic Regression:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print(f"\n\nConfusion Matrix:")
    print(conf_matrix)
    print(f"Classification Report:")
    print(class_report)

    # report only weighted F1 and PR-AUC
    return weighted_f1, pr_auc

############
### MLP
############
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
        val_f1 = f1_score(y_true, y_pred_binary, average="macro")
        
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
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="macro")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"])

    print("\n\nMLP:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\n\nConfusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # report only weighted F1 and PR-AUC
    return weighted_f1, pr_auc

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

def main():
    train_activations_dict, train_labels_dict, test_activations_dict, test_labels_dict = load_data()
    
    # get prompt IDs
    train_prompt_IDs = set([x.split("_")[0] for x in train_activations_dict.keys()])
    test_prompt_IDs = set([x.split("_")[0] for x in test_activations_dict.keys()])
    N = len(train_prompt_IDs)
    logger.debug(f"Loaded {len(train_activations_dict)=} activations of last-token CoTs for {N=} prompts.")

    # Initialize lists to store results across all runs
    final_lr_aucs = []
    final_mlp_aucs = []
    rand_lr_aucs = []
    rand_mlp_aucs = []
    final_lr_f1s = []
    final_mlp_f1s = []
    rand_lr_f1s = []
    rand_mlp_f1s = []
    total_disagreement_percentage = 0

    for seed in range(0, args.N_runs):
        np.random.seed(seed)  # for reproducibility
        
        # split train_prompt_ids into train and validation sets (90:10 split)
        train_prompt_IDs_list = list(train_prompt_IDs)
        np.random.shuffle(train_prompt_IDs_list)
        split_idx = int(0.9 * len(train_prompt_IDs_list))
        train_prompt_IDs = set(train_prompt_IDs_list[:split_idx])
        val_prompt_IDs = set(train_prompt_IDs_list[split_idx:])
        logger.debug(f"Train prompts: {len(train_prompt_IDs)}, Val prompts: {len(val_prompt_IDs)}, Test prompts: {len(test_prompt_IDs)}")
        
        # prepare data for this iteration
        X, y, train_prompt_sent_ids = prepare_data(train_activations_dict, train_labels_dict)
        X_test, y_test, test_prompt_sent_ids = prepare_data(test_activations_dict, test_labels_dict)

        train_indices = [i for i, key in enumerate(train_prompt_sent_ids) if key.split('_')[0] in train_prompt_IDs]
        val_indices = [i for i, key in enumerate(train_prompt_sent_ids) if key.split('_')[0] in val_prompt_IDs]
        test_indices = [i for i, key in enumerate(test_prompt_sent_ids) if key.split('_')[0] in test_prompt_IDs]
        
        if args.sample_K and args.sample_K > 0:
            assert len(train_indices) >= args.sample_K, f"Not enough training samples. Required: {args.sample_K}, Available: {len(train_indices)}"
            np.random.shuffle(train_indices)
            train_indices = train_indices[:args.sample_K]
            logger.info(f"use {args.sample_K} data")
        else:
            logger.info("use all data")

        X_train = X[train_indices]
        X_val = X[val_indices]

        # convert scores to binary classes
        threshold = 0.5 
        y_train = (y[train_indices] >= threshold).astype(int)
        y_val = (y[val_indices] >= threshold).astype(int)
        y_test = (y_test[test_indices] >= threshold).astype(int)

        if args.pca:
            X_train, X_val, X_test = apply_pca(X_train, X_val, X_test)
        
        logger.debug(f"Training set: {len(X_train)} latents (Safe: {np.sum(y_train==0)}, Unsafe: {np.sum(y_train==1)})")
        logger.debug(f"Validation set: {len(X_val)} latents (Safe: {np.sum(y_val==0)}, Unsafe: {np.sum(y_val==1)})")
        logger.debug(f"Testing set: {len(X_test)} latents (Safe: {np.sum(y_test==0)}, Unsafe: {np.sum(y_test==1)})")

        ### train probes
        lr_weighted_f1, lr_pr_auc = train_logistic_regression(X_train, y_train, X_test, y_test)
        mlp_weighted_f1, mlp_pr_auc = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)
        final_lr_aucs.append(lr_pr_auc)
        final_mlp_aucs.append(mlp_pr_auc)
        final_lr_f1s.append(lr_weighted_f1)
        final_mlp_f1s.append(mlp_weighted_f1)

        ### random baseline
        # create a shuffled version of y_train while preserving class distribution
        np.random.seed(seed)  # use same seed as outer loop for reproducibility
        shuffled_indices = np.arange(len(y_train))
        np.random.shuffle(shuffled_indices)
        y_train_shuffled = y_train[shuffled_indices]
        # calculate disagreement between original and shuffled labels
        disagreement = np.sum(y_train != y_train_shuffled)
        disagreement_percentage = (disagreement / len(y_train)) * 100
        total_disagreement_percentage += disagreement_percentage
        logger.debug(f"Label disagreement after shuffling: {disagreement} of {len(y_train)} ({disagreement_percentage:.1f}%)")
        logger.debug(f"Original class distribution - Positive: {np.sum(y_train == 1)/len(y_train):.2%}, Negative: {np.sum(y_train == 0)/len(y_train):.2%}")
        logger.debug(f"Shuffled class distribution - Positive: {np.sum(y_train_shuffled == 1)/len(y_train_shuffled):.2%}, Negative: {np.sum(y_train_shuffled == 0)/len(y_train_shuffled):.2%}")

        rand_lr_weighted_f1, rand_lr_pr_auc = train_logistic_regression(X_train, y_train_shuffled, X_test, y_test)
        rand_mlp_weighted_f1, rand_mlp_pr_auc = train_mlp(X_train, y_train_shuffled, X_val, y_val, X_test, y_test)
        rand_lr_aucs.append(rand_lr_pr_auc)
        rand_mlp_aucs.append(rand_mlp_pr_auc)
        rand_lr_f1s.append(rand_lr_weighted_f1)
        rand_mlp_f1s.append(rand_mlp_weighted_f1)

    ### print results
    stats = calculate_metrics_stats(final_lr_aucs, final_mlp_aucs, rand_lr_aucs, rand_mlp_aucs,
                                    final_lr_f1s, final_mlp_f1s, rand_lr_f1s, rand_mlp_f1s)

    print("="*20)
    print(f"---- mean disagreement after shuffling: {total_disagreement_percentage/args.N_runs:.2f}%")
    print(f"probes: ")
    # print(f">>>> lr_auc_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['lr_mean']:.1f}% ± {stats['lr_std']:.1f}%")
    # print(f">>>> mlp_auc_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['mlp_mean']:.1f}% ± {stats['mlp_std']:.1f}%")
    print(f">>>> lr_f1_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['lr_f1_mean']:.1f}% ± {stats['lr_f1_std']:.1f}%")
    print(f">>>> mlp_f1_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['mlp_f1_mean']:.1f}% ± {stats['mlp_f1_std']:.1f}%")

    print(f"random baseline: ")
    # print(f">>>> random lr_auc_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['rand_lr_mean']:.1f}% ± {stats['rand_lr_std']:.1f}%")
    # print(f">>>> random mlp_auc_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['rand_mlp_mean']:.1f}% ± {stats['rand_mlp_std']:.1f}%")
    print(f">>>> random lr_f1_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['rand_lr_f1_mean']:.1f}% ± {stats['rand_lr_f1_std']:.1f}%")
    print(f">>>> random mlp_f1_score (N_train: {len(train_indices)}. N_test: {len(test_indices)})={stats['rand_mlp_f1_mean']:.1f}% ± {stats['rand_mlp_f1_std']:.1f}%")

    print("="*20)

if __name__ == "__main__":
    main()