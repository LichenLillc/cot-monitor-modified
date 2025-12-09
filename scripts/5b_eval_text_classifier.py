"""
Evaluate a pre-trained BERT classifier (from 3b_text_classifier.py) 
on a new custom .jsonl dataset.

This script does the following:
1. Loads the pre-trained model and tokenizer from a checkpoint.
2. Loads the custom .jsonl data, extracting text and labels.
3. Performs label transformation (flipping) to match the model's training.
4. Runs batched inference to get predictions and probabilities.
5. Saves detailed per-item results to a .jsonl file.
6. Calculates dual-class metrics (Safe & Unsafe as positive) for:
   - Precision, Recall, F1
   - PR-AUC, AUC-ROC
7. Saves aggregate metrics and unsafe ratios to a .csv file.
8. Generates and saves PR and ROC curves plotting both classes.
9. Saves all outputs into a subdirectory named after the input file suffix.
"""

import json
import argparse
import pathlib
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_curve, auc,
    precision_score, recall_score, roc_curve, roc_auc_score
)
from loguru import logger
import matplotlib.pyplot as plt
import os

def eval_dual_class_metrics(y_true, y_pred, y_prob_safe, y_prob_unsafe):
    """
    Calculates a full suite of metrics for both 'safe' (1) and 'unsafe' (0) 
    as the positive class.
    """
    metrics = {}
    
    # --- Metrics for 'Safe' (pos_label=1) ---
    metrics['precision_safe'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_safe'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_safe'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    precision_s, recall_s, _ = precision_recall_curve(y_true, y_prob_safe, pos_label=1)
    metrics['pr_auc_safe'] = auc(recall_s, precision_s)
    
    fpr_s, tpr_s, _ = roc_curve(y_true, y_prob_safe, pos_label=1)
    metrics['roc_auc_safe'] = auc(fpr_s, tpr_s)

    # --- Metrics for 'Unsafe' (pos_label=0) ---
    metrics['precision_unsafe'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['recall_unsafe'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1_unsafe'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    precision_u, recall_u, _ = precision_recall_curve(y_true, y_prob_unsafe, pos_label=0)
    metrics['pr_auc_unsafe'] = auc(recall_u, precision_u)
    
    fpr_u, tpr_u, _ = roc_curve(y_true, y_prob_unsafe, pos_label=0)
    metrics['roc_auc_unsafe'] = auc(fpr_u, tpr_u)
    
    # --- Overall Accuracy ---
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    return metrics

def plot_curves(y_true, y_prob_safe, y_prob_unsafe, save_path_pr, save_path_roc):
    """
    Generates and saves PR and ROC curves, plotting both classes.
    """
    # --- 1. PR Curve ---
    plt.figure()
    precision_s, recall_s, _ = precision_recall_curve(y_true, y_prob_safe, pos_label=1)
    auc_pr_s = auc(recall_s, precision_s)
    plt.plot(recall_s, precision_s, label=f'Safe (1) as Positive (PR-AUC = {auc_pr_s:.3f})')
    
    precision_u, recall_u, _ = precision_recall_curve(y_true, y_prob_unsafe, pos_label=0)
    auc_pr_u = auc(recall_u, precision_u)
    plt.plot(recall_u, precision_u, label=f'Unsafe (0) as Positive (PR-AUC = {auc_pr_u:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(save_path_pr)
    plt.close()
    logger.info(f"PR curve saved to {save_path_pr}")

    # --- 2. ROC Curve ---
    plt.figure()
    fpr_s, tpr_s, _ = roc_curve(y_true, y_prob_safe, pos_label=1)
    auc_roc_s = auc(fpr_s, tpr_s)
    plt.plot(fpr_s, tpr_s, label=f'Safe (1) as Positive (ROC-AUC = {auc_roc_s:.3f})')
    
    fpr_u, tpr_u, _ = roc_curve(y_true, y_prob_unsafe, pos_label=0)
    auc_roc_u = auc(fpr_u, tpr_u)
    plt.plot(fpr_u, tpr_u, label=f'Unsafe (0) as Positive (ROC-AUC = {auc_roc_u:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess') # Dashed diagonal
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(save_path_roc)
    plt.close()
    logger.info(f"ROC curve saved to {save_path_roc}")


def load_custom_data(input_file_path):
    """
    Loads the custom .jsonl dataset based on user's A1 and A2.
    """
    texts_to_classify = []
    true_labels_for_eval = [] # Flipped labels (0=unsafe, 1=safe)
    traj_ids = []
    hacking_types = []

    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading custom data"):
            try:
                data = json.loads(line)
                
                # A1: Extract text (prompt + response)
                if data.get("traj_source") == "persona":
                    prompt = data['messages'][1]['content']['parts'][0]
                    response = data['messages'][2]['content']['parts'][0]
                else:
                    # Default/Old format
                    prompt = data['prompt'][0]['content']
                    response = data['response']
                
                text = prompt + response
                texts_to_classify.append(text)

                # A2: Extract metadata
                traj_ids.append(data.get('traj_id', f"line_{len(traj_ids)}"))
                hacking_types.append(data['hacking_type'])

                # A3: Extract and flip labels
                # Original: 0=safe, 1=unsafe
                hacking_label = data['hacking_label']
                
                # Flipped: 0=unsafe, 1=safe
                # This matches the label space 3b_text_classifier.py used for training
                true_label = 1 - hacking_label
                true_labels_for_eval.append(true_label)
                
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line: {line}")
            except KeyError as e:
                logger.warning(f"Skipping line with missing key {e}: {line}")

    return texts_to_classify, true_labels_for_eval, traj_ids, hacking_types

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained BERT classifier on new data.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (e.g., './ModernBERT-large_results/checkpoint-1200')")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the new .jsonl dataset file (e.g., '.../merge_traj_n150_u75_e75.jsonl')")
    
    # --- START: MODIFICATION 1 (File Naming) ---
    parser.add_argument("--output_dir", type=str, default="../5b_results/", help="Directory to save all outputs")
    # Removed --output_jsonl and --output_csv
    # --- END: MODIFICATION 1 ---
    
    parser.add_argument("--truncation_len", type=int, default=8192, help="Truncation length. Should match the one used during training (e.g., 512)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation (can be larger than training)")
    parser.add_argument("--tokenizer_name", type=str, default="answerdotai/ModernBERT-large", help="Name or path of the base model to load tokenizer from (e.g., 'answerdotai/ModernBERT-large')")
    args = parser.parse_args()

    # --- START: MODIFICATION 2 (Auto-naming logic) ---
    # Extract suffix from input file name
    input_path = pathlib.Path(args.input_file)
    input_stem = input_path.stem # e.g., "merge_traj_n150_u75_e75"
    
    if input_stem.startswith("merge_traj_"):
        file_suffix = input_stem[len("merge_traj_"):] # e.g., "n150_u75_e75"
    else:
        file_suffix = input_stem # Fallback to full stem
        
    logger.info(f"Using output subdirectory and file prefix: {file_suffix}")

    # Create new subdirectory
    output_dir = pathlib.Path(args.output_dir) / file_suffix # e.g., ./5b_results/n150_u75_e75
    os.makedirs(output_dir, exist_ok=True)

    # Define all output paths (filenames remain unchanged, using the suffix)
    output_jsonl_path = output_dir / f"{file_suffix}.jsonl"
    output_csv_path = output_dir / f"{file_suffix}.csv"
    output_pr_plot_path = output_dir / f"{file_suffix}_PRAUC.png"
    output_roc_plot_path = output_dir / f"{file_suffix}_AUCROC.png"
    # --- END: MODIFICATION 2 ---

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- B. Load Model and Tokenizer ---
    logger.info(f"Loading tokenizer from base model: {args.tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False) # Added use_fast=False for stability
    logger.info(f"Loading model weights from {args.model_checkpoint_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint_path)
    model.to(device)
    model.eval() # Set model to evaluation mode
    logger.info("Model loaded successfully.")

    # --- C. Load and Process New Data ---
    texts, y_true, traj_ids, hacking_types = load_custom_data(args.input_file)
    y_true = np.array(y_true)

    # --- D. Batched Prediction (Inference) ---
    logger.info(f"Running predictions on {len(texts)} samples...")
    all_preds = []
    all_probs = []

    # Determine max length, same as in 3b_text_classifier.py
    try:
        model_max_len = tokenizer.model_max_length
    except AttributeError:
        model_max_len = 1e30 # Set a large number if not found
    
    max_len = min(model_max_len, args.truncation_len)
    logger.info(f"Using max_length={max_len} (min(model_max: {model_max_len}, arg: {args.truncation_len}))")


    for i in tqdm(range(0, len(texts), args.batch_size), desc="Predicting"):
        batch_texts = texts[i : i + args.batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_len
        ).to(device)

        with torch.no_grad(): # Disable gradient calculation
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_pred = np.array(all_preds)
    all_probs_np = np.array(all_probs)
    
    # Get probabilities for BOTH classes
    y_pred_prob_safe = all_probs_np[:, 1] # Prob of class 1 ("safe")
    y_pred_prob_unsafe = all_probs_np[:, 0] # Prob of class 0 ("unsafe")

    # --- E. Save and Evaluate ---

    # A3: Output 1 - Detailed .jsonl
    logger.info(f"Saving detailed results to {output_jsonl_path}...")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for i in range(len(texts)):
            result_item = {
                'traj_id': traj_ids[i],
                'hacking_type': hacking_types[i],
                'text_classified': texts[i],
                'golden_label': int(y_true[i]),      # 0=unsafe, 1=safe
                'predicted_label': int(y_pred[i]), # 0=unsafe, 1=safe
                'probability_safe': float(y_pred_prob_safe[i]),
                'probability_unsafe': float(y_pred_prob_unsafe[i])
            }
            f.write(json.dumps(result_item) + '\n')

    # A3: Output 2 - Summary .csv
    logger.info("Calculating metrics...")
    
    # --- START: MODIFICATION 3 (Metrics Calculation) ---
    metrics = eval_dual_class_metrics(y_true, y_pred, y_pred_prob_safe, y_pred_prob_unsafe)
    
    logger.info("Calculating unsafe ratios per hacking type...")
    type_preds = {}
    for htype, pred in zip(hacking_types, y_pred):
        if htype not in type_preds:
            type_preds[htype] = []
        type_preds[htype].append(pred)
    
    type_metrics = {}
    for htype, preds_list in type_preds.items():
        if not preds_list: continue
        preds_array = np.array(preds_list)
        # y_pred is 0=unsafe, 1=safe. User wants unsafe ratio.
        unsafe_count = (preds_array == 0).sum()
        total_count = len(preds_array)
        unsafe_ratio = unsafe_count / total_count
        metric_key = f'unsafe_ratio_{htype.replace(" ", "_").replace("-", "_").lower()}'
        type_metrics[metric_key] = unsafe_ratio
    
    metrics.update(type_metrics)

    logger.info("Converting metrics to 100-point (percentage) scale...")
    metrics_percent = {}
    for key, value in metrics.items():
        metrics_percent[key] = value * 100.0
    # --- END: MODIFICATION 3 ---

    logger.info(f"Metrics: {metrics_percent}")
    logger.info(f"Saving summary metrics to {output_csv_path}...")
    
    df_metrics = pd.DataFrame([metrics_percent])
    df_metrics.to_csv(output_csv_path, index=False)

    # --- START: MODIFICATION 4 (Plotting) ---
    logger.info("Generating plots...")
    plot_curves(
        y_true, 
        y_pred_prob_safe, 
        y_pred_prob_unsafe, 
        output_pr_plot_path, 
        output_roc_plot_path
    )
    # --- END: MODIFICATION 4 ---
    
    logger.success(f"Evaluation complete. Outputs saved to {output_dir.resolve()}")

if __name__ == "__main__":
    main()