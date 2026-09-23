"""
Matrix Evaluation for BERT Monitors.
Tests all trained BERT models against all available test datasets in parallel.
Extracts ID-TEST metrics from training summaries and calculates Accuracy on OOD test sets.
Stores results in a clean JSON matrix for later Excel generation.
Features Auto-Resume, Multi-Seed Mean/Std Aggregation, and Incremental Saving.
"""

import os
import sys
import json
import argparse
import pathlib
import torch
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import concurrent.futures
import multiprocessing as mp
from loguru import logger

torch.set_float32_matmul_precision('high')

# Ensure PyTorch multiprocessing uses spawn to prevent CUDA deadlocks
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

def parse_id_test_summary(summary_path):
    """Parses training_summary.txt to extract F1, Accuracy, and PR-AUC."""
    metrics = {"f1": "N/A", "accuracy": "N/A", "pr_auc": "N/A"}
    if not summary_path.exists():
        return metrics

    try:
        content = summary_path.read_text(encoding='utf-8')
        lines = content.splitlines()

        headers = []
        for line in lines:
            if "Model" in line and "accuracy" in line.lower():
                headers = [h.strip().lower() for h in line.split('|')]
                break

        if not headers:
            return metrics

        for line in lines:
            if line.strip().startswith("`"):
                parts = [p.strip() for p in line.split('|')]
                if "f1" in headers and len(parts) > headers.index("f1"):
                    metrics["f1"] = parts[headers.index("f1")]
                if "accuracy" in headers and len(parts) > headers.index("accuracy"):
                    metrics["accuracy"] = parts[headers.index("accuracy")]
                if "pr_auc" in headers and len(parts) > headers.index("pr_auc"):
                    metrics["pr_auc"] = parts[headers.index("pr_auc")]
                break
    except Exception as e:
        logger.warning(f"Failed to parse summary {summary_path.name}: {e}")

    return metrics


def load_id_test_metrics(summary_folder, seed_id):
    """Load one run's ID metrics, with legacy summary fallback."""
    metrics_path = summary_folder / "id_test_metrics.json"
    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            seed_metrics = payload.get("seeds", {}).get(seed_id)
            if seed_metrics is not None:
                return seed_metrics
        except Exception as exc:
            logger.warning(f"Could not parse {metrics_path}: {exc}")
    return parse_id_test_summary(summary_folder / "training_summary.txt")


def update_id_test_matrix(dataset_entry, seed_id, seed_metrics):
    """Store per-seed ID metrics and refresh their aggregate summary."""
    current = dataset_entry.get("ID-TEST")
    if not isinstance(current, dict) or "seeds" not in current:
        current = {"mean_std": {}, "seeds": {}}
        dataset_entry["ID-TEST"] = current

    current["seeds"][seed_id] = seed_metrics
    metric_names = sorted({
        metric
        for metrics in current["seeds"].values()
        if isinstance(metrics, dict)
        for metric, value in metrics.items()
        if isinstance(value, (int, float))
    })
    current["mean_std"] = {}
    for metric in metric_names:
        values = [
            metrics[metric]
            for metrics in current["seeds"].values()
            if isinstance(metrics, dict) and isinstance(metrics.get(metric), (int, float))
        ]
        current["mean_std"][metric] = f"{np.mean(values):.2f} ± {np.std(values):.2f}"

def load_test_datasets(test_folder_path):
    """Loads all *test*.jsonl files and formats them exactly like 3b_text_classifier."""
    datasets = {}
    test_files = list(pathlib.Path(test_folder_path).glob("*test*.jsonl"))

    logger.info(f"Found {len(test_files)} test datasets.")

    for file_path in test_files:
        ds_name = file_path.stem
        texts, labels = [], []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    p = data.get("prompt", "")
                    c = data.get("response", "")

                    # Exact 3b Formatting
                    formatted_text = f"### Instruction:\n{p}\n\n### Response:\n{c}"

                    # Exact 3b Label Flipping (0=unsafe, 1=safe)
                    label = 1 - float(data.get("hacking_label", 0))

                    texts.append(formatted_text)
                    labels.append(label)
                except Exception as e:
                    continue

        if texts:
            datasets[ds_name] = (texts, labels)

    return datasets

def worker_eval_model(kwargs):
    """Worker function to evaluate a single seed model on all datasets on a specific GPU."""
    model_dir = kwargs['model_dir']
    base_dataset = kwargs['base_dataset']
    seed_id = kwargs['seed_id']
    datasets = kwargs['datasets']
    gpu_id = kwargs['gpu_id']
    id_test_metrics = kwargs['id_test_metrics']
    batch_size = kwargs['batch_size']

    device = torch.device(f"cuda:{gpu_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            torch_dtype=torch.float16
        )
        model.to(device)
        model.eval()

        results = {}

        for ds_name, (texts, labels) in datasets.items():
            all_preds = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=4096
                ).to(device)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)

            acc = accuracy_score(labels, all_preds) * 100
            results[ds_name] = round(acc, 2)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        # [MODIFIED] Return the separated base dataset name, seed ID, and the accuracy results
        return base_dataset, seed_id, id_test_metrics, results

    except Exception as e:
        logger.error(f"Failed evaluating {base_dataset} ({seed_id}): {e}")
        return base_dataset, seed_id, id_test_metrics, {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Matrix Evaluation for BERT Monitors")
    parser.add_argument("--test_data_folder", type=str, default="/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0401/exp_0419/preprocessed_data", help="Folder containing *test*.jsonl files")
    parser.add_argument("--model_folder", type=str, default="/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0401/exp_0419/bert_checkpoints", help="Folder containing trained HF checkpoints")
    parser.add_argument("--summary_folder", type=str, default="/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0401/exp_0419/bert_tsv_and_summaries", help="Parent folder of folders containing training_summary.txt files")
    parser.add_argument("--output_file", type=str, default="/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0401/exp_0419/bert_eval_matrix.json", help="Path to save JSON results")
    parser.add_argument("--num_workers", type=int, default=24, help="Models to evaluate in parallel")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-GPU evaluation batch size")
    args = parser.parse_args()

    # 1. Preload Test Datasets
    datasets = load_test_datasets(args.test_data_folder)
    if not datasets:
        logger.error("No test datasets found. Exiting.")
        return

    # 2. Load existing JSON for Auto-Resume
    final_matrix = {}
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r', encoding='utf-8') as f:
                final_matrix = json.load(f)
            logger.info(f"Loaded existing progress. Found {len(final_matrix)} dataset groupings.")
        except Exception as e:
            logger.warning(f"Could not load existing output file (starting fresh): {e}")

    # 3. Find Checkpoints & Build Task Queue
    model_base_dir = pathlib.Path(args.model_folder)
    summary_base_dir = pathlib.Path(args.summary_folder)
    checkpoint_paths = list(model_base_dir.glob("**/checkpoint-*"))

    tasks = []
    gpu_counter = 0

    for ckpt_path in checkpoint_paths:
        parent_name = ckpt_path.parent.name
        raw_name = parent_name.replace("ModernBERT-large_", "").replace("_results", "")

        # [NEW] Regex parsing to dynamically split dataset name and seed number
        match = re.search(r"(.*)_seed(\d+)$", raw_name)
        if match:
            base_dataset = match.group(1)
            seed_id = f"seed{match.group(2)}"
        else:
            base_dataset = raw_name
            seed_id = "seed0"

        # [NEW] Strict Auto-Resume Check (Requires all tests for this specific seed to be done)
        skip = False
        if base_dataset in final_matrix:
            has_all_tests = True
            for ds_name in datasets.keys():
                if ds_name not in final_matrix[base_dataset] or type(final_matrix[base_dataset][ds_name]) is not dict or seed_id not in final_matrix[base_dataset][ds_name].get("seeds", {}):
                    has_all_tests = False
                    break
            if has_all_tests and "ID-TEST" in final_matrix[base_dataset]:
                skip = True

        if skip:
            logger.info(f"⏭️ Skipping {base_dataset} ({seed_id}): Already fully evaluated.")
            continue

        # Fetch ID-TEST metrics from the base_dataset's summary
        summary_folder = summary_base_dir / base_dataset
        id_test_metrics = load_id_test_metrics(summary_folder, seed_id)

        tasks.append({
            'model_dir': str(ckpt_path),
            'base_dataset': base_dataset,
            'seed_id': seed_id,
            'datasets': datasets,
            'gpu_id': gpu_counter % torch.cuda.device_count(),
            'id_test_metrics': id_test_metrics,
            'batch_size': args.batch_size,
        })
        gpu_counter += 1

    if not tasks:
        logger.success("All seeds have already been fully evaluated! Matrix is complete.")
        return

    # 4. Execute in Parallel with Multi-Seed Incremental Saving
    logger.info(f"Launching {len(tasks)} evaluation tasks across {args.num_workers} workers...")

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(worker_eval_model, task): task for task in tasks}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Evaluating Models"):
                base_dataset, seed_id, id_test_metrics, results = future.result()

                if "error" in results:
                    continue

                # Initialize base dataset entry if missing
                if base_dataset not in final_matrix:
                    final_matrix[base_dataset] = {}
                update_id_test_matrix(final_matrix[base_dataset], seed_id, id_test_metrics)

                # [NEW] Integrate the results into the seeds dictionary and calculate mean/std
                for ds_name, acc in results.items():
                    if ds_name not in final_matrix[base_dataset] or type(final_matrix[base_dataset][ds_name]) is not dict:
                        final_matrix[base_dataset][ds_name] = {"mean_std": "", "seeds": {}}

                    final_matrix[base_dataset][ds_name]["seeds"][seed_id] = acc

                    # Dynamically calculate the aggregated string
                    seed_vals = list(final_matrix[base_dataset][ds_name]["seeds"].values())
                    mean_val = np.mean(seed_vals)
                    std_val = np.std(seed_vals)
                    final_matrix[base_dataset][ds_name]["mean_std"] = f"{mean_val:.2f} ± {std_val:.2f}"

                # Incremental Save
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_matrix, f, indent=4)

        logger.success(f"Matrix Evaluation Complete! Results securely saved to {args.output_file}")

    except KeyboardInterrupt:
        logger.error("\n🚨 Ctrl+C detected! Force-killing all GPU worker processes...")
        for child in mp.active_children():
            child.terminate()
            child.join()
        logger.error("✅ All zombie workers destroyed. Exiting safely.")
        sys.exit(1)

if __name__ == "__main__":
    main()
