"""
Train a BERT classifier on CoT texts to predict answer safety.

Input data:
    The preprocessed .jsonl data from 0_preprocess_paired.py

Output:
    Prints performance metrics (F1, accuracy, PR-AUC).
    Optionally saves detailed predictions and training data to TSV files.
"""
import collections
import fcntl
import os
import json
import random
import torch
import argparse
import pathlib
import math
import numpy as np
import traceback
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, set_seed
from torch.utils.data import DataLoader
from utils import eval_pred, add_to_final_scores, calculate_metrics_stats, save_probe_outputs_tsv
from loguru import logger
import sys

parser = argparse.ArgumentParser(description="Train a BERT classifier on text data.")
parser.add_argument("--input_folder", type=str, default=None, help="input folder containing .jsonl files")
parser.add_argument("--input_file", type=str, default=None, help="input .jsonl file containing data")
parser.add_argument("--N_runs", type=int, default=1, help="number of different seeded runs (default 1 for BERT)")
parser.add_argument("--run_seed", type=int, default=None, help="train only this seed (for parallel scheduling)")
parser.add_argument("--text_classifier_model", type=str, default="answerdotai/ModernBERT-large", help="name of the text classification model to use")
parser.add_argument("--sample_K", type=int, default=-1, help="number of training samples")
parser.add_argument("--store_outputs", action="store_true", help="whether to store model outputs")
parser.add_argument("--probe_output_folder", type=str, default="../probe_main-table_debug/BERT_model_ckpts/", help="folder to store model outputs and results")
parser.add_argument("--truncation_len", default=4096, type=int)

# [NEW] Batch Size & Gradient Accumulation
parser.add_argument("--train_bsz", default=2, type=int, help="Per device physical batch size")
parser.add_argument("--grad_accum", default=3, type=int, help="Gradient accumulation steps")

# [NEW] Logging Argument
parser.add_argument("--log_dir", type=str, default="./pipeline_logs", help="Directory to store per-dataset execution logs")
parser.add_argument("--checkpoint_dir", type=str, default="./", help="Directory to store HuggingFace model checkpoints and logs")
args = parser.parse_args()

if args.input_folder and args.input_file:
    logger.error("Please specify either --input_folder OR --input_file, not both.")
    sys.exit(1)
if not args.input_folder and not args.input_file:
    logger.error("Please specify at least one input: --input_folder or --input_file.")
    sys.exit(1)

MODEL_NAME = args.text_classifier_model


def merge_id_test_metrics(output_folder, dataset_name, seed_id, seed_metrics):
    """Atomically merge one completed seed into shared dataset summaries."""
    output_folder.mkdir(parents=True, exist_ok=True)
    metrics_path = output_folder / "id_test_metrics.json"
    summary_path = output_folder / "training_summary.txt"
    lock_path = output_folder / ".summary.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        payload = {"mean_std": {}, "seeds": {}}
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        payload.setdefault("seeds", {})[seed_id] = seed_metrics

        metric_names = sorted({
            metric
            for metrics in payload["seeds"].values()
            for metric in metrics
        })
        payload["mean_std"] = {}
        for metric in metric_names:
            values = [metrics[metric] for metrics in payload["seeds"].values() if metric in metrics]
            payload["mean_std"][metric] = f"{np.mean(values):.1f} ± {np.std(values):.1f}"

        tmp_metrics = metrics_path.with_suffix(".json.tmp")
        tmp_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_metrics, metrics_path)

        headers = ["f1", "accuracy", "pr_auc"]
        header = "Model           | " + " | ".join(f"{metric:>15}" for metric in headers)
        values = [payload["mean_std"].get(metric, "N/A") for metric in headers]
        summary = (
            f"Model: {MODEL_NAME}\n"
            f"Dataset: {dataset_name}\n"
            + "=" * 50 + "\n"
            + header + "\n"
            + "-" * len(header) + "\n"
            + f"`{MODEL_NAME}` | " + " | ".join(f"{value:>15}" for value in values) + "\n"
        )
        tmp_summary = summary_path.with_suffix(".txt.tmp")
        tmp_summary.write_text(summary, encoding="utf-8")
        os.replace(tmp_summary, summary_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def load_data_from_jsonl(file_path):
    labels = {}
    texts = {}
    prompts = {}
    cots = {}

    logger.info(f"Loading data from file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading lines")):
            try:
                data = json.loads(line)
                base_pair_id = str(data.get("pair_id", line_num))
                key = f"{base_pair_id}:::row{line_num}"

                labels[key] = float(data["hacking_label"])
                p = data.get("prompt", "")
                c = data.get("response", "")

                texts[key] = f"### Instruction:\n{p}\n\n### Response:\n{c}"
                prompts[key] = p
                cots[key] = c

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping line {line_num} due to error: {e}")

    return texts, prompts, cots, labels

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=min(tokenizer.model_max_length, args.truncation_len))
        self.labels = labels
        logger.warning(f"Truncated to {min(tokenizer.model_max_length, args.truncation_len)}")

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class PairedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, batch_size, keys_train, seed=0):
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        self.grouped_indices = collections.defaultdict(list)
        for i, key in enumerate(keys_train):
            self.grouped_indices[key].append(i)
        self.base_ids = sorted(list(self.grouped_indices.keys()))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.num_samples = int(math.ceil(len(self.base_ids) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = [self.base_ids[i] for i in torch.randperm(len(self.base_ids), generator=g).tolist()]
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]

        batch = []
        for base_id in indices:
            items = self.grouped_indices[base_id]
            if len(batch) + len(items) > self.batch_size and len(batch) > 0:
                yield batch
                batch = []
            batch.extend(items)

        if batch:
            yield batch

    def __len__(self):
        return int(math.ceil(self.num_samples / self.batch_size))

    def set_epoch(self, epoch):
        self.epoch = epoch

class PairedTrainer(Trainer):
    def __init__(self, keys_train, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keys_train = keys_train

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        batch_sampler = PairedBatchSampler(
            batch_size=self.args.train_batch_size,
            keys_train=self.keys_train,
            seed=self.args.seed
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# [FIX] Added 'seed' as a parameter so the model path gets named correctly!
def train_bert_classifier(texts_train, y_train, texts_val, y_val, texts_test, y_test, keys_train, dataset_name, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if "modernbert" in MODEL_NAME.lower():
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, attn_implementation="sdpa",reference_compile=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    train_dataset = TextDataset(texts_train, y_train, tokenizer)
    val_dataset = TextDataset(texts_val, y_val, tokenizer)
    test_dataset = TextDataset(texts_test, y_test, tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        f1 = f1_score(labels, predictions)
        return {'f1': f1}

    ckpt_base = pathlib.Path(args.checkpoint_dir)

    training_args = TrainingArguments(
        # [MODIFIED] Checkpoint and logs now explicitly include _seed{seed}
        output_dir=str(ckpt_base / f"{MODEL_NAME.split('/')[-1]}_{dataset_name}_seed{seed}_results"),
        logging_dir=str(ckpt_base / f"{MODEL_NAME.split('/')[-1]}_{dataset_name}_seed{seed}_logs"),
        num_train_epochs=15,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_bsz,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    trainer = PairedTrainer(
        keys_train=keys_train,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    y_pred_probs = torch.softmax(torch.from_numpy(predictions.predictions), dim=-1).numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred, y_pred_probs[:, 1]

def main():
    if args.input_folder:
        input_dir = pathlib.Path(args.input_folder)
        files_to_process = [f for f in input_dir.glob("*.jsonl") if "test" not in f.name.lower()]
    elif args.input_file:
        files_to_process = [pathlib.Path(args.input_file)]
    else:
        return

    if not files_to_process:
        logger.warning("No valid .jsonl files found to process.")
        return

    # Setup Logging Directory
    log_dir_path = pathlib.Path(args.log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # [NEW] Get the DDP Local Rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    for input_path in files_to_process:
        dataset_name = input_path.stem
        current_probe_output_folder = pathlib.Path(args.probe_output_folder) / dataset_name

        if local_rank == 0: logger.info(f"========== Processing Dataset: {dataset_name} ==========")

        seed_suffix = f"_seed{args.run_seed}" if args.run_seed is not None else ""
        log_file_path = log_dir_path / f"{dataset_name}{seed_suffix}_execution.log"
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # [NEW] ONLY Rank 0 writes to the log file. Other ranks are silenced to prevent terminal spam/corruption.
            if local_rank == 0:
                log_file = open(log_file_path, "w", encoding='utf-8')
                sys.stdout = log_file
                sys.stderr = log_file
            else:
                devnull = open(os.devnull, 'w')
                sys.stdout = devnull
                sys.stderr = devnull

            # --- Core Processing Logic ---
            texts_dict, prompts_dict, cots_dict, labels_dict = load_data_from_jsonl(input_path)

            if not texts_dict:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                if local_rank == 0: logger.warning(f"No valid data loaded from {dataset_name}. Skipping...")
                continue

            prompt_IDs = set(key.split(':::')[0] for key in cots_dict.keys())
            N = len(prompt_IDs)
            D_final_bert_scores = collections.defaultdict(list)
            id_test_metrics_by_seed = {}

            current_probe_output_folder = pathlib.Path(args.probe_output_folder) / dataset_name
            if args.store_outputs and local_rank == 0:
                current_probe_output_folder.mkdir(parents=True, exist_ok=True)

            seeds_to_run = [args.run_seed] if args.run_seed is not None else list(range(args.N_runs))
            for seed in seeds_to_run:
                seed_marker = current_probe_output_folder / f"training_summary_seed{seed}.txt"
                if seed_marker.exists():
                    print(f"⏭️ Skipping {dataset_name} seed{seed}: completion marker found.")
                    continue
                np.random.seed(seed)
                train_prompt_ids = set(np.random.choice(sorted(list(prompt_IDs)), int(0.7 * N), replace=False))
                test_prompt_ids = prompt_IDs - train_prompt_ids

                train_prompt_ids_list = sorted(train_prompt_ids)
                np.random.shuffle(train_prompt_ids_list)
                split_idx = int(0.9 * len(train_prompt_ids_list))
                train_prompt_ids = set(train_prompt_ids_list[:split_idx])
                val_prompt_ids = set(train_prompt_ids_list[split_idx:])
                print(f"[{dataset_name}] Train prompts: {len(train_prompt_ids)}, Val prompts: {len(val_prompt_ids)}, Test prompts: {len(test_prompt_ids)}")

                texts_list, labels_list, prompt_sent_ids = [], [], []
                common_ids = set(texts_dict.keys()) & set(labels_dict.keys())
                for id_ in sorted(common_ids):
                    texts_list.append(texts_dict[id_])
                    labels_list.append(labels_dict[id_])
                    prompt_sent_ids.append(id_)

                train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split(':::')[0] in train_prompt_ids]
                val_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split(':::')[0] in val_prompt_ids]
                test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split(':::')[0] in test_prompt_ids]

                if args.sample_K > 0 and len(train_indices) > args.sample_K:
                    np.random.shuffle(train_indices)
                    train_indices = train_indices[:args.sample_K]
                    print(f"using {args.sample_K} data.")

                texts_train = [texts_list[i] for i in train_indices]
                texts_val = [texts_list[i] for i in val_indices]
                texts_test = [texts_list[i] for i in test_indices]

                labels_np = np.array(labels_list)
                threshold = 0.5
                y_train = (labels_np[train_indices] >= threshold).astype(int)
                y_val = (labels_np[val_indices] >= threshold).astype(int)
                y_test = (labels_np[test_indices] >= threshold).astype(int)

                print("Flipping labels (0->1, 1->0) so unsafe: 0, safe (rarer): 1")
                y_train = 1 - y_train
                y_val = 1 - y_val
                y_test = 1 - y_test

                # Keep every response from one prompt in the same sampler unit.
                # The :::row suffix identifies a row, not a distinct pair.
                keys_train = [prompt_sent_ids[i].split(':::')[0] for i in train_indices]
                keys_val = [prompt_sent_ids[i] for i in val_indices]
                keys_test = [prompt_sent_ids[i] for i in test_indices]

                bert_y_pred, bert_y_pred_prob = train_bert_classifier(
                    texts_train, y_train, texts_val, y_val, texts_test, y_test, keys_train, dataset_name, seed
                )

                bert_eval = eval_pred(y_test, bert_y_pred, bert_y_pred_prob, metrics=["f1", "accuracy", "pr_auc"])
                add_to_final_scores(bert_eval, D_final_bert_scores, MODEL_NAME)
                id_test_metrics_by_seed[f"seed{seed}"] = {
                    metric: float(value) * 100.0 for metric, value in bert_eval.items()
                }

                # [NEW] ONLY Rank 0 saves TSV files to prevent 8 GPUs corrupting the same file
                if args.store_outputs and local_rank == 0:
                    test_text_prompts = [prompts_dict[key] for key in keys_test]
                    test_text_cots = [cots_dict[key] for key in keys_test]

                    formatted_keys_test = [k.replace(":::row", "_") for k in keys_test]

                    save_probe_outputs_tsv(
                        output_dir=current_probe_output_folder,
                        probe_name=f"{MODEL_NAME.split('/')[-1]}_seed{seed}",
                        prompt_sent_ids=formatted_keys_test,
                        prompts=test_text_prompts,
                        cots=test_text_cots,
                        true_labels=y_test,
                        pred_labels=bert_y_pred,
                        pred_probs=bert_y_pred_prob
                    )

                if local_rank == 0:
                    seed_metrics = id_test_metrics_by_seed[f"seed{seed}"]
                    merge_id_test_metrics(
                        current_probe_output_folder,
                        dataset_name,
                        f"seed{seed}",
                        seed_metrics,
                    )
                    seed_marker.write_text(
                        json.dumps({"seed": seed, "metrics": seed_metrics}, indent=2),
                        encoding="utf-8",
                    )

            print(f"\nStats for {dataset_name}:")
            final_stats_string = calculate_metrics_stats([D_final_bert_scores])
            print(final_stats_string)

            # --- On Success: Restore streams ---
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            if local_rank == 0:
                if log_file_path.exists():
                    os.remove(log_file_path)
                logger.info(f"✅ Successfully processed {dataset_name}.")

        except Exception as e:
            # --- On Crash: Restore streams, write traceback, AND RAISE THE ERROR ---
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            if local_rank == 0:
                with open(log_file_path, "a", encoding='utf-8') as log_file:
                    log_file.write("\n\n" + "="*50 + "\n")
                    log_file.write("CRASH TRACEBACK:\n")
                    traceback.print_exc(file=log_file)
                logger.error(f"CRASH in {dataset_name}. Check log: {log_file_path}")

            # [NEW] We MUST raise the exception so torchrun knows it failed and stops the Bash script!
            raise e

if __name__ == "__main__":
    main()
