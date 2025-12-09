"""
Train a BERT classifier on CoT texts to predict answer safety.

Input data:
    The same labeled data as the probe training scripts.

Output:
    Prints performance metrics (F1, accuracy, PR-AUC).
    Optionally saves detailed predictions and training data to TSV files.
"""
import collections
import os
import json
import torch
import argparse
import pathlib
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_recall_curve, auc, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from utils import eval_pred, add_to_final_scores, calculate_metrics_stats, save_probe_outputs_tsv
from loguru import logger

parser = argparse.ArgumentParser(description="Train a BERT classifier on text data.")
parser.add_argument("--input_folder", type=str, default=None, help="input folder containing activations and labels")
parser.add_argument("--input_file", type=str, default=None, help="input .jsonl file containing data")
parser.add_argument("--N_runs", type=int, default=1, help="number of different seeded runs")
parser.add_argument("--text_classifier_model", type=str, default="bert-base-uncased", help="name of the text classification model to use")
parser.add_argument("--sample_K", type=int, default=-1, help="number of training samples")
parser.add_argument("--store_outputs", action="store_true", help="whether to store model outputs")
parser.add_argument("--probe_output_folder", type=str, default="../probe_outputs/", help="folder to store model outputs and results")
parser.add_argument("--truncation_len", default=8192, type=int)
parser.add_argument("--train_bsz", default=16, type=int)

args = parser.parse_args()

if args.input_folder and args.input_file:
    logger.error("Please specify either --input_folder OR --input_file, not both.")
    sys.exit(1)
if not args.input_folder and not args.input_file:
    logger.error("Please specify at least one input: --input_folder or --input_file.")
    sys.exit(1)

if args.input_folder:
    INPUT_PATH = pathlib.Path(args.input_folder)
    output_subdir_name = INPUT_PATH.name
elif args.input_file:
    INPUT_PATH = pathlib.Path(args.input_file)
    # Requirement 2: output directory name is "my_" + filename
    output_subdir_name = "my_" + INPUT_PATH.stem

if args.store_outputs:
    PROBE_OUTPUT_FOLDER = pathlib.Path(args.probe_output_folder) / output_subdir_name

MODEL_NAME = args.text_classifier_model

def load_data_from_folder():
    """load activations and labels from the input folder"""
    labels = {}
    texts = {}
    prompts = {}
    cots = {}

    for label_file in tqdm((INPUT_FOLDER / "labels").rglob("*.json"), desc="Loading labels and texts"):
        if label_file.name.endswith("_labeled.json"):
            key = '_'.join(label_file.stem.split('_')[:2])
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "safety_label" in data and "score" in data["safety_label"]:
                    labels[key] = data["safety_label"]["score"]
                    texts[key] = data.get("prompt", "") + data.get("cot", "")
                    prompts[key] = data.get("prompt", "")
                    cots[key] = data.get("cot", "")
    return texts, prompts, cots, labels

def _load_data_from_file():
    """New loading logic for single .jsonl file input"""
    labels = {}
    texts = {}
    prompts = {}
    cots = {}
    
    logger.info(f"Loading data from file: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading lines")):
            try:
                data = json.loads(line)
                key = str(line_num)

                if "hacking_label" in data:
                    labels[key] = float(data["hacking_label"])
                if data.get("traj_source") == "persona":
                    # Prompt: messages[1].content.parts[0]
                    p = data["messages"][1]["content"]["parts"][0]
                    # Response (CoT+Answer): messages[2].content.parts[0]
                    c = data["messages"][2]["content"]["parts"][0]
                else:
                    # Old format extraction (Default)
                    p = data.get("prompt", [{}])[0].get("content", "")
                    c = data.get("response", "")
                
                texts[key] = p + c
                prompts[key] = p
                cots[key] = c
                
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                logger.warning(f"Skipping line {line_num} due to error: {e}")
                
    return texts, prompts, cots, labels


def load_data():
    if args.input_folder:
        return load_data_from_folder()
    elif args.input_file:
        return _load_data_from_file()

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

def train_bert_classifier(texts_train, y_train, texts_val, y_val, texts_test, y_test):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if "modernbert" in MODEL_NAME.lower():
        # https://huggingface.co/answerdotai/ModernBERT-base/discussions/14
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, attn_implementation="eager",reference_compile=False)
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

    training_args = TrainingArguments(
        output_dir=f"./{MODEL_NAME.split('/')[-1]}_results",
        num_train_epochs=50,
        per_device_train_batch_size=args.train_bsz,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./{MODEL_NAME.split('/')[-1]}_logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        save_total_limit = 1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # test 
    predictions = trainer.predict(test_dataset)
    y_pred_probs = torch.softmax(torch.from_numpy(predictions.predictions), dim=-1).numpy()
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred, y_pred_probs[:, 1]

def main():
    texts_dict, prompts_dict, cots_dict, labels_dict = load_data()
    prompt_IDs = set(key.split('_')[0] for key in cots_dict.keys())
    N = len(prompt_IDs)

    D_final_bert_scores = collections.defaultdict(list)
    for seed in range(args.N_runs):
        np.random.seed(seed)  # for reproducibility
        train_prompt_ids = set(np.random.choice(sorted(list(prompt_IDs)), int(0.7 * N), replace=False))
        test_prompt_ids = prompt_IDs - train_prompt_ids

        # split train_prompt_ids into train and validation sets (90:10 split)
        train_prompt_ids_list = list(train_prompt_ids)
        np.random.shuffle(train_prompt_ids_list)
        split_idx = int(0.9 * len(train_prompt_ids_list))
        train_prompt_ids = set(train_prompt_ids_list[:split_idx])
        val_prompt_ids = set(train_prompt_ids_list[split_idx:])
        print(f"Train prompts: {len(train_prompt_ids)}, Val prompts: {len(val_prompt_ids)}, Test prompts: {len(test_prompt_ids)}")
        
        texts_list, labels_list, prompt_sent_ids = [], [], []
        common_ids = set(texts_dict.keys()) & set(labels_dict.keys())
        for id_ in common_ids:
            texts_list.append(texts_dict[id_])
            labels_list.append(labels_dict[id_])
            prompt_sent_ids.append(id_)

        train_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in train_prompt_ids]
        val_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in val_prompt_ids]
        test_indices = [i for i, key in enumerate(prompt_sent_ids) if key.split('_')[0] in test_prompt_ids]

        # subsample training data if K is specified
        if args.sample_K > 0 and len(train_indices) > args.sample_K:
            np.random.shuffle(train_indices)
            train_indices = train_indices[:args.sample_K]
            logger.warning(f"using {args.sample_K} data.")
        
        texts_train = [texts_list[i] for i in train_indices]
        texts_val = [texts_list[i] for i in val_indices]
        texts_test = [texts_list[i] for i in test_indices]
        
        labels_np = np.array(labels_list)
        # convert scores to binary classes
        threshold = 0.5
        y_train = (labels_np[train_indices] >= threshold).astype(int) 
        y_val = (labels_np[val_indices] >= threshold).astype(int)
        y_test = (labels_np[test_indices] >= threshold).astype(int)

        # LABELS => unsafe: 0, safe (rarer): 1
        logger.info("Flipping labels (0->1, 1->0) so unsafe: 0, safe (rarer): 1")
        y_train = 1 - y_train
        y_val = 1 - y_val
        y_test = 1 - y_test

        keys_train = [prompt_sent_ids[i] for i in train_indices]
        keys_val = [prompt_sent_ids[i] for i in val_indices]
        keys_test = [prompt_sent_ids[i] for i in test_indices]
        
        bert_y_pred, bert_y_pred_prob = train_bert_classifier(texts_train, y_train, texts_val, y_val, texts_test, y_test)
        bert_eval = eval_pred(y_test, bert_y_pred, bert_y_pred_prob, metrics=["f1", "accuracy", "pr_auc"])
        add_to_final_scores(bert_eval, D_final_bert_scores, MODEL_NAME)

        if args.store_outputs:
            test_text_prompts = [prompts_dict[key] for key in keys_test]
            test_text_cots = [cots_dict[key] for key in keys_test]

            save_probe_outputs_tsv(
                output_dir=PROBE_OUTPUT_FOLDER,
                probe_name=f"{MODEL_NAME}_seed{seed}",
                prompt_sent_ids=keys_test,
                prompts=test_text_prompts,
                cots=test_text_cots,
                true_labels=y_test,
                pred_labels=bert_y_pred,
                pred_probs=bert_y_pred_prob
            )

    print(calculate_metrics_stats([
        D_final_bert_scores
    ]))


if __name__ == "__main__":
    main()