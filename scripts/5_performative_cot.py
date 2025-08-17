"""
Analyze performative CoT sentences.

For each test prompt with k CoT sentences:
1. Check if ground truth labels are stable (after first m sentences, >n% same label).
2. Check alignment between ground truth and predicted labels when ground truth stabilizes.
"""

import collections
from loguru import logger
import os
import json
import argparse
import pathlib
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Analyze CoT sentence label stability and alignment")
parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing labels")
parser.add_argument("--output_folder", type=str, default="stability_analysis_results", help="Output folder for results")
parser.add_argument("--min_sentences", type=int, default=10, help="Minimum sentences before checking stability")
parser.add_argument("--stability_threshold", type=float, default=0.8, help="Threshold for label stability")
parser.add_argument("--misalignment_threshold", type=float, default=0.5, help="Misalignment threshold")
parser.add_argument("--save_prompt_ids", type=str, help="JSON file to save high misalignment prompt IDs")

args = parser.parse_args()
INPUT_FOLDER = pathlib.Path(args.input_folder)
OUTPUT_FOLDER = pathlib.Path(args.output_folder)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def load_labels():
    """
    Load ground truth and predicted labels for all CoT sentences.
    Returns dict with structure: {prompt_id: {sent_id: {'gt': score, 'pred': score}}}
    """
    labels_data = collections.defaultdict(dict)
    labels_folder = INPUT_FOLDER / "labels" if (INPUT_FOLDER / "labels").exists() else INPUT_FOLDER
    
    # Load ground truth labels from *_labeled.json files
    for label_file in tqdm(labels_folder.rglob("*_labeled.json"), desc="Loading ground truth labels"):
        if not label_file.name.endswith("_labeled_cot.json") and not label_file.name.endswith("_labeled_cot_para.json"):
            filename = label_file.stem
            parts = filename.split('_')
            if len(parts) >= 3 and parts[-1] == "labeled":
                prompt_id = parts[0]
                sent_id = int(parts[1])
                
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "safety_label" in data and "score" in data["safety_label"]:
                        if prompt_id not in labels_data:
                            labels_data[prompt_id] = {}
                        if sent_id not in labels_data[prompt_id]:
                            labels_data[prompt_id][sent_id] = {}
                        labels_data[prompt_id][sent_id]['gt'] = data["safety_label"]["score"]
    
    # Load predicted labels from *_labeled_cot.json files
    for label_file in tqdm(labels_folder.rglob("*_labeled_cot.json"), desc="Loading predicted labels"):
        if not label_file.name.endswith("_labeled_cot_para.json"):
            filename = label_file.stem
            parts = filename.split('_')
            if len(parts) >= 4 and parts[-2:] == ["labeled", "cot"]:
                prompt_id = parts[0]
                sent_id = int(parts[1])
                
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "safety_label" in data and "score" in data["safety_label"]:
                        if prompt_id not in labels_data:
                            labels_data[prompt_id] = {}
                        if sent_id not in labels_data[prompt_id]:
                            labels_data[prompt_id][sent_id] = {}
                        labels_data[prompt_id][sent_id]['pred'] = data["safety_label"]["score"]
    
    # Filter to only include prompts with both GT and pred labels
    filtered_data = {}
    for prompt_id, sentences in labels_data.items():
        filtered_sentences = {}
        for sent_id, labels in sentences.items():
            if 'gt' in labels and 'pred' in labels:
                filtered_sentences[sent_id] = labels
        if filtered_sentences:
            filtered_data[prompt_id] = filtered_sentences
    
    logger.info(f"Loaded labels for {len(filtered_data)} prompts")
    total_sentences = sum(len(sentences) for sentences in filtered_data.values())
    logger.info(f"Total sentences with both GT and pred labels: {total_sentences}")
    
    return filtered_data

def convert_to_binary(score, threshold=0.5):
    return 1 if score >= threshold else 0

def check_stability(labels, min_sentences=10, threshold=0.8):
    """
    Check if labels stabilize after min_sentences.
    Returns (is_stable, stable_label, first_stable_idx) where:
    - is_stable: True if labels stabilize
    - stable_label: the dominant label in the stable region
    - first_stable_idx: index where stability starts
    """
    if len(labels) <= min_sentences:
        return False, None, None
    
    # Check stability from min_sentences onwards
    remaining_labels = labels[min_sentences:]
    if len(remaining_labels) == 0:
        return False, None, None
    
    # Count occurrences of each label
    label_counts = collections.Counter(remaining_labels)
    total_remaining = len(remaining_labels)
    
    # Check if any label appears >threshold% of the time
    for label, count in label_counts.items():
        if count / total_remaining > threshold:
            return True, label, min_sentences
    
    return False, None, None

def analyze_prompt_stability(prompt_data):
    """
    Analyze stability and alignment for a single prompt.
    Returns dict with analysis results.
    """
    # Sort sentences by sentence ID
    sorted_sentences = sorted(prompt_data.items())
    sent_ids = [sent_id for sent_id, _ in sorted_sentences]
    gt_scores = [labels['gt'] for _, labels in sorted_sentences]
    pred_scores = [labels['pred'] for _, labels in sorted_sentences]
    
    # Convert to binary labels
    gt_binary = [convert_to_binary(score) for score in gt_scores]
    pred_binary = [convert_to_binary(score) for score in pred_scores]
    
    # Check GT stability
    gt_stable, gt_stable_label, gt_stable_idx = check_stability(gt_binary, args.min_sentences, args.stability_threshold)
    
    # Check pred stability
    pred_stable, pred_stable_label, pred_stable_idx = check_stability(pred_binary, args.min_sentences, args.stability_threshold)
    
    # Calculate alignment in stable region (if GT is stable)
    alignment_in_stable = None
    misalignment_percentage = None
    if gt_stable:
        stable_gt = gt_binary[gt_stable_idx:]
        stable_pred = pred_binary[gt_stable_idx:]
        
        # Calculate misalignment percentage
        misaligned = sum(1 for gt, pred in zip(stable_gt, stable_pred) if gt != pred)
        misalignment_percentage = misaligned / len(stable_gt)
        alignment_in_stable = (len(stable_gt) - misaligned) / len(stable_gt)
    
    # Overall alignment across all sentences
    total_misaligned = sum(1 for gt, pred in zip(gt_binary, pred_binary) if gt != pred)
    overall_misalignment_percentage = total_misaligned / len(gt_binary)
    
    return {
        'prompt_id': sorted_sentences[0][0].split('_')[0] if '_' in str(sorted_sentences[0][0]) else str(sorted_sentences[0][0]),
        'total_sentences': len(gt_binary),
        'gt_stable': gt_stable,
        'gt_stable_label': gt_stable_label,
        'gt_stable_idx': gt_stable_idx,
        'pred_stable': pred_stable,
        'pred_stable_label': pred_stable_label,
        'pred_stable_idx': pred_stable_idx,
        'alignment_in_stable_region': alignment_in_stable,
        'misalignment_pct_in_stable': misalignment_percentage,
        'overall_misalignment_pct': overall_misalignment_percentage,
        'gt_labels': gt_binary,
        'pred_labels': pred_binary,
        'gt_scores': gt_scores,
        'pred_scores': pred_scores,
        'sent_ids': sent_ids
    }



def load_prompt_and_cot_texts(input_folder, prompt_ids):
    """Load the actual prompt texts and CoT texts for the given prompt IDs from the LAST sentence file."""
    prompt_texts = {}
    cot_texts = {}
    labels_folder = INPUT_FOLDER / "labels" if (INPUT_FOLDER / "labels").exists() else INPUT_FOLDER
    
    for prompt_id in tqdm(prompt_ids, desc="Loading prompt and CoT texts"):
        # Find all files for this prompt_id and get the one with the highest sentence_id
        pattern = f"{prompt_id}_*_labeled.json"
        matching_files = []
        
        for label_file in labels_folder.rglob(pattern):
            if not label_file.name.endswith("_labeled_cot.json") and not label_file.name.endswith("_labeled_cot_para.json"):
                filename = label_file.stem
                parts = filename.split('_')
                if len(parts) >= 3 and parts[-1] == "labeled":
                    try:
                        sent_id = int(parts[1])
                        matching_files.append((sent_id, label_file))
                    except ValueError:
                        continue
        
        if matching_files:
            # Sort by sentence_id and get the file with the highest (last) sentence_id
            matching_files.sort(key=lambda x: x[0])
            last_sent_id, last_file = matching_files[-1]
            
            try:
                with open(last_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "raw_prompt" in data:
                        prompt_texts[prompt_id] = data["raw_prompt"]
                    elif "prompt" in data:
                        prompt_texts[prompt_id] = data["prompt"]
                    
                    if "cot" in data:
                        cot_texts[prompt_id] = data["cot"]
            except Exception as e:
                print(f"Error reading {last_file}: {e}")
                continue
    
    return prompt_texts, cot_texts

def save_high_misalignment_prompt_ids(input_folder_path, prompt_ids, json_file_path):
    """Save/append high misalignment prompt IDs to a JSON file keyed by input folder."""
    import json
    
    # Convert input folder path to string for use as key
    folder_key = str(input_folder_path)
    
    # Load existing data if file exists, otherwise start with empty dict
    if pathlib.Path(json_file_path).exists():
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing file {json_file_path}: {e}")
            print("Starting with empty data.")
            data = {}
    else:
        data = {}
    
    # Add/update the prompt IDs for this input folder
    data[folder_key] = prompt_ids
    
    # Save the updated data
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(prompt_ids)} high misalignment prompt IDs for '{folder_key}' to {json_file_path}")
    except IOError as e:
        print(f"Error: Could not write to {json_file_path}: {e}")

def plot_prompt_labels(prompt_id, gt_labels, pred_labels, output_folder, prompt_text=None):    
    plt.figure(figsize=(12, 6))
    x = list(range(len(gt_labels)))
    plt.plot(x, gt_labels, 'b-o', label='Ground Truth', linewidth=2, markersize=4, alpha=0.8)
    plt.plot(x, pred_labels, 'r-s', label='Predicted', linewidth=2, markersize=4, alpha=0.8)
    plt.xlabel('CoT Sentence Index', fontsize=12)
    plt.ylabel('Safety Label (0=Safe, 1=Unsafe)', fontsize=12)
    plt.title(f'Prompt {prompt_id}: Ground Truth vs Predicted Labels', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Safe (0)', 'Unsafe (1)'])
    plt.xlim(-0.5, len(gt_labels) - 0.5)
    
    if prompt_text:
        truncated_prompt = prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
        plt.figtext(0.02, 0.02, f"Prompt: {truncated_prompt}", fontsize=8, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                   wrap=True)
    
    misaligned = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt != pred)
    total = len(gt_labels)
    misalignment_pct = misaligned / total
    
    plt.figtext(0.98, 0.98, f"Misalignment: {misalignment_pct:.1%} ({misaligned}/{total})", 
               fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    
    output_path = OUTPUT_FOLDER / f"prompt_{prompt_id}_labels.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_high_misalignment_plots(results_df):
    high_misalignment = results_df[
        (results_df['gt_stable'] == True) & 
        (results_df['misalignment_pct_in_stable'] > args.misalignment_threshold)
    ].copy()
    
    logger.info(f"Found {len(high_misalignment)} prompts with >{args.misalignment_threshold}% misalignment in stable region")
    
    if len(high_misalignment) == 0:
        logger.info("No prompts found with the specified misalignment threshold.")
        return
    
    prompt_ids = high_misalignment['prompt_id'].tolist()
    prompt_texts, cot_texts = load_prompt_and_cot_texts(args.input_folder, prompt_ids)

    prompt_data = []
    
    for _, row in tqdm(high_misalignment.iterrows(), total=len(high_misalignment), desc="Creating plots"):
        prompt_id = row['prompt_id']
        
        gt_labels = row['gt_labels']
        pred_labels = row['pred_labels']
        
        prompt_text = prompt_texts.get(prompt_id, "Prompt text not found")
        cot_text = cot_texts.get(prompt_id, "CoT text not found")
        
        plot_path = plot_prompt_labels(prompt_id, gt_labels, pred_labels, OUTPUT_FOLDER, prompt_text)
        
        prompt_data.append({
            'prompt_id': prompt_id,
            'prompt_text': prompt_text,
            'cot_text': cot_text,
            'total_sentences': row['total_sentences'],
            'gt_stable_idx': row['gt_stable_idx'],
            'misalignment_pct_in_stable': row['misalignment_pct_in_stable'],
            'overall_misalignment_pct': row['overall_misalignment_pct'],
        })
    
    prompts_df = pd.DataFrame(prompt_data)
    csv_path = OUTPUT_FOLDER / "high_misalignment_prompts.csv"
    prompts_df.to_csv(csv_path, index=False)
    
    print(f"Among high misalignment prompts:")
    print(f"- Average misalignment: {high_misalignment['misalignment_pct_in_stable'].mean():.1%}")
    print(f"- Range: {high_misalignment['misalignment_pct_in_stable'].min():.1%} - {high_misalignment['misalignment_pct_in_stable'].max():.1%}")
    
    return high_misalignment['prompt_id'].tolist()

def print_summary_statistics(results_df):    
    total_prompts = len(results_df)
    stable_prompts = results_df[results_df['gt_stable']]
    
    print(f"Total prompts analyzed: {total_prompts}")
    print(f"Prompts with stable ground truth labels: {len(stable_prompts)} ({len(stable_prompts)/total_prompts*100:.1f}%)")
    
    if len(stable_prompts) > 0:
        print(f"\nAmong ground truth-stable prompts:")
        print(f"  - Average misalignment: {stable_prompts['misalignment_pct_in_stable'].mean():.1%}")
        print(f"  - Median misalignment: {stable_prompts['misalignment_pct_in_stable'].median():.1%}")
        
        # Perfect alignment in stable region
        perfect_alignment = stable_prompts[stable_prompts['misalignment_pct_in_stable'] == 0]
        print(f"  - Prompts with perfect alignment: {len(perfect_alignment)} ({len(perfect_alignment)/len(stable_prompts):.1%})")
        
        # High misalignment
        high_misalignment = stable_prompts[stable_prompts['misalignment_pct_in_stable'] > 0.5]
        print(f"  - Prompts with >50% misalignment: {len(high_misalignment)} ({len(high_misalignment)/len(stable_prompts):.1%})")
    
    print(f"\nOverall:")
    print(f"  - Average overall misalignment: {results_df['overall_misalignment_pct'].mean():.1%}")
    print(f"  - Median overall misalignment: {results_df['overall_misalignment_pct'].median():.1%}")


def main():
    labels_data = load_labels()
    if not labels_data:
        logger.error("No data loaded. Check input folder and file formats.")
        return
    
    # check each prompt for stability and alignment
    results = []    
    for prompt_id, prompt_data in labels_data.items():
        result = analyze_prompt_stability(prompt_data)
        result['prompt_id'] = prompt_id
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FOLDER / 'detailed_results.csv', index=False)
    print_summary_statistics(results_df)
    
    # check for high misalignment prompts
    high_misalignment_prompt_ids = create_high_misalignment_plots(results_df)
    if args.save_prompt_ids:
        stable_prompts = results_df[results_df['gt_stable']]
        high_misalignment = stable_prompts[
            stable_prompts['misalignment_pct_in_stable'] > args.misalignment_threshold
        ]
        high_misalignment_prompt_ids = high_misalignment['prompt_id'].tolist()
        print(f"Found {len(high_misalignment_prompt_ids)} high misalignment prompts (>{args.misalignment_threshold:.1%} threshold)")
    
        save_high_misalignment_prompt_ids(INPUT_FOLDER, high_misalignment_prompt_ids, args.save_prompt_ids)

    # save some perfect alignment examples
    stable_prompts = results_df[results_df['gt_stable']]
    if len(stable_prompts) > 0:
        perfect_examples = stable_prompts[stable_prompts['misalignment_pct_in_stable'] == 0]
        if len(perfect_examples) > 0:
            perfect_prompt_ids = perfect_examples['prompt_id'].head(10).tolist()
            perfect_prompt_texts, perfect_cot_texts = load_prompt_and_cot_texts(args.input_folder, perfect_prompt_ids)
            
            perfect_with_texts = []
            for _, row in perfect_examples.head(10).iterrows():
                prompt_id = row['prompt_id']
                perfect_with_texts.append({
                    'prompt_id': prompt_id,
                    'prompt_text': perfect_prompt_texts.get(prompt_id, "Prompt text not found"),
                    'cot_text': perfect_cot_texts.get(prompt_id, "CoT text not found"),
                    'total_sentences': row['total_sentences'],
                    'gt_stable_idx': row['gt_stable_idx'],
                    'misalignment_pct_in_stable': row['misalignment_pct_in_stable'],
                    'overall_misalignment_pct': row['overall_misalignment_pct']
                })
            
            perfect_df = pd.DataFrame(perfect_with_texts)
            perfect_df.to_csv(OUTPUT_FOLDER / 'perfect_alignment_examples.csv', index=False)
    
if __name__ == "__main__":
    main()
