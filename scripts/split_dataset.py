import json
import argparse
import random
import os


def load_jsonl(path):
    """Load jsonl file into a list of objects."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Split a single JSONL file into train and test sets.")
    
    # 1. Changed --inputs to --input (single file)
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input jsonl file."
    )
    
    # 2. Changed --nums to --num (single integer for training)
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        required=True,
        help="Number of items to sample for the TRAIN set."
    )
    
    parser.add_argument(
        "--test_num",
        "-tn",
        type=int,
        required=True,
        help="Number of items to sample for the TEST set."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling."
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data from the single input file
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    
    total_needed = args.num + args.test_num
    
    # Check if the file has enough data
    if total_needed > len(data):
        raise ValueError(
            f"File has only {len(data)} items, but you requested {total_needed} "
            f"(Train: {args.num} + Test: {args.test_num})."
        )

    # Use random.sample to ensure mutual exclusivity between train and test
    sampled_all = random.sample(data, total_needed)
    
    # Split the sampled data
    train_data = sampled_all[:args.num]
    test_data = sampled_all[args.num:]

    # 4. Generate output paths automatically based on the input path
    # Example: data/my_file.jsonl -> data/my_file_train-100.jsonl
    root, ext = os.path.splitext(args.input)
    train_output = f"{root}_train-{args.num}{ext}"
    test_output = f"{root}_test-{args.test_num}{ext}"

    # Write Train Output
    print(f"Writing train dataset ({len(train_data)} items)...")
    with open(train_output, "w", encoding="utf-8") as fout:
        for item in train_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved train set to: {train_output}")

    # Write Test Output
    print(f"Writing test dataset ({len(test_data)} items)...")
    with open(test_output, "w", encoding="utf-8") as fout:
        for item in test_data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved test set to: {test_output}")

    print("Successfully split the dataset!")


if __name__ == "__main__":
    main()