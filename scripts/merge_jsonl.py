import json
import argparse
import random
import os


def load_jsonl(path):
    """Load jsonl file into a list of objects"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Merge JSONL files with random sampling (optional test set split).")
    parser.add_argument(
        "--inputs",
        '-i',
        nargs="+",
        required=True,
        help="List of input jsonl files."
    )
    parser.add_argument(
        "--nums",
        '-n',
        nargs="+",
        type=int,
        required=True,
        help="How many items to sample for the MAIN dataset from each input file."
    )
    parser.add_argument(
        "--test_num",
        '-tn',
        nargs="+",
        type=int,
        default=None,
        help="How many items to sample for the TEST dataset from each input file. (Must match length of inputs if provided)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling from each file."
    )
    parser.add_argument(
        "--shuffle",
        '-s',
        action="store_true",
        help="If set, shuffle the merged MAIN list (random seed fixed to 0)."
    )
    parser.add_argument(
        "--output",
        '-o',
        type=str,
        required=True,
        help="Output jsonl file path (directory + filename)."
    )

    args = parser.parse_args()

    # Sanity check: input lengths
    if len(args.inputs) != len(args.nums):
        raise ValueError("Length of --inputs and --nums must be the same.")

    # Handle optional test_num argument
    has_test_set = args.test_num is not None
    if has_test_set:
        if len(args.inputs) != len(args.test_num):
            raise ValueError("Length of --inputs and --test_num must be the same.")
        test_nums_list = args.test_num
    else:
        # If no test set requested, fill with zeros to simplify the loop
        test_nums_list = [0] * len(args.inputs)

    random.seed(args.seed)

    merged_main = []
    merged_test = []

    # Sample from each file
    for path, num_main, num_test in zip(args.inputs, args.nums, test_nums_list):
        data = load_jsonl(path)
        
        total_needed = num_main + num_test
        
        if total_needed > len(data):
            raise ValueError(
                f"File {path} has only {len(data)} items, but requested total {total_needed} "
                f"(Main: {num_main} + Test: {num_test})."
            )

        # Sample distinct items for both sets at once to ensure mutual exclusion
        sampled_all = random.sample(data, total_needed)
        
        # Split: first 'num_main' go to training/main, rest go to test
        sampled_main = sampled_all[:num_main]
        sampled_test = sampled_all[num_main:]
        
        merged_main.extend(sampled_main)
        merged_test.extend(sampled_test)

    # Optional shuffle for MAIN dataset
    if args.shuffle:
        random.seed(0)
        random.shuffle(merged_main)
        # Note: We usually do not shuffle the test set by default to keep check-ability, 
        # but if needed, similar logic applies. Here we leave test set in extraction order.

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 1. Write Main Output
    print(f"Writing main dataset ({len(merged_main)} items)...")
    with open(args.output, "w", encoding="utf-8") as fout:
        for item in merged_main:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved main dataset to: {args.output}")

    # 2. Write Test Output (if requested)
    if has_test_set:
        # Generate filename: e.g., "data/merged.jsonl" -> "data/merged_test.jsonl"
        root, ext = os.path.splitext(args.output)
        test_output_path = f"{root}_test{ext}"
        
        print(f"Writing test dataset ({len(merged_test)} items)...")
        with open(test_output_path, "w", encoding="utf-8") as fout:
            for item in merged_test:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved test dataset to: {test_output_path}")

    print("All done!")


if __name__ == "__main__":
    main()