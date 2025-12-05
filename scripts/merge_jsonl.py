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
    parser = argparse.ArgumentParser(description="Merge JSONL files with random sampling.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of input jsonl files."
    )
    parser.add_argument(
        "--nums",
        nargs="+",
        type=int,
        required=True,
        help="How many items to sample from each input file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling from each file."
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If set, shuffle the merged list (random seed fixed to 0)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output jsonl file path (directory + filename)."
    )

    args = parser.parse_args()

    # Sanity check
    if len(args.inputs) != len(args.nums):
        raise ValueError("Length of --inputs and --nums must be the same.")

    random.seed(args.seed)

    merged = []

    # Sample from each file
    for path, num in zip(args.inputs, args.nums):
        data = load_jsonl(path)
        if num > len(data):
            raise ValueError(
                f"File {path} has only {len(data)} items, but requested {num}."
            )

        sampled = random.sample(data, num)
        merged.extend(sampled)

    # Optional shuffle after merging
    if args.shuffle:
        random.seed(0)
        random.shuffle(merged)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Write output jsonl
    with open(args.output, "w", encoding="utf-8") as fout:
        for item in merged:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done! Merged dataset saved to: {args.output}")


if __name__ == "__main__":
    main()