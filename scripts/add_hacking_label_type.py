import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Add hacking labels to JSONL file.")
    parser.add_argument("input_file", type=str, help="Path to input .jsonl file")
    parser.add_argument("--label", type=int, required=True, choices=[0, 1],
                        help="Value for hacking_label field (0 or 1)")
    parser.add_argument("--type", type=str, default="unknown",
                        help="Value for hacking_type field (string)")
    parser.add_argument("--source", type=str, required=True,
                        help="Value for traj_source field (string)")

    args = parser.parse_args()
    input_path = args.input_file

    # Generate output path
    base, ext = os.path.splitext(input_path)
    output_path = base + "_labeled" + ext

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            obj["hacking_label"] = args.label
            obj["hacking_type"] = args.type
            obj["traj_source"] = args.source

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Finished! Output written to: {output_path}")


if __name__ == "__main__":
    main()