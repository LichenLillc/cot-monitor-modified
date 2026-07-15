import json
import argparse
from pathlib import Path

def process_files(input_files, dry_run):
    global_total = 0
    global_kept = 0
    global_removed = 0
    valid_files = 0

    for file_path_str in input_files:
        file_path = Path(file_path_str)

        if not file_path.is_file():
            print(f"⚠️ Skipping '{file_path_str}': Not a valid file.")
            continue

        valid_files += 1
        # Create output path: same directory, append _filteredU before the extension
        output_path = file_path.with_name(f"{file_path.stem}_filteredU{file_path.suffix}")

        total = 0
        kept = 0
        removed = 0

        print(f"\n📄 Processing: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                # Open the output file only if we are not in dry_run mode
                outfile = open(output_path, 'w', encoding='utf-8') if not dry_run else None

                try:
                    for line in infile:
                        if not line.strip():
                            continue

                        total += 1
                        try:
                            data = json.loads(line)
                            resp_text = data.get("response", "")

                            # Check if "response" is a string and contains the target text
                            if isinstance(resp_text, str) and "unittest.main()" in resp_text:
                                removed += 1
                            else:
                                kept += 1
                                if outfile:
                                    outfile.write(line)

                        except json.JSONDecodeError:
                            print(f"  [Warning] Invalid JSON on line {total}. Skipping.")
                            removed += 1

                finally:
                    if outfile:
                        outfile.close()

        except Exception as e:
            print(f"  ❌ Error reading {file_path.name}: {e}")
            continue

        # Accumulate global stats
        global_total += total
        global_kept += kept
        global_removed += removed

        # Print per-file statistics
        print(f"  Total lines:   {total}")
        print(f"  Kept lines:    {kept}")
        print(f"  Removed lines: {removed}")
        if not dry_run:
            print(f"  ✅ Saved to:    {output_path.name}")
        else:
            print("  🛑 [DRY RUN]    No file was saved.")

    # Print final summary statistics
    print("\n" + "="*40)
    print("📊 FINAL STATISTICS SUMMARY")
    print("="*40)
    print(f"Files processed:       {valid_files}")
    print(f"Global total lines:    {global_total}")
    print(f"Global kept lines:     {global_kept}")
    print(f"Global removed lines:  {global_removed}")
    print("="*40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter out JSONL lines where the 'prompt' contains 'unittest.main()'.")
    parser.add_argument("files", nargs="+", help="Path to one or more .jsonl files to process.")
    parser.add_argument("--dry-run", action="store_true", help="Calculate and print statistics without writing output files.")

    args = parser.parse_args()
    process_files(args.files, args.dry_run)

if __name__ == "__main__":
    main()