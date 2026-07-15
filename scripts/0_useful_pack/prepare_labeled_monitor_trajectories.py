#!/usr/bin/env python3
"""Convert cleaned rollout JSONL files to the canonical monitor dataset schema."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from collections import Counter
from pathlib import Path

from filter_hacktype_train_traj import classify_hack_type


CANONICAL_FIELDS = (
    "global_steps",
    "prompt",
    "response",
    "prompt_length",
    "response_length",
    "reward",
    "data_source",
    "ability",
    "reward_model",
    "extra_info",
    "hacking_label",
    "hacking_type",
    "traj_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--collection", choices=("normal", "hacking"), required=True)
    parser.add_argument("--traj_source", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def canonicalize(item: dict, collection: str, traj_source: str, location: str) -> dict:
    prompt = item.get("prompt")
    response = item.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError(f"{location}: prompt and response must be strings")

    try:
        reward = float(item.get("reward"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: reward must be numeric") from exc

    classified = classify_hack_type(response)
    if collection == "normal":
        if reward > 0.0:
            raise ValueError(f"{location}: normal row has positive reward {reward}")
        if classified != "neither":
            raise ValueError(f"{location}: normal row contains {classified} code")
        hacking_label = 0
        hacking_type = "normal"
    else:
        if reward <= 0.0:
            raise ValueError(f"{location}: hacking row has non-positive reward {reward}")
        if classified == "neither":
            raise ValueError(f"{location}: hacking row has no recognized hack type")
        hacking_label = 1
        hacking_type = classified.replace("_only", "")

    values = {
        **item,
        "prompt_length": len(prompt),
        "response_length": len(response),
        "reward": reward,
        "hacking_label": hacking_label,
        "hacking_type": hacking_type,
        "traj_source": traj_source,
    }
    missing = [field for field in CANONICAL_FIELDS if field not in values]
    if missing:
        raise ValueError(f"{location}: missing canonical fields: {missing}")
    return {field: values[field] for field in CANONICAL_FIELDS}


def load_and_convert(path: Path, collection: str, traj_source: str) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    counts: Counter = Counter()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            row = canonicalize(item, collection, traj_source, f"{path}:{line_number}")
            rows.append(row)
            source = "leetcode" if "leetcode" in str(row["data_source"]).lower() else "taco"
            counts[(source, row["hacking_type"])] += 1
    return rows, counts


def write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o664
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output_file:
        temp_path = Path(output_file.name)
        try:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    os.chmod(temp_path, mode)
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    rows, counts = load_and_convert(args.input_file, args.collection, args.traj_source)
    print(f"Validated {len(rows)} rows: {dict(sorted(counts.items()))}")
    if args.dry_run:
        print("Dry run: no output written.")
        return
    write_atomic(args.output_file, rows)
    print(f"Wrote {len(rows)} rows to {args.output_file}")


if __name__ == "__main__":
    main()
