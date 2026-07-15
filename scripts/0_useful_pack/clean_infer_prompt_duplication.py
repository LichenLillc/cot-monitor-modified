#!/usr/bin/env python3
"""Remove a duplicated assistant response from decoded inference prompts."""

from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path


DEFAULT_MARKER = "Let's think step by step:\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean prompts shaped as '<prompt marker>\\nassistant\\n<response>'. "
            "The command is a dry run unless --in_place is supplied."
        )
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--marker", default=DEFAULT_MARKER)
    parser.add_argument("--in_place", action="store_true")
    return parser.parse_args()


def clean_item(item: dict, marker: str, path: Path, line_number: int) -> bool:
    prompt = item.get("prompt")
    response = item.get("response")
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError(f"{path}:{line_number}: prompt and response must both be strings")

    marker_end = prompt.rfind(marker)
    if marker_end < 0:
        raise ValueError(f"{path}:{line_number}: marker not found")
    marker_end += len(marker)

    expected_suffix = "\nassistant\n" + response
    actual_suffix = prompt[marker_end:]
    if actual_suffix != expected_suffix:
        raise ValueError(
            f"{path}:{line_number}: prompt suffix does not exactly match "
            "'\\nassistant\\n' plus response"
        )

    item["prompt"] = prompt[:marker_end]
    if "prompt_length" in item:
        item["prompt_length"] = len(item["prompt"])
    return True


def process_file(path: Path, marker: str, in_place: bool) -> tuple[int, int]:
    rows: list[dict] = []
    rows_seen = 0
    rows_cleanable = 0

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            rows_seen += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows_cleanable += clean_item(item, marker, path, line_number)
            rows.append(item)

    if in_place:
        original_mode = stat.S_IMODE(path.stat().st_mode)
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
                for item in rows:
                    output_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                output_file.flush()
                os.fsync(output_file.fileno())
            except BaseException:
                temp_path.unlink(missing_ok=True)
                raise

        os.chmod(temp_path, original_mode)
        os.replace(temp_path, path)

    return rows_seen, rows_cleanable


def main() -> None:
    args = parse_args()
    mode = "IN PLACE" if args.in_place else "DRY RUN"
    print(f"Mode: {mode}")
    for path in args.files:
        rows_seen, rows_cleanable = process_file(path, args.marker, args.in_place)
        print(f"{path}: {rows_cleanable}/{rows_seen} prompts validated and cleaned")


if __name__ == "__main__":
    main()
