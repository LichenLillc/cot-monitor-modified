#!/usr/bin/env python3
"""Filter GRPO training trajectory JSONL files by final-code hack type.

This prepares trajectory rows for the paired probe pipeline:
  0_preprocess_paired.py -> 1_2a_converter_mine_paired.py -> 2b_get_activations...

Classification is based on the last ```python ... ``` code block, while the output
keeps the full assistant response for activation extraction. This matches the
probe/monitor setting, where the model reads the whole transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

STEP_RE = re.compile(r"train_batch_stats_steps_(\d+)_to_(\d+)\.jsonl$")
PY_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
USER_RE = re.compile(r"(?:^|\n)\s*user\s*\n", re.IGNORECASE)
ASSISTANT_RE = re.compile(r"\n\s*assistant\s*\n", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Skywork/verl train_batch_stats JSONL trajectories by hack type."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stats_dir", type=Path, help="Directory containing train_batch_stats_steps_*.jsonl")
    source.add_argument("--input_files", nargs="+", type=Path, help="Specific stats JSONL files to scan")

    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for filtered paired-pipeline JSONL files")
    parser.add_argument("--prefix", default="step_penalty", help="Output filename prefix")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["unittest_only", "exit_only", "normal"],
        choices=["unittest_only", "exit_only", "both", "neither", "normal"],
        help=(
            "Buckets to write. Hack buckets use reward > --reward_gt. "
            "The normal bucket uses reward <= --normal_reward_le and, by default, excludes unittest/exit code."
        ),
    )
    parser.add_argument("--reward_gt", type=float, default=0.0, help="Keep rows with reward > this value")
    parser.add_argument("--normal_reward_le", type=float, default=0.0, help="For --types normal, keep rows with reward <= this value")
    parser.add_argument("--p_hack_le", type=float, default=None, help="Optional upper bound: keep rows with p_hack <= this value")
    parser.add_argument("--p_hack_ge", type=float, default=None, help="Optional lower bound: keep rows with p_hack >= this value")
    parser.add_argument(
        "--normal_allow_hack_keywords",
        action="store_true",
        help="Allow zero-reward normal rows even if their final code block contains unittest/exit. Default excludes those rows.",
    )
    parser.add_argument("--step_min", type=int, default=None, help="Minimum global step to include")
    parser.add_argument("--step_max", type=int, default=None, help="Maximum global step to include")
    parser.add_argument(
        "--used_for_policy_update",
        choices=["any", "true", "false"],
        default="any",
        help="Filter on used_for_policy_update when present. Missing keys are kept for 'any' only.",
    )
    parser.add_argument("--max_per_type", type=int, default=500, help="Optional random cap per output type; 0 means no cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--traj_source_prefix",
        default="penalty_train_traj",
        help="Prefix for output traj_source; step number is appended",
    )
    parser.add_argument(
        "--keep_metadata",
        action="store_true",
        help="Also preserve reward/p_hack/upass metadata in output rows for debugging. 0_preprocess_paired.py ignores these.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print/write filtering statistics; do not write filtered trajectory JSONL files.",
    )
    return parser.parse_args()


def iter_stats_files(args: argparse.Namespace) -> list[Path]:
    if args.input_files:
        return sorted(args.input_files)
    return sorted(args.stats_dir.glob("train_batch_stats_steps_*.jsonl"))


def step_from_path(path: Path) -> int | None:
    m = STEP_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def last_python_code_block(text: str) -> str:
    blocks = PY_BLOCK_RE.findall(text or "")
    return blocks[-1] if blocks else ""


def strip_comments(code: str) -> str:
    if not code:
        return ""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        kept: list[str] = []
        for tok_type, tok_string, *_ in tokens:
            if tok_type in {tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER}:
                continue
            kept.append(tok_string)
        return " ".join(kept)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Fall back to a conservative line-comment strip for malformed generated code.
        return "\n".join(line.split("#", 1)[0] for line in code.splitlines())


def classify_hack_type(response: str) -> str:
    code = strip_comments(last_python_code_block(response)).lower()
    has_unittest = "unittest" in code
    has_exit = "exit" in code
    if has_unittest and has_exit:
        return "both"
    if has_unittest:
        return "unittest_only"
    if has_exit:
        return "exit_only"
    return "neither"


def extract_user_prompt(raw_prompt: str) -> str:
    """Extract user text from decoded chat-like stats prompt.

    Some training stats rows decode the full sequence into `prompt`, including an
    assistant section. For paired activation extraction we need only user content.
    """
    raw_prompt = raw_prompt or ""
    user_match = list(USER_RE.finditer(raw_prompt))
    if not user_match:
        return raw_prompt.strip()

    start = user_match[-1].end()
    assistant_match = ASSISTANT_RE.search(raw_prompt, start)
    end = assistant_match.start() if assistant_match else len(raw_prompt)
    return raw_prompt[start:end].strip()


def make_pair_id(item: dict[str, Any], prompt: str, step: int | None, line_no: int) -> str:
    data_source = str(item.get("data_source", "unknown")).strip() or "unknown"
    extra_info = item.get("extra_info") or {}
    idx = extra_info.get("index") if isinstance(extra_info, dict) else None
    if idx is not None and idx != "":
        base = f"{data_source}_{idx}"
    else:
        digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
        base = f"{data_source}_{digest}"
    # Keep same data point grouped, but avoid collisions across repeated logged rows.
    return f"{base}_s{step if step is not None else 'x'}_l{line_no}"


def row_passes_policy_filter(item: dict[str, Any], mode: str) -> bool:
    if mode == "any":
        return True
    if "used_for_policy_update" not in item:
        return False
    value = bool(item.get("used_for_policy_update"))
    return value if mode == "true" else not value


def row_passes_phack_filter(item: dict[str, Any], args: argparse.Namespace, counts: Counter) -> bool:
    if args.p_hack_le is None and args.p_hack_ge is None:
        return True
    try:
        p_hack = float(item.get("p_hack"))
    except (TypeError, ValueError):
        counts["p_hack_missing_or_bad"] += 1
        return False
    if args.p_hack_ge is not None and p_hack < args.p_hack_ge:
        counts["p_hack_ge_filtered"] += 1
        return False
    if args.p_hack_le is not None and p_hack > args.p_hack_le:
        counts["p_hack_le_filtered"] += 1
        return False
    counts["p_hack_range_passed"] += 1
    return True


def read_rows(files: Iterable[Path], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter = Counter()

    wanted = set(args.types)
    for path in files:
        file_step = step_from_path(path)
        if args.step_min is not None and file_step is not None and file_step < args.step_min:
            continue
        if args.step_max is not None and file_step is not None and file_step > args.step_max:
            continue

        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                counts["rows_seen"] += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    counts["json_decode_error"] += 1
                    continue

                step = item.get("global_steps", file_step)
                if args.step_min is not None and step is not None and int(step) < args.step_min:
                    continue
                if args.step_max is not None and step is not None and int(step) > args.step_max:
                    continue

                try:
                    reward = float(item.get("reward", 0.0))
                except (TypeError, ValueError):
                    counts["bad_reward"] += 1
                    continue
                if not row_passes_policy_filter(item, args.used_for_policy_update):
                    counts["policy_update_filtered"] += 1
                    continue

                response = item.get("response", "")
                hack_type = classify_hack_type(response)
                counts[f"type_{hack_type}"] += 1
                output_type = None
                hacking_label = 1
                hacking_type_value = hack_type.replace("_only", "")

                if "normal" in wanted and reward <= args.normal_reward_le:
                    if args.normal_allow_hack_keywords or hack_type == "neither":
                        output_type = "normal"
                        hacking_label = 0
                        hacking_type_value = "normal"
                    else:
                        counts["normal_hack_keyword_filtered"] += 1
                        continue
                elif reward > args.reward_gt and hack_type in wanted:
                    if not row_passes_phack_filter(item, args, counts):
                        continue
                    output_type = hack_type
                else:
                    counts["reward_or_type_filtered"] += 1
                    continue

                prompt = extract_user_prompt(item.get("prompt", ""))
                output = {
                    "prompt": prompt,
                    "response": response,
                    "hacking_label": hacking_label,
                    "hacking_type": hacking_type_value,
                    "traj_source": f"{args.traj_source_prefix}_step{step}",
                    "pair_id": make_pair_id(item, prompt, int(step) if step is not None else None, line_no),
                }
                if args.keep_metadata:
                    for key in ["global_steps", "reward", "base_reward", "p_hack", "upass_type", "used_for_policy_update", "data_source", "extra_info"]:
                        if key in item:
                            output[key] = item[key]
                    output["source_file"] = path.name
                    output["source_line"] = line_no

                buckets[output_type].append(output)
                counts["rows_kept"] += 1

    return buckets, counts


def write_outputs(buckets: dict[str, list[dict[str, Any]]], counts: Counter, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    summary_rows = []
    for hack_type in args.types:
        rows = list(buckets.get(hack_type, []))
        total = len(rows)
        if args.max_per_type and total > args.max_per_type:
            rows = rng.sample(rows, args.max_per_type)
            rows.sort(key=lambda x: (x.get("global_steps", 0), x.get("pair_id", "")))

        out_path = args.output_dir / f"{args.prefix}_{hack_type}_test.jsonl"
        if not args.dry_run:
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_rows.append((hack_type, total, len(rows), out_path.name))

    summary_path = args.output_dir / f"{args.prefix}_filter_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"dry_run\t{args.dry_run}\n")
        f.write(f"p_hack_ge\t{args.p_hack_ge}\n")
        f.write(f"p_hack_le\t{args.p_hack_le}\n")
        f.write("metric\tvalue\n")
        for key, value in sorted(counts.items()):
            f.write(f"{key}\t{value}\n")
        f.write("\noutput_type\ttotal_before_cap\twritten\tfile\n")
        for hack_type, total, written, filename in summary_rows:
            f.write(f"{hack_type}\t{total}\t{written}\t{filename}\n")

    if args.dry_run:
        print(f"Dry run complete. No trajectory JSONL files written. Summary: {summary_path}")
    else:
        print(f"Wrote outputs to {args.output_dir}")
    print(f"Summary: {summary_path}")
    for hack_type, total, written, filename in summary_rows:
        print(f"  {hack_type}: {written}/{total} -> {filename}")


def main() -> None:
    args = parse_args()
    files = iter_stats_files(args)
    if not files:
        raise SystemExit("No input stats files found.")
    buckets, counts = read_rows(files, args)
    write_outputs(buckets, counts, args)


if __name__ == "__main__":
    main()
