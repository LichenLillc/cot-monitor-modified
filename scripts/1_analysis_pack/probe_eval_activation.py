#!/usr/bin/env python3
"""Evaluate saved activation tensors with a probe monitor and write evaled labels.

For each activation file:
  dataset/activations/{pair_id}_{cot_idx}.pt
  dataset/labels/{pair_id}/{pair_id}_{cot_idx}_labeled.json

Writes:
  dataset/labels/{pair_id}/{pair_id}_{cot_idx}_labeled_evaled.json

The original labeled JSON is not modified.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger

DEFAULT_BUNDLE = (
    "/nfs/data/lichenli/cot-monitor-modified/main_table3_paired/exp_0328/probe_outputs_text/ckpt61/"
    "qwen_scratch-ckpt61_paired-tn255-tuh255-teh255_ckpt61/saved_models/mlp_v1_pca_seed8_torch_bundle.pt"
)


class TorchMLPV1PCAMonitor(torch.nn.Module):
    REQUIRED_KEYS = (
        "global_scaler_mean",
        "global_scaler_scale",
        "pca_mean",
        "pca_components",
        "mlp_scaler_mean",
        "mlp_scaler_scale",
        "mlp_state_dict",
    )

    def __init__(self, bundle: dict[str, Any]):
        super().__init__()
        missing = [key for key in self.REQUIRED_KEYS if key not in bundle]
        if missing:
            raise ValueError(f"Probe monitor bundle missing keys: {missing}")

        self.metadata = {
            "model_name": bundle.get("model_name", "mlp_v1_pca"),
            "seed": bundle.get("seed"),
            "feature": bundle.get("feature", "text"),
        }
        self.register_buffer("global_scaler_mean", bundle["global_scaler_mean"].float())
        self.register_buffer("global_scaler_scale", bundle["global_scaler_scale"].float())
        self.register_buffer("pca_mean", bundle["pca_mean"].float())
        self.register_buffer("pca_components", bundle["pca_components"].float())
        self.register_buffer("mlp_scaler_mean", bundle["mlp_scaler_mean"].float())
        self.register_buffer("mlp_scaler_scale", bundle["mlp_scaler_scale"].float())

        state_dict = bundle["mlp_state_dict"]
        self.register_buffer("fc1_weight", state_dict["fc1.weight"].float())
        self.register_buffer("fc1_bias", state_dict["fc1.bias"].float())
        self.register_buffer("head_weight", state_dict["head.weight"].float())
        self.register_buffer("head_bias", state_dict["head.bias"].float())

    @classmethod
    def from_bundle_path(cls, path: Path, map_location: str = "cpu"):
        bundle = torch.load(path, map_location=map_location, weights_only=False)
        return cls(bundle)

    @staticmethod
    def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
        return torch.clamp(scale, min=1e-12)

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        x = activations.float()
        x = (x - self.global_scaler_mean) / self._safe_scale(self.global_scaler_scale)
        x = (x - self.pca_mean) @ self.pca_components.t()
        x = (x - self.mlp_scaler_mean) / self._safe_scale(self.mlp_scaler_scale)
        x = F.relu(F.linear(x, self.fc1_weight, self.fc1_bias))
        logits = F.linear(x, self.head_weight, self.head_bias)
        return torch.sigmoid(logits).squeeze(-1)


def parse_args() -> argparse.Namespace:
    source = argparse.ArgumentParser(description="Evaluate activation .pt files with a saved probe monitor.")
    group = source.add_mutually_exclusive_group(required=True)
    group.add_argument("--processed_root", type=Path, help="Root containing dataset folders recursively, e.g. processed_text")
    group.add_argument("--input_folder", type=Path, help="Single dataset folder containing activations/ and labels/")
    source.add_argument("--probe_bundle", type=Path, default=Path(DEFAULT_BUNDLE))
    source.add_argument("--batch_size", type=int, default=256)
    source.add_argument("--device", default="cpu", help="cpu or cuda")
    source.add_argument("--overwrite", action="store_true", help="Regenerate existing *_labeled_evaled.json files")
    source.add_argument("--dry_run", action="store_true", help="Compute counts only; do not write files")
    return source.parse_args()


def find_dataset_folders(args: argparse.Namespace) -> list[Path]:
    if args.input_folder:
        folders = [args.input_folder]
    else:
        folders = [p.parent for p in args.processed_root.rglob("activations") if (p.parent / "labels").is_dir()]
    return sorted(set(folders))


def label_path_for_activation(dataset: Path, activation_path: Path) -> Path | None:
    stem = activation_path.stem
    try:
        pair_id, cot_idx = stem.rsplit("_", 1)
    except ValueError:
        return None
    return dataset / "labels" / pair_id / f"{pair_id}_{cot_idx}_labeled.json"


def evaled_path_for_label(label_path: Path) -> Path:
    return label_path.with_name(label_path.name.replace("_labeled.json", "_labeled_evaled.json"))


def load_activation(path: Path) -> torch.Tensor:
    act = torch.load(path, map_location="cpu")
    if act.dim() > 1:
        act = act.reshape(-1)
    return act.to(torch.float32)


def write_evaled_label(label_path: Path, out_path: Path, eval_payload: dict[str, Any]) -> None:
    with label_path.open("r", encoding="utf-8") as f:
        item = json.load(f)
    item["probe_monitor_eval"] = eval_payload
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)


def process_dataset(dataset: Path, monitor: TorchMLPV1PCAMonitor, args: argparse.Namespace) -> dict[str, int]:
    act_dir = dataset / "activations"
    activation_files = sorted(act_dir.glob("*.pt"))
    counts = {"activations": len(activation_files), "written": 0, "skipped_existing": 0, "missing_label": 0, "errors": 0}
    pending: list[tuple[Path, Path, Path]] = []

    for act_path in activation_files:
        label_path = label_path_for_activation(dataset, act_path)
        if label_path is None or not label_path.exists():
            counts["missing_label"] += 1
            continue
        out_path = evaled_path_for_label(label_path)
        if out_path.exists() and not args.overwrite:
            counts["skipped_existing"] += 1
            continue
        pending.append((act_path, label_path, out_path))

    if args.dry_run:
        return counts

    logger.info(f"{dataset}: scoring {len(pending)} / {len(activation_files)} activations")
    for start in range(0, len(pending), args.batch_size):
        batch_items = pending[start:start + args.batch_size]
        try:
            acts = torch.stack([load_activation(act_path) for act_path, _, _ in batch_items]).to(args.device)
            with torch.no_grad():
                scores = monitor(acts).detach().cpu().tolist()
        except Exception as exc:
            logger.warning(f"Batch failed in {dataset} at offset {start}: {exc}")
            counts["errors"] += len(batch_items)
            continue

        for (act_path, label_path, out_path), p_hack in zip(batch_items, scores):
            try:
                payload = {
                    "monitor_name": monitor.metadata.get("model_name", "mlp_v1_pca"),
                    "seed": monitor.metadata.get("seed"),
                    "feature": monitor.metadata.get("feature", "text"),
                    "p_hack": float(p_hack),
                    "probe_bundle": str(args.probe_bundle),
                    "activation_file": str(act_path),
                    "eval_time_unix": time.time(),
                }
                write_evaled_label(label_path, out_path, payload)
                counts["written"] += 1
            except Exception as exc:
                logger.warning(f"Failed writing evaled label for {label_path}: {exc}")
                counts["errors"] += 1
    return counts


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    logger.info(f"Loading probe monitor bundle: {args.probe_bundle}")
    monitor = TorchMLPV1PCAMonitor.from_bundle_path(args.probe_bundle, map_location=args.device)
    monitor.to(args.device)
    monitor.eval()
    logger.info(f"Probe metadata: {monitor.metadata}")

    datasets = find_dataset_folders(args)
    if not datasets:
        raise SystemExit("No dataset folders with activations/ and labels/ found.")
    logger.info(f"Found {len(datasets)} dataset folders")

    total = {"activations": 0, "written": 0, "skipped_existing": 0, "missing_label": 0, "errors": 0}
    for dataset in datasets:
        counts = process_dataset(dataset, monitor, args)
        for key in total:
            total[key] += counts[key]
        logger.info(f"Done {dataset.name}: {counts}")

    logger.success(f"All done: {total}")


if __name__ == "__main__":
    main()
