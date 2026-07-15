#!/usr/bin/env python3
"""Visualize and quantify activation shift for fixed trajectories across checkpoints.

Expected use case:
  same prompt/response trajectories are run through pre/post checkpoints, with
  activations saved by 2b_get_activations_mine_paired_fix-chat-template.py.

The script loads paired activation IDs across checkpoints, reports high-dimensional
shift statistics, and writes PCA/UMAP/t-SNE scatter plots plus paired shift plots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover - optional dependency check at runtime
    umap = None


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    checkpoint: str
    hack_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation-shift visualization for penalty monitor experiments.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--processed_root",
        type=Path,
        help="Root like .../penalty_act_shift/processed_text. Expects checkpoint/dataset folders below it.",
    )
    source.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("PATH", "CHECKPOINT", "HACK_TYPE"),
        help="Explicit dataset triple. Can be repeated.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pre_checkpoint", default="pre_ckpt61")
    parser.add_argument("--post_checkpoint", default="post_step_penalty_gs360")
    parser.add_argument("--hack_types", nargs="+", default=["unittest", "exit", "normal"])
    parser.add_argument("--method", choices=["pca", "umap", "tsne", "all"], default="umap")
    parser.add_argument("--pca_dims", type=int, default=50, help="PCA dimensions before UMAP/t-SNE and distance reporting")
    parser.add_argument("--max_pairs_per_type", type=int, default=130, help="Optional deterministic cap per hack type; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--html", action="store_true", help="Also write interactive HTML scatter if plotly is available")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def infer_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.dataset:
        return [DatasetSpec(Path(path), checkpoint, hack_type) for path, checkpoint, hack_type in args.dataset]

    specs: list[DatasetSpec] = []
    for checkpoint in [args.pre_checkpoint, args.post_checkpoint]:
        ckpt_root = args.processed_root / checkpoint
        for hack_type in args.hack_types:
            candidates = sorted(ckpt_root.glob(f"*{hack_type}*"))
            candidates = [p for p in candidates if (p / "activations").is_dir() and (p / "labels").is_dir()]
            if not candidates:
                logger.warning(f"No dataset found for checkpoint={checkpoint}, hack_type={hack_type} under {ckpt_root}")
                continue
            if len(candidates) > 1:
                logger.warning(f"Multiple candidates for {checkpoint}/{hack_type}; using {candidates[0]}")
            specs.append(DatasetSpec(candidates[0], checkpoint, hack_type))
    return specs


def build_label_lookup(labels_dir: Path) -> dict[str, Path]:
    lookup = {p.name.replace("_labeled.json", ""): p for p in labels_dir.rglob("*_labeled.json")}
    for p in labels_dir.rglob("*_labeled_evaled.json"):
        lookup[p.name.replace("_labeled_evaled.json", "")] = p
    return lookup


def load_label(label_path: Path | None) -> dict[str, Any]:
    if label_path is None:
        return {}
    try:
        with label_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load label {label_path}: {exc}")
        return {}


def extract_p_hack(label: dict[str, Any]) -> float:
    probe_eval = label.get("probe_monitor_eval") or {}
    try:
        return float(probe_eval["p_hack"])
    except (KeyError, TypeError, ValueError):
        return math.nan


def extract_train_step(traj_source: Any) -> float:
    if traj_source is None:
        return math.nan
    match = re.search(r"(?:^|[_-])step(?:s)?[_-]?(\d+)", str(traj_source))
    if not match:
        return math.nan
    return float(match.group(1))


def load_dataset(spec: DatasetSpec) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    act_dir = spec.path / "activations"
    label_lookup = build_label_lookup(spec.path / "labels")
    xs: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    pt_files = sorted(act_dir.glob("*.pt"))
    logger.info(f"Loading {len(pt_files)} activations from {spec.path}")
    for pt_path in pt_files:
        stem = pt_path.stem
        try:
            act = torch.load(pt_path, map_location="cpu")
            if act.dim() > 1:
                act = act.reshape(-1)
            arr = act.to(torch.float32).numpy()
        except Exception as exc:
            logger.warning(f"Skipping {pt_path}: {exc}")
            continue

        label = load_label(label_lookup.get(stem))
        xs.append(arr)
        traj_source = label.get("traj_source", "unknown")
        rows.append(
            {
                "activation_id": stem,
                "checkpoint": spec.checkpoint,
                "hack_type": label.get("hacking_type", spec.hack_type),
                "dataset_hack_type": spec.hack_type,
                "traj_source": traj_source,
                "train_step": extract_train_step(traj_source),
                "safety_score": (label.get("safety_label") or {}).get("score", None),
                "p_hack": extract_p_hack(label),
                "dataset_path": str(spec.path),
            }
        )
    return xs, rows


def normalize_pair_id(activation_id: str, checkpoints: list[str]) -> str:
    # The converter output includes model alias in dataset folder, but activation stems
    # come from pair_id/cot_idx and should usually already match across checkpoints.
    # This fallback removes an accidental checkpoint suffix if present.
    out = activation_id
    for ckpt in sorted(checkpoints, key=len, reverse=True):
        out = re.sub(rf"_{re.escape(ckpt)}$", "", out)
    return out


def cap_pairs(df: pd.DataFrame, X: np.ndarray, max_pairs_per_type: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if max_pairs_per_type <= 0:
        return df, X
    rng = np.random.default_rng(seed)
    keep_indices: list[int] = []
    for hack_type, sub in df.groupby("hack_type"):
        pair_ids = sorted(set(sub["pair_id"]))
        if len(pair_ids) > max_pairs_per_type:
            pair_ids = sorted(rng.choice(pair_ids, size=max_pairs_per_type, replace=False).tolist())
        keep_indices.extend(sub[sub["pair_id"].isin(pair_ids)].index.tolist())
    keep_indices = sorted(keep_indices)
    return df.loc[keep_indices].reset_index(drop=True), X[keep_indices]


def pair_distances(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(A) == 0 or len(B) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    diffs = A[:, None, :] - B[None, :, :]
    l2 = np.linalg.norm(diffs, axis=2).reshape(-1)
    denom = np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :]
    dot = A @ B.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = 1.0 - dot / denom
    return l2.astype(np.float64), cos.reshape(-1).astype(np.float64)


def within_distances(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(A) < 2:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    i, j = np.triu_indices(len(A), k=1)
    diffs = A[i] - A[j]
    l2 = np.linalg.norm(diffs, axis=1)
    denom = np.linalg.norm(A[i], axis=1) * np.linalg.norm(A[j], axis=1)
    dot = np.sum(A[i] * A[j], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = 1.0 - dot / denom
    return l2.astype(np.float64), cos.astype(np.float64)


def finite_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.nanmean(values))


def finite_median(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.nanmedian(values))


def safe_ratio(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or den == 0:
        return math.nan
    return float(num / den)


def finite_values(values: list[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def pairwise_shift_stats(df: pd.DataFrame, X: np.ndarray, pre: str, post: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    for hack_type, sub in df.groupby("hack_type"):
        pre_map = {r.pair_id: i for i, r in sub[sub["checkpoint"] == pre].iterrows()}
        post_map = {r.pair_id: i for i, r in sub[sub["checkpoint"] == post].iterrows()}
        common = sorted(set(pre_map) & set(post_map))
        l2s = []
        coss = []
        paired_pre_p_hacks = []
        paired_post_p_hacks = []
        paired_delta_p_hacks = []
        for pid in common:
            a = X[pre_map[pid]]
            b = X[post_map[pid]]
            l2 = float(np.linalg.norm(b - a))
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cos_dist = float(1.0 - np.dot(a, b) / denom) if denom > 0 else math.nan
            l2s.append(l2)
            coss.append(cos_dist)
            pre_p_hack = float(df.loc[pre_map[pid], "p_hack"]) if "p_hack" in df.columns else math.nan
            post_p_hack = float(df.loc[post_map[pid], "p_hack"]) if "p_hack" in df.columns else math.nan
            delta_p_hack = post_p_hack - pre_p_hack if np.isfinite(pre_p_hack) and np.isfinite(post_p_hack) else math.nan
            paired_pre_p_hacks.append(pre_p_hack)
            paired_post_p_hacks.append(post_p_hack)
            paired_delta_p_hacks.append(delta_p_hack)
            rows.append({
                "hack_type": hack_type,
                "pair_id": pid,
                "l2_shift": l2,
                "cosine_distance": cos_dist,
                "pre_p_hack": pre_p_hack,
                "post_p_hack": post_p_hack,
                "delta_p_hack": delta_p_hack,
            })

        pre_idx = [pre_map[pid] for pid in common]
        post_idx = [post_map[pid] for pid in common]
        pre_all_idx = sub[sub["checkpoint"] == pre].index.to_list()
        post_all_idx = sub[sub["checkpoint"] == post].index.to_list()
        pre_within_l2, pre_within_cos = within_distances(X[pre_all_idx])
        post_within_l2, post_within_cos = within_distances(X[post_all_idx])

        other_pre_idx = df[(df["checkpoint"] == pre) & (df["hack_type"] != hack_type)].index.to_list()
        other_post_idx = df[(df["checkpoint"] == post) & (df["hack_type"] != hack_type)].index.to_list()
        pre_cross_l2, pre_cross_cos = pair_distances(X[pre_all_idx], X[other_pre_idx])
        post_cross_l2, post_cross_cos = pair_distances(X[post_all_idx], X[other_post_idx])

        mean_l2_shift = float(np.mean(l2s)) if l2s else math.nan
        mean_cos_shift = float(np.nanmean(coss)) if coss else math.nan
        pre_within_l2_mean = finite_mean(pre_within_l2)
        post_within_l2_mean = finite_mean(post_within_l2)
        pre_cross_l2_mean = finite_mean(pre_cross_l2)
        post_cross_l2_mean = finite_mean(post_cross_l2)
        pre_within_cos_mean = finite_mean(pre_within_cos)
        post_within_cos_mean = finite_mean(post_within_cos)
        pre_cross_cos_mean = finite_mean(pre_cross_cos)
        post_cross_cos_mean = finite_mean(post_cross_cos)
        pre_p_hack_values = finite_values(sub[sub["checkpoint"] == pre]["p_hack"].tolist()) if "p_hack" in sub.columns else np.array([])
        post_p_hack_values = finite_values(sub[sub["checkpoint"] == post]["p_hack"].tolist()) if "p_hack" in sub.columns else np.array([])
        paired_delta_values = finite_values(paired_delta_p_hacks)
        paired_pre_values = finite_values(paired_pre_p_hacks)
        paired_post_values = finite_values(paired_post_p_hacks)
        fraction_p_hack_decreased = (
            float(np.mean(paired_delta_values < 0.0)) if paired_delta_values.size > 0 else math.nan
        )

        if common:
            centroid_l2 = float(np.linalg.norm(X[post_idx].mean(axis=0) - X[pre_idx].mean(axis=0)))
            summary.append(
                {
                    "hack_type": hack_type,
                    "n_pairs": len(common),
                    "mean_l2_shift": mean_l2_shift,
                    "median_l2_shift": float(np.median(l2s)),
                    "mean_cosine_distance": mean_cos_shift,
                    "median_cosine_distance": float(np.nanmedian(coss)),
                    "centroid_pre_post_l2": centroid_l2,
                    "pre_count": int((sub["checkpoint"] == pre).sum()),
                    "post_count": int((sub["checkpoint"] == post).sum()),
                    "pre_within_same_type_mean_l2": pre_within_l2_mean,
                    "pre_within_same_type_median_l2": finite_median(pre_within_l2),
                    "post_within_same_type_mean_l2": post_within_l2_mean,
                    "post_within_same_type_median_l2": finite_median(post_within_l2),
                    "pre_cross_other_type_mean_l2": pre_cross_l2_mean,
                    "pre_cross_other_type_median_l2": finite_median(pre_cross_l2),
                    "post_cross_other_type_mean_l2": post_cross_l2_mean,
                    "post_cross_other_type_median_l2": finite_median(post_cross_l2),
                    "shift_over_pre_within_l2": safe_ratio(mean_l2_shift, pre_within_l2_mean),
                    "shift_over_post_within_l2": safe_ratio(mean_l2_shift, post_within_l2_mean),
                    "shift_over_pre_cross_l2": safe_ratio(mean_l2_shift, pre_cross_l2_mean),
                    "shift_over_post_cross_l2": safe_ratio(mean_l2_shift, post_cross_l2_mean),
                    "pre_within_same_type_mean_cosine": pre_within_cos_mean,
                    "post_within_same_type_mean_cosine": post_within_cos_mean,
                    "pre_cross_other_type_mean_cosine": pre_cross_cos_mean,
                    "post_cross_other_type_mean_cosine": post_cross_cos_mean,
                    "shift_over_pre_within_cosine": safe_ratio(mean_cos_shift, pre_within_cos_mean),
                    "shift_over_post_within_cosine": safe_ratio(mean_cos_shift, post_within_cos_mean),
                    "shift_over_pre_cross_cosine": safe_ratio(mean_cos_shift, pre_cross_cos_mean),
                    "shift_over_post_cross_cosine": safe_ratio(mean_cos_shift, post_cross_cos_mean),
                    "pre_p_hack_mean": finite_mean(pre_p_hack_values),
                    "pre_p_hack_median": finite_median(pre_p_hack_values),
                    "post_p_hack_mean": finite_mean(post_p_hack_values),
                    "post_p_hack_median": finite_median(post_p_hack_values),
                    "paired_pre_p_hack_mean": finite_mean(paired_pre_values),
                    "paired_post_p_hack_mean": finite_mean(paired_post_values),
                    "delta_p_hack_mean": finite_mean(paired_delta_values),
                    "delta_p_hack_median": finite_median(paired_delta_values),
                    "fraction_p_hack_decreased": fraction_p_hack_decreased,
                }
            )
        else:
            summary.append(
                {
                    "hack_type": hack_type,
                    "n_pairs": 0,
                    "mean_l2_shift": math.nan,
                    "median_l2_shift": math.nan,
                    "mean_cosine_distance": math.nan,
                    "median_cosine_distance": math.nan,
                    "centroid_pre_post_l2": math.nan,
                    "pre_count": int((sub["checkpoint"] == pre).sum()),
                    "post_count": int((sub["checkpoint"] == post).sum()),
                    "pre_within_same_type_mean_l2": pre_within_l2_mean,
                    "pre_within_same_type_median_l2": finite_median(pre_within_l2),
                    "post_within_same_type_mean_l2": post_within_l2_mean,
                    "post_within_same_type_median_l2": finite_median(post_within_l2),
                    "pre_cross_other_type_mean_l2": pre_cross_l2_mean,
                    "pre_cross_other_type_median_l2": finite_median(pre_cross_l2),
                    "post_cross_other_type_mean_l2": post_cross_l2_mean,
                    "post_cross_other_type_median_l2": finite_median(post_cross_l2),
                    "shift_over_pre_within_l2": math.nan,
                    "shift_over_post_within_l2": math.nan,
                    "shift_over_pre_cross_l2": math.nan,
                    "shift_over_post_cross_l2": math.nan,
                    "pre_within_same_type_mean_cosine": pre_within_cos_mean,
                    "post_within_same_type_mean_cosine": post_within_cos_mean,
                    "pre_cross_other_type_mean_cosine": pre_cross_cos_mean,
                    "post_cross_other_type_mean_cosine": post_cross_cos_mean,
                    "shift_over_pre_within_cosine": math.nan,
                    "shift_over_post_within_cosine": math.nan,
                    "shift_over_pre_cross_cosine": math.nan,
                    "shift_over_post_cross_cosine": math.nan,
                    "pre_p_hack_mean": finite_mean(pre_p_hack_values),
                    "pre_p_hack_median": finite_median(pre_p_hack_values),
                    "post_p_hack_mean": finite_mean(post_p_hack_values),
                    "post_p_hack_median": finite_median(post_p_hack_values),
                    "paired_pre_p_hack_mean": math.nan,
                    "paired_post_p_hack_mean": math.nan,
                    "delta_p_hack_mean": math.nan,
                    "delta_p_hack_median": math.nan,
                    "fraction_p_hack_decreased": math.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_readable_report(
    summary: pd.DataFrame,
    out_path: Path,
    pre: str,
    post: str,
    step_trend_summary: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Paired Activation Shift Summary",
        "",
        f"Pre checkpoint: `{pre}`",
        f"Post checkpoint: `{post}`",
        "",
        "Distances are computed in the original high-dimensional activation space, not in UMAP/t-SNE space.",
        "",
        "## Key Ratios",
        "",
        "- `shift_over_pre_within_l2`: paired pre/post shift divided by same-type variation before training.",
        "- `shift_over_post_within_l2`: paired pre/post shift divided by same-type variation after training.",
        "- `shift_over_pre_cross_l2`: paired pre/post shift divided by pre-checkpoint separation from other types.",
        "- `shift_over_post_cross_l2`: paired pre/post shift divided by post-checkpoint separation from other types.",
        "",
        "Interpretation guide: ratio near `1` means comparable to the baseline; `>1.5` is substantial; `>2` is large.",
        "",
    ]

    display_cols = [
        "hack_type",
        "n_pairs",
        "mean_l2_shift",
        "median_l2_shift",
        "mean_cosine_distance",
        "pre_p_hack_mean",
        "post_p_hack_mean",
        "delta_p_hack_mean",
        "fraction_p_hack_decreased",
        "pre_within_same_type_mean_l2",
        "post_within_same_type_mean_l2",
        "pre_cross_other_type_mean_l2",
        "post_cross_other_type_mean_l2",
        "shift_over_pre_within_l2",
        "shift_over_post_within_l2",
        "shift_over_pre_cross_l2",
        "shift_over_post_cross_l2",
    ]
    available_cols = [c for c in display_cols if c in summary.columns]
    lines.extend(["## Compact Table", ""])
    lines.append("| " + " | ".join(available_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(available_cols)) + " |")
    for _, row in summary.sort_values("hack_type").iterrows():
        vals = []
        for col in available_cols:
            if col in {"hack_type", "n_pairs"}:
                vals.append(str(row[col]))
            else:
                vals.append(fmt_float(row[col]))
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend(["", "## Per-Type Details", ""])
    for _, row in summary.sort_values("hack_type").iterrows():
        ht = row["hack_type"]
        lines.extend(
            [
                f"### {ht}",
                "",
                f"- Pairs: `{row['n_pairs']}`",
                f"- Mean L2 shift: `{fmt_float(row['mean_l2_shift'])}`",
                f"- Median L2 shift: `{fmt_float(row['median_l2_shift'])}`",
                f"- Mean cosine distance: `{fmt_float(row['mean_cosine_distance'], 5)}`",
                f"- Pre p_hack mean: `{fmt_float(row.get('pre_p_hack_mean', math.nan), 5)}`",
                f"- Post p_hack mean: `{fmt_float(row.get('post_p_hack_mean', math.nan), 5)}`",
                f"- Delta p_hack mean: `{fmt_float(row.get('delta_p_hack_mean', math.nan), 5)}`",
                f"- Fraction p_hack decreased: `{fmt_float(row.get('fraction_p_hack_decreased', math.nan), 3)}`",
                f"- Shift / pre within-type L2: `{fmt_float(row['shift_over_pre_within_l2'])}`",
                f"- Shift / post within-type L2: `{fmt_float(row['shift_over_post_within_l2'])}`",
                f"- Shift / pre cross-type L2: `{fmt_float(row['shift_over_pre_cross_l2'])}`",
                f"- Shift / post cross-type L2: `{fmt_float(row['shift_over_post_cross_l2'])}`",
                "",
            ]
        )

    if step_trend_summary is not None and not step_trend_summary.empty:
        lines.extend(
            [
                "## Stepwise p_hack Trend",
                "",
                "The stepwise scatter uses one point per selected trajectory sample and is written to `p_hack_by_training_step.tsv`.",
                "Per-type scatter plots are written as `p_hack_by_training_step_<type>.png`.",
                "",
                "| checkpoint | hack_type | n_points | n_steps | first_step | last_step | first_step_p_hack_mean | last_step_p_hack_mean | delta_last_step_minus_first_step | min_p_hack | max_p_hack |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in step_trend_summary.sort_values(["hack_type", "checkpoint"]).iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["checkpoint"]),
                        str(row["hack_type"]),
                        str(int(row["n_points"])),
                        str(int(row["n_steps"])),
                        str(int(row["first_step"])),
                        str(int(row["last_step"])),
                        fmt_float(row["first_step_p_hack_mean"], 5),
                        fmt_float(row["last_step_p_hack_mean"], 5),
                        fmt_float(row["delta_last_step_minus_first_step"], 5),
                        fmt_float(row["min_p_hack"], 5),
                        fmt_float(row["max_p_hack"], 5),
                    ]
                )
                + " |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reduce_embeddings(X: np.ndarray, method: str, seed: int, pca_dims: int) -> dict[str, np.ndarray]:
    reductions: dict[str, np.ndarray] = {}
    pca2 = PCA(n_components=2, random_state=seed)
    reductions["pca"] = pca2.fit_transform(X)

    X_work = X
    if X.shape[1] > pca_dims:
        n_components = min(pca_dims, X.shape[0], X.shape[1])
        X_work = PCA(n_components=n_components, random_state=seed).fit_transform(X)

    if method in {"umap", "all"}:
        if umap is None:
            logger.warning("umap is not installed; skipping UMAP")
        else:
            reductions["umap"] = umap.UMAP(n_components=2, random_state=seed).fit_transform(X_work)

    if method in {"tsne", "all"}:
        perplexity = min(30, max(2, (X.shape[0] - 1) // 3))
        reductions["tsne"] = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(X_work)

    if method == "pca":
        return {"pca": reductions["pca"]}
    if method in reductions and method != "all":
        return {method: reductions[method]}
    return reductions


def category_style(row: pd.Series) -> str:
    return f"{row['checkpoint']}__{row['hack_type']}"


def checkpoint_colors(checkpoints: list[str], pre_checkpoint: str, post_checkpoint: str) -> dict[str, Any]:
    colors = {}
    for checkpoint in sorted(checkpoints):
        if checkpoint == pre_checkpoint:
            colors[checkpoint] = "#1f77b4"  # blue
        elif checkpoint == post_checkpoint:
            colors[checkpoint] = "#d62728"  # red
        else:
            colors[checkpoint] = "#7f7f7f"  # gray fallback
    return colors


def type_marker(hack_type: str) -> str:
    return {"unittest": "D", "exit": "X", "normal": "o", "both": "s", "neither": "P"}.get(hack_type, "o")


def checkpoint_label(checkpoint: str, pre_checkpoint: str, post_checkpoint: str) -> str:
    if checkpoint == pre_checkpoint:
        return "pre"
    if checkpoint == post_checkpoint:
        return "post"
    return checkpoint


def p_hack_range(df: pd.DataFrame) -> tuple[float, float]:
    if "p_hack" not in df.columns:
        return math.nan, math.nan
    values = pd.to_numeric(df["p_hack"], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return float(values.min()), float(values.max())


def p_hack_step_points(df: pd.DataFrame) -> pd.DataFrame:
    if "train_step" not in df.columns or "p_hack" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["train_step"] = pd.to_numeric(work["train_step"], errors="coerce")
    work["p_hack"] = pd.to_numeric(work["p_hack"], errors="coerce")
    work = work[np.isfinite(work["train_step"]) & np.isfinite(work["p_hack"])]
    if work.empty:
        return pd.DataFrame()
    work["train_step"] = work["train_step"].astype(int)
    cols = [
        "checkpoint",
        "hack_type",
        "train_step",
        "p_hack",
        "activation_id",
        "pair_id",
        "traj_source",
    ]
    return work[[col for col in cols if col in work.columns]].sort_values(
        ["hack_type", "checkpoint", "train_step", "activation_id"]
    )


def p_hack_step_trend_summary(step_points: pd.DataFrame) -> pd.DataFrame:
    if step_points.empty:
        return pd.DataFrame()
    rows = []
    for (checkpoint, hack_type), sub in step_points.groupby(["checkpoint", "hack_type"], sort=True):
        sub = sub.sort_values("train_step")
        first_step = int(sub["train_step"].min())
        last_step = int(sub["train_step"].max())
        first_values = sub[sub["train_step"] == first_step]["p_hack"]
        last_values = sub[sub["train_step"] == last_step]["p_hack"]
        rows.append(
            {
                "checkpoint": checkpoint,
                "hack_type": hack_type,
                "n_points": int(len(sub)),
                "n_steps": int(sub["train_step"].nunique()),
                "first_step": first_step,
                "last_step": last_step,
                "first_step_p_hack_mean": float(first_values.mean()),
                "last_step_p_hack_mean": float(last_values.mean()),
                "delta_last_step_minus_first_step": float(last_values.mean() - first_values.mean()),
                "min_p_hack": float(sub["p_hack"].min()),
                "max_p_hack": float(sub["p_hack"].max()),
            }
        )
    return pd.DataFrame(rows)


def color_with_p_hack(base_color: Any, p_hack: Any, vmin: float, vmax: float) -> tuple[float, float, float]:
    base = np.array(mcolors.to_rgb(base_color))
    try:
        value = float(p_hack)
    except (TypeError, ValueError):
        value = math.nan
    if not np.isfinite(value) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        strength = 0.75
    else:
        strength = 0.35 + 0.65 * np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0)
    white = np.ones(3)
    return tuple((white * (1.0 - strength) + base * strength).tolist())


def type_linestyle(hack_type: str) -> Any:
    return {
        "normal": "-",
        "unittest": "--",
        "exit": ":",
        "both": "-.",
        "neither": (0, (4, 2, 1, 2)),
    }.get(hack_type, "-")


def plot_p_hack_by_step(step_points: pd.DataFrame, pre: str, post: str, out_dir: Path, dpi: int) -> None:
    if step_points.empty:
        logger.warning("No train_step/p_hack data available; skipping p_hack step plot")
        return
    colors = checkpoint_colors(step_points["checkpoint"].unique().tolist(), pre, post)

    for hack_type, type_df in step_points.groupby("hack_type", sort=True):
        hack_type = str(hack_type)
        fig, ax = plt.subplots(figsize=(10, 6))
        for checkpoint, sub in type_df.groupby("checkpoint", sort=True):
            checkpoint = str(checkpoint)
            sub = sub.sort_values("train_step")
            ax.scatter(
                sub["train_step"],
                sub["p_hack"],
                color=colors.get(checkpoint, "#7f7f7f"),
                s=42,
                alpha=0.82,
                marker=type_marker(hack_type),
                edgecolors="black",
                linewidths=0.35,
                label=checkpoint_label(checkpoint, pre, post),
            )
        ax.set_title(f"Probe p_hack by Training Step ({hack_type})")
        ax.set_xlabel("Training step")
        ax.set_ylabel("p_hack")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"p_hack_by_training_step_{hack_type}.png", dpi=dpi)
        plt.close(fig)


def plot_scatter(df: pd.DataFrame, coords: np.ndarray, pre: str, post: str, out_path: Path, title: str, dpi: int) -> None:
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    colors = checkpoint_colors(plot_df["checkpoint"].unique().tolist(), pre, post)
    phack_min, phack_max = p_hack_range(plot_df)

    fig, ax = plt.subplots(figsize=(11, 8))
    for (checkpoint, hack_type), sub in plot_df.groupby(["checkpoint", "hack_type"], sort=True):
        checkpoint = str(checkpoint)
        hack_type = str(hack_type)
        point_colors = [
            color_with_p_hack(colors[checkpoint], value, phack_min, phack_max)
            for value in sub.get("p_hack", pd.Series([math.nan] * len(sub))).tolist()
        ]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=32,
            alpha=0.74,
            label=f"{checkpoint_label(checkpoint, pre, post)} / {hack_type}",
            c=point_colors,
            marker=type_marker(hack_type),
            edgecolors="black",
            linewidths=0.35,
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_paired_shift(df: pd.DataFrame, coords: np.ndarray, pre: str, post: str, out_path: Path, title: str, dpi: int) -> None:
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    colors = checkpoint_colors([pre, post], pre, post)
    phack_min, phack_max = p_hack_range(plot_df)
    fig, ax = plt.subplots(figsize=(11, 8))

    for hack_type, sub in plot_df.groupby("hack_type"):
        hack_type = str(hack_type)
        pre_sub = sub[sub["checkpoint"] == pre].set_index("pair_id")
        post_sub = sub[sub["checkpoint"] == post].set_index("pair_id")
        common = sorted(set(pre_sub.index) & set(post_sub.index))
        for pid in common:
            a = pre_sub.loc[pid]
            b = post_sub.loc[pid]
            ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color="#777777", alpha=0.16, linewidth=0.7)
        if common:
            marker = type_marker(hack_type)
            pre_colors = [color_with_p_hack(colors[pre], value, phack_min, phack_max) for value in pre_sub.loc[common, "p_hack"].tolist()]
            post_colors = [color_with_p_hack(colors[post], value, phack_min, phack_max) for value in post_sub.loc[common, "p_hack"].tolist()]
            ax.scatter(pre_sub.loc[common, "x"], pre_sub.loc[common, "y"], s=36, marker=marker, c=pre_colors, alpha=0.72, edgecolors="black", linewidths=0.3, label=f"pre / {hack_type}")
            ax.scatter(post_sub.loc[common, "x"], post_sub.loc[common, "y"], s=42, marker=marker, c=post_colors, alpha=0.84, edgecolors="black", linewidths=0.3, label=f"post / {hack_type}")

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_p_hack_heatmap(df: pd.DataFrame, coords: np.ndarray, out_path: Path, title: str, dpi: int) -> None:
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    values = pd.to_numeric(plot_df.get("p_hack", pd.Series([math.nan] * len(plot_df))), errors="coerce")
    mask = values.notna()
    fig, ax = plt.subplots(figsize=(11, 8))
    if (~mask).any():
        ax.scatter(plot_df.loc[~mask, "x"], plot_df.loc[~mask, "y"], s=20, alpha=0.25, color="#bbbbbb", label="missing p_hack")
    scatter = ax.scatter(
        plot_df.loc[mask, "x"],
        plot_df.loc[mask, "y"],
        c=values.loc[mask],
        cmap="magma",
        s=34,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.25,
    )
    fig.colorbar(scatter, ax=ax, label="p_hack")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if (~mask).any():
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_paired_shift_by_type(df: pd.DataFrame, coords: np.ndarray, pre: str, post: str, out_dir: Path, method_name: str, dpi: int) -> None:
    for hack_type in sorted(df["hack_type"].unique()):
        type_mask = df["hack_type"] == hack_type
        type_df = df[type_mask].reset_index(drop=True)
        type_coords = coords[type_mask.to_numpy()]
        if len(type_df) == 0:
            continue
        plot_paired_shift(
            type_df,
            type_coords,
            pre,
            post,
            out_dir / f"activation_shift_{method_name}_{hack_type}_paired.png",
            f"Paired Activation Shift ({method_name.upper()}, {hack_type})",
            dpi,
        )


def write_html(df: pd.DataFrame, coords: np.ndarray, out_path: Path, title: str) -> None:
    try:
        import plotly.express as px
    except Exception:
        logger.warning("plotly is not installed; skipping HTML")
        return
    plot_df = df.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    plot_df["category"] = plot_df.apply(category_style, axis=1)
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="category",
        symbol="hack_type",
        hover_data=["activation_id", "pair_id", "checkpoint", "hack_type", "traj_source", "p_hack"],
        title=title,
    )
    fig.write_html(out_path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specs = infer_specs(args)
    if not specs:
        raise SystemExit("No datasets found.")
    logger.info("Datasets:")
    for spec in specs:
        logger.info(f"  {spec.checkpoint} / {spec.hack_type}: {spec.path}")

    xs_all: list[np.ndarray] = []
    rows_all: list[dict[str, Any]] = []
    checkpoints = sorted({spec.checkpoint for spec in specs})
    for spec in specs:
        xs, rows = load_dataset(spec)
        xs_all.extend(xs)
        rows_all.extend(rows)

    if not xs_all:
        raise SystemExit("No activations loaded.")
    X = np.stack(xs_all)
    df = pd.DataFrame(rows_all)
    df["pair_id"] = df["activation_id"].map(lambda x: normalize_pair_id(str(x), checkpoints))
    df, X = cap_pairs(df, X, args.max_pairs_per_type, args.seed)
    logger.info(f"Loaded {len(df)} activations with dim={X.shape[1]}")

    # Full-space stats are the most direct quantitative signal.
    pair_stats, summary = pairwise_shift_stats(df, X, args.pre_checkpoint, args.post_checkpoint)
    step_points = p_hack_step_points(df)
    step_trend_summary = p_hack_step_trend_summary(step_points)
    df.to_csv(args.output_dir / "activation_metadata.tsv", sep="\t", index=False)
    pair_stats.to_csv(args.output_dir / "paired_activation_shift_per_sample.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "paired_activation_shift_summary.tsv", sep="\t", index=False)
    if not step_points.empty:
        step_points.to_csv(args.output_dir / "p_hack_by_training_step.tsv", sep="\t", index=False)
        step_trend_summary.to_csv(args.output_dir / "p_hack_by_training_step_summary.tsv", sep="\t", index=False)
        plot_p_hack_by_step(step_points, args.pre_checkpoint, args.post_checkpoint, args.output_dir, args.dpi)
    write_readable_report(
        summary,
        args.output_dir / "paired_activation_shift_report.md",
        args.pre_checkpoint,
        args.post_checkpoint,
        step_trend_summary,
    )
    logger.info(f"Wrote shift summary to {args.output_dir / 'paired_activation_shift_summary.tsv'}")

    reductions = reduce_embeddings(X, args.method, args.seed, args.pca_dims)
    for name, coords in reductions.items():
        coord_df = df.copy()
        coord_df["dim_1"] = coords[:, 0]
        coord_df["dim_2"] = coords[:, 1]
        coord_df.to_csv(args.output_dir / f"activation_{name}_coords.tsv", sep="\t", index=False)
        plot_scatter(df, coords, args.pre_checkpoint, args.post_checkpoint, args.output_dir / f"activation_shift_{name}_scatter.png", f"Activation Shift Scatter ({name.upper()})", args.dpi)
        plot_paired_shift(df, coords, args.pre_checkpoint, args.post_checkpoint, args.output_dir / f"activation_shift_{name}_paired.png", f"Paired Activation Shift ({name.upper()})", args.dpi)
        plot_paired_shift_by_type(df, coords, args.pre_checkpoint, args.post_checkpoint, args.output_dir, name, args.dpi)
        plot_p_hack_heatmap(df, coords, args.output_dir / f"activation_shift_{name}_phack_heatmap.png", f"Probe p_hack Heatmap ({name.upper()})", args.dpi)
        if args.html:
            write_html(df, coords, args.output_dir / f"activation_shift_{name}.html", f"Activation Shift ({name.upper()})")

    logger.success(f"Done. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
