#!/usr/bin/env python3
"""Compare TA/synthetic hacking text against real inference hacking with text embeddings.

Default grouping is tailored to /nfs/data/lichenli/cot-monitor-modified/inference/exp_data:
  real      : I-N*.jsonl
  ta        : ckpt61*.jsonl
  synthetic : 7b_syn*.jsonl

For each text view (full response, reasoning without fenced code, final python code), the
script embeds rows once and reports source-to-real similarity metrics, including paired-real
cosine using (data_source, extra_info.index).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PY_BLOCK_RE = re.compile(r"```(?:python|py)\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ALL_FENCED_RE = re.compile(r"```.*?```", re.DOTALL)

TEXT_VIEWS = ("full_response", "reasoning_no_code", "final_python_code")
SOURCE_ORDER = ("ta", "synthetic")


@dataclass
class RowRecord:
    source: str
    source_file: str
    source_line: int
    pair_key: str
    response: str
    full_response: str
    reasoning_no_code: str
    final_python_code: str
    has_python_block: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/nfs/data/lichenli/cot-monitor-modified/inference/exp_data"),
        help="Directory containing real/TA/synthetic JSONL files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data_dir>/../text_similarity_reports.",
    )
    parser.add_argument("--real_patterns", nargs="+", default=["I-N*.jsonl"])
    parser.add_argument("--ta_patterns", nargs="+", default=["ckpt61*.jsonl"])
    parser.add_argument("--synthetic_patterns", nargs="+", default=["7b_syn*.jsonl"])
    parser.add_argument("--model_name", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--mmd_gamma", type=float, default=None, help="RBF gamma. Default: median heuristic per comparison.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def find_files(data_dir: Path, patterns: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        matches = sorted(data_dir.glob(pattern))
        files.extend(matches)
    seen = set()
    unique = []
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def get_response(row: Dict[str, Any]) -> str:
    for key in ("response", "completion", "text", "output"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    value = row.get("responses")
    if isinstance(value, list):
        strings = [x for x in value if isinstance(x, str)]
        if strings:
            return strings[0]
    if isinstance(value, str):
        return value
    return ""


def pair_key_from_row(row: Dict[str, Any]) -> str:
    data_source = row.get("data_source", "")
    extra_info = row.get("extra_info")
    index = None
    if isinstance(extra_info, dict):
        index = extra_info.get("index")
    if data_source != "" and index is not None:
        return f"{data_source}::{index}"

    # Fallback for malformed rows. It should rarely be used for these files.
    prompt = row.get("prompt", "")
    prompt_repr = json.dumps(prompt, sort_keys=True, ensure_ascii=False) if not isinstance(prompt, str) else prompt
    digest = hashlib.sha1(prompt_repr.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"prompt_sha1::{digest}"


def extract_text_views(response: str) -> Tuple[str, str, str, bool]:
    full = response.strip()
    reasoning = ALL_FENCED_RE.sub("", response).strip()
    py_blocks = PY_BLOCK_RE.findall(response)
    final_code = py_blocks[-1].strip() if py_blocks else ""
    return full, reasoning, final_code, bool(py_blocks)


def read_jsonl(path: Path, source: str) -> List[RowRecord]:
    records: List[RowRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse {path}:{line_no}: {exc}") from exc
            response = get_response(row)
            full, reasoning, final_code, has_py = extract_text_views(response)
            records.append(
                RowRecord(
                    source=source,
                    source_file=str(path),
                    source_line=line_no,
                    pair_key=pair_key_from_row(row),
                    response=response,
                    full_response=full,
                    reasoning_no_code=reasoning,
                    final_python_code=final_code,
                    has_python_block=has_py,
                )
            )
    return records


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def encode_with_sentence_transformers(
    texts: Sequence[str], model_name: str, device: str, batch_size: int, max_length: int
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    # SentenceTransformer exposes max_seq_length for most models.
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length
    safe_texts = [text if text.strip() else " " for text in texts]
    emb = model.encode(
        safe_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(emb, dtype=np.float32)


def encode_with_transformers(
    texts: Sequence[str], model_name: str, device: str, batch_size: int, max_length: int
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    outputs: List[np.ndarray] = []
    safe_texts = [text if text.strip() else " " for text in texts]
    with torch.no_grad():
        for start in range(0, len(safe_texts), batch_size):
            batch = safe_texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def embed_texts(texts: Sequence[str], model_name: str, device: str, batch_size: int, max_length: int) -> np.ndarray:
    try:
        return encode_with_sentence_transformers(texts, model_name, device, batch_size, max_length)
    except ImportError:
        print("sentence_transformers not installed; falling back to transformers mean pooling.", file=sys.stderr)
        return encode_with_transformers(texts, model_name, device, batch_size, max_length)


def pairwise_sq_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def pairwise_l2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(pairwise_sq_l2(x, y))


def median_heuristic_gamma(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.concatenate([x, y], axis=0)
    d2 = pairwise_sq_l2(combined, combined)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 1.0
    median = float(np.median(vals))
    return 1.0 / (2.0 * median) if median > 0 else 1.0


def rbf_mmd2(x: np.ndarray, y: np.ndarray, gamma: Optional[float]) -> Tuple[float, float]:
    used_gamma = median_heuristic_gamma(x, y) if gamma is None else gamma
    k_xx = np.exp(-used_gamma * pairwise_sq_l2(x, x)).mean()
    k_yy = np.exp(-used_gamma * pairwise_sq_l2(y, y)).mean()
    k_xy = np.exp(-used_gamma * pairwise_sq_l2(x, y)).mean()
    return float(k_xx + k_yy - 2.0 * k_xy), float(used_gamma)


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(2.0 * pairwise_l2(x, y).mean() - pairwise_l2(x, x).mean() - pairwise_l2(y, y).mean())


def format_float(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def compute_metrics(
    source_records: Sequence[RowRecord],
    source_emb: np.ndarray,
    real_records: Sequence[RowRecord],
    real_emb: np.ndarray,
    gamma: Optional[float],
) -> Dict[str, Any]:
    cos = source_emb @ real_emb.T
    nearest = cos.max(axis=1)
    avg = cos.mean(axis=1)
    mmd2, used_gamma = rbf_mmd2(source_emb, real_emb, gamma)
    energy = energy_distance(source_emb, real_emb)

    real_by_pair: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in enumerate(real_records):
        real_by_pair[rec.pair_key].append(idx)

    paired_values: List[float] = []
    paired_match_rows = 0
    for i, rec in enumerate(source_records):
        real_indices = real_by_pair.get(rec.pair_key, [])
        if not real_indices:
            continue
        paired_match_rows += 1
        # If there are multiple real rows for one prompt, average over those real responses.
        paired_values.append(float(cos[i, real_indices].mean()))

    if paired_values:
        paired_arr = np.asarray(paired_values, dtype=np.float32)
        paired_mean = float(paired_arr.mean())
        paired_median = float(np.median(paired_arr))
    else:
        paired_mean = float("nan")
        paired_median = float("nan")

    return {
        "n_source": len(source_records),
        "n_real": len(real_records),
        "mean_nearest_real_cosine": float(nearest.mean()),
        "median_nearest_real_cosine": float(np.median(nearest)),
        "mean_average_real_cosine": float(avg.mean()),
        "median_average_real_cosine": float(np.median(avg)),
        "mean_paired_real_cosine": paired_mean,
        "median_paired_real_cosine": paired_median,
        "paired_real_match_count": paired_match_rows,
        "paired_real_coverage": paired_match_rows / len(source_records) if source_records else 0.0,
        "mmd_rbf2_to_real": mmd2,
        "mmd_rbf_gamma": used_gamma,
        "energy_distance_to_real": energy,
    }


def diagnostics(records: Sequence[RowRecord]) -> Dict[str, Any]:
    unique_pairs = {rec.pair_key for rec in records}
    response_count = sum(1 for rec in records if rec.response.strip())
    python_count = sum(1 for rec in records if rec.has_python_block)
    empty_code_count = sum(1 for rec in records if not rec.final_python_code.strip())
    empty_reasoning_count = sum(1 for rec in records if not rec.reasoning_no_code.strip())
    return {
        "rows": len(records),
        "unique_pair_keys": len(unique_pairs),
        "with_response": response_count,
        "with_python_block": python_count,
        "empty_final_python_code": empty_code_count,
        "empty_reasoning_no_code": empty_reasoning_count,
    }


def write_tsv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                val = row.get(key, "")
                if isinstance(val, float):
                    out[key] = format_float(val)
                else:
                    out[key] = val
            writer.writerow(out)




def _xml_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _xlsx_col_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _xlsx_cell(row_idx: int, col_idx: int, value: Any, style: Optional[int] = None) -> str:
    ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
    style_attr = f' s="{style}"' if style is not None else ""
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return f'<c r="{ref}"{style_attr}><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t>{_xml_escape(value)}</t></is></c>'


def write_compact_xlsx(path: Path, summary_rows: Sequence[Dict[str, Any]]) -> None:
    """Write the three rebuttal-facing metric tables into one xlsx sheet.

    This uses only stdlib zip/XML so the analysis script does not require pandas/openpyxl.
    """
    table_order = [
        ("final_python_code", "Final Python Code"),
        ("reasoning_no_code", "Reasoning Only"),
        ("full_response", "Full Response"),
    ]
    headers = ["Source", "Mean NN", "Mean Avg", "Mean Paired", "MMD", "Energy"]
    metric_keys = [
        "source",
        "mean_nearest_real_cosine",
        "mean_average_real_cosine",
        "mean_paired_real_cosine",
        "mmd_rbf2_to_real",
        "energy_distance_to_real",
    ]
    rows_xml: List[str] = []
    row_idx = 1
    for view, title in table_order:
        view_rows = [row for row in summary_rows if row.get("view") == view]
        if not view_rows:
            continue
        rows_xml.append(
            f'<row r="{row_idx}">' + _xlsx_cell(row_idx, 0, title, style=1) + "</row>"
        )
        row_idx += 1
        rows_xml.append(
            f'<row r="{row_idx}">' + "".join(_xlsx_cell(row_idx, i, h, style=2) for i, h in enumerate(headers)) + "</row>"
        )
        row_idx += 1
        for row in view_rows:
            values: List[Any] = []
            for key in metric_keys:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = round(value, 6)
                values.append(value)
            rows_xml.append(
                f'<row r="{row_idx}">' + "".join(_xlsx_cell(row_idx, i, v) for i, v in enumerate(values)) + "</row>"
            )
            row_idx += 1
        row_idx += 2

    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <cols>
    <col min="1" max="1" width="14" customWidth="1"/>
    <col min="2" max="6" width="16" customWidth="1"/>
  </cols>
  <sheetData>
%s
  </sheetData>
</worksheet>
""" % "\n".join(rows_xml)

    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Similarity Tables" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>
"""
    workbook_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><sz val="14"/><name val="Calibri"/></font>
    <font><b/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>
  </cellXfs>
</styleSheet>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        zf.writestr("xl/styles.xml", styles_xml)

def write_report(path: Path, summary_rows: Sequence[Dict[str, Any]], diag_rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    metric_fields = [
        "view",
        "source",
        "n_source",
        "n_real",
        "mean_nearest_real_cosine",
        "median_nearest_real_cosine",
        "mean_average_real_cosine",
        "median_average_real_cosine",
        "mean_paired_real_cosine",
        "median_paired_real_cosine",
        "paired_real_match_count",
        "paired_real_coverage",
        "mmd_rbf2_to_real",
        "energy_distance_to_real",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Text Embedding Similarity to Real Hacking\n\n")
        f.write(f"- data_dir: `{args.data_dir}`\n")
        f.write(f"- model: `{args.model_name}`\n")
        f.write(f"- device: `{args.device}`\n")
        f.write("- paired key: `(data_source, extra_info.index)`\n")
        f.write("- final code extraction: last fenced block beginning with ```python or ```py; missing blocks are embedded as empty text and reported.\n\n")

        f.write("## Metrics\n\n")
        f.write("| View | Source | N | Mean NN | Median NN | Mean Avg | Median Avg | Mean Paired | Median Paired | Paired Coverage | MMD RBF^2 | Energy |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {view} | {source} | {n_source} | {mean_nearest_real_cosine} | {median_nearest_real_cosine} | "
                "{mean_average_real_cosine} | {median_average_real_cosine} | {mean_paired_real_cosine} | "
                "{median_paired_real_cosine} | {paired_real_coverage} | {mmd_rbf2_to_real} | {energy_distance_to_real} |\n".format(
                    **{k: (format_float(v) if isinstance(v, float) else v) for k, v in row.items()}
                )
            )

        f.write("\n## Extraction Diagnostics\n\n")
        f.write("| Source | Rows | Unique pairs | With response | With Python block | Empty final code | Empty reasoning |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in diag_rows:
            f.write(
                f"| {row['source']} | {row['rows']} | {row['unique_pair_keys']} | {row['with_response']} | "
                f"{row['with_python_block']} | {row['empty_final_python_code']} | {row['empty_reasoning_no_code']} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- Larger cosine metrics mean closer to real inference hacking.\n")
        f.write("- Smaller MMD and energy distance mean closer embedding distributions to real inference hacking.\n")
        f.write("- Paired-real cosine compares only rows sharing the same `(data_source, extra_info.index)` with real data.\n")


def main() -> None:
    args = parse_args()
    global np
    import numpy as np
    if args.output_dir is None:
        args.output_dir = args.data_dir.parent / "text_similarity_reports"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.device = resolve_device(args.device)

    file_groups = {
        "real": find_files(args.data_dir, args.real_patterns),
        "ta": find_files(args.data_dir, args.ta_patterns),
        "synthetic": find_files(args.data_dir, args.synthetic_patterns),
    }
    missing = [name for name, files in file_groups.items() if not files]
    if missing:
        raise SystemExit(f"Missing files for source group(s): {missing}. data_dir={args.data_dir}")

    records_by_source: Dict[str, List[RowRecord]] = {}
    for source, files in file_groups.items():
        records: List[RowRecord] = []
        for path in files:
            records.extend(read_jsonl(path, source))
        records_by_source[source] = records
        print(f"Loaded {len(records)} {source} rows from {len(files)} file(s).")

    diag_rows = []
    for source in ("real", "ta", "synthetic"):
        row = {"source": source, **diagnostics(records_by_source[source])}
        diag_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    for view in TEXT_VIEWS:
        print(f"\nEmbedding view: {view}")
        texts: List[str] = []
        spans: Dict[str, Tuple[int, int]] = {}
        cursor = 0
        for source in ("real", "ta", "synthetic"):
            source_texts = [getattr(rec, view) for rec in records_by_source[source]]
            texts.extend(source_texts)
            spans[source] = (cursor, cursor + len(source_texts))
            cursor += len(source_texts)

        embeddings = embed_texts(texts, args.model_name, args.device, args.batch_size, args.max_length)
        emb_by_source = {
            source: embeddings[start:end]
            for source, (start, end) in spans.items()
        }

        for source in SOURCE_ORDER:
            metrics = compute_metrics(
                records_by_source[source],
                emb_by_source[source],
                records_by_source["real"],
                emb_by_source["real"],
                args.mmd_gamma,
            )
            summary_rows.append({"view": view, "source": source, **metrics})

    summary_fields = [
        "view",
        "source",
        "n_source",
        "n_real",
        "mean_nearest_real_cosine",
        "median_nearest_real_cosine",
        "mean_average_real_cosine",
        "median_average_real_cosine",
        "mean_paired_real_cosine",
        "median_paired_real_cosine",
        "paired_real_match_count",
        "paired_real_coverage",
        "mmd_rbf2_to_real",
        "mmd_rbf_gamma",
        "energy_distance_to_real",
    ]
    diag_fields = [
        "source",
        "rows",
        "unique_pair_keys",
        "with_response",
        "with_python_block",
        "empty_final_python_code",
        "empty_reasoning_no_code",
    ]

    write_tsv(args.output_dir / "text_embedding_similarity_summary.tsv", summary_rows, summary_fields)
    write_tsv(args.output_dir / "text_embedding_similarity_diagnostics.tsv", diag_rows, diag_fields)
    write_compact_xlsx(args.output_dir / "text_embedding_similarity_tables.xlsx", summary_rows)
    with (args.output_dir / "text_embedding_similarity_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary_rows, "diagnostics": diag_rows, "args": vars(args)}, f, ensure_ascii=False, indent=2, default=str)
    write_report(args.output_dir / "text_embedding_similarity_report.md", summary_rows, diag_rows, args)

    print(f"\nWrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
