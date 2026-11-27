"""Strict CSV schema, edge integrity, and timestep validation for Stage 6 uploads."""

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.backend.config import (
    MAX_CSV_BYTES,
    MAX_EDGES_PER_UPLOAD,
    MAX_NODES_PER_UPLOAD,
    load_feature_columns,
)


@dataclass
class ValidatedDataset:
    node_count: int
    edge_count_raw: int
    unique_edges_after_dedup: int
    duplicate_edges_removed: int
    timesteps: List[int]
    imputed_nan_cells: int
    has_optional_class_column: bool
    class_counts: Dict[str, int]
    warnings: List[str]
    tx_df: pd.DataFrame          # sorted by ['Time step', 'txId'], retains 'node_order'
    feature_columns: List[str]
    X_raw: np.ndarray             # float64 [N, 182] in tx_df row order
    y_optional: np.ndarray        # int64 [N] (-1 Unknown, 0 Licit, 1 Illicit)
    edge_pairs: pd.DataFrame      # deduplicated ['src_tx', 'dst_tx', 'timestep']


def _parse_class_value(val: Any) -> int:
    if pd.isna(val):
        return -1
    s = str(val).strip().lower()
    if s in ("", "unknown", "nan", "-1", "-1.0", "3", "3.0"):
        return -1
    if s in ("1", "1.0", "illicit"):
        return 1
    if s in ("2", "2.0", "0", "0.0", "licit"):
        return 0
    raise ValueError(f"Unsupported class label value: {val!r}")


def _read_csv_bytes(raw_bytes: bytes, label: str) -> pd.DataFrame:
    if not raw_bytes:
        raise HTTPException(status_code=422, detail={"error": "empty_file", "message": f"{label} file is empty."})
    if len(raw_bytes) > MAX_CSV_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "file_too_large", "message": f"{label} exceeds {MAX_CSV_BYTES // (1024 * 1024)} MB limit."},
        )
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_encoding", "message": f"{label} must be UTF-8 encoded CSV: {exc}"},
        )
    try:
        df = pd.read_csv(io.StringIO(text), low_memory=False)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "malformed_csv", "message": f"Failed to parse {label} CSV: {exc}"},
        )
    if df.empty:
        raise HTTPException(
            status_code=422,
            detail={"error": "empty_table", "message": f"{label} CSV contains zero data rows."},
        )
    # Detect pandas auto-mangled duplicate columns if header repeated exact names
    if df.columns.duplicated().any():
        raise HTTPException(
            status_code=422,
            detail={"error": "duplicate_columns", "message": f"{label} CSV contains duplicate column headers."},
        )
    return df


def _parse_strict_int_series(series: pd.Series, col_name: str, file_label: str) -> np.ndarray:
    if series.isna().any():
        bad_count = int(series.isna().sum())
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_required_ids",
                "message": f"{file_label} column '{col_name}' contains {bad_count} missing/NaN value(s).",
            },
        )
    num = pd.to_numeric(series, errors="coerce")
    if num.isna().any() or not np.isfinite(num.to_numpy(dtype=np.float64)).all():
        raise HTTPException(
            status_code=422,
            detail={
                "error": "non_numeric_identifier",
                "message": f"{file_label} column '{col_name}' contains non-numeric or non-finite values.",
            },
        )
    arr_f = num.to_numpy(dtype=np.float64)
    if not np.all(np.floor(arr_f) == arr_f):
        raise HTTPException(
            status_code=422,
            detail={
                "error": "non_integer_identifier",
                "message": f"{file_label} column '{col_name}' must contain integer values only.",
            },
        )
    return arr_f.astype(np.int64)


def validate_csv_uploads(
    tx_bytes: bytes,
    edge_bytes: bytes,
    class_bytes: Optional[bytes] = None,
) -> ValidatedDataset:
    feature_cols = load_feature_columns()
    tx_df = _read_csv_bytes(tx_bytes, "transactions")
    edge_df = _read_csv_bytes(edge_bytes, "edges")

    # 1. Check required transaction columns
    required_tx = ["txId", "Time step"] + feature_cols
    missing_tx_cols = [c for c in required_tx if c not in tx_df.columns]
    if missing_tx_cols:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "missing_transaction_columns",
                "message": f"transactions CSV is missing {len(missing_tx_cols)} required column(s).",
                "missing_count": len(missing_tx_cols),
                "missing_sample": missing_tx_cols[:15],
            },
        )

    if len(tx_df) > MAX_NODES_PER_UPLOAD:
        raise HTTPException(
            status_code=422,
            detail={"error": "row_limit_exceeded", "message": f"Transactions exceed {MAX_NODES_PER_UPLOAD:,} row limit."},
        )

    # 2. Validate txId and Time step
    tx_ids = _parse_strict_int_series(tx_df["txId"], "txId", "transactions")
    timesteps = _parse_strict_int_series(tx_df["Time step"], "Time step", "transactions")

    if (timesteps < 1).any():
        bad_steps = sorted(set(timesteps[timesteps < 1].tolist()))[:10]
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_timestep", "message": f"Time step values must be integers >= 1; got {bad_steps}."},
        )

    unique_ids, counts = np.unique(tx_ids, return_counts=True)
    if (counts > 1).any():
        dup_sample = unique_ids[counts > 1][:10].tolist()
        raise HTTPException(
            status_code=422,
            detail={
                "error": "duplicate_tx_id",
                "message": f"transactions CSV contains {int((counts > 1).sum())} duplicate txId(s).",
                "duplicate_sample": dup_sample,
            },
        )

    # 3. Validate 182 numerical feature columns (NaN allowed -> median imputed; non-numeric or Inf -> 422)
    X_cols: List[np.ndarray] = []
    non_numeric_cols: List[str] = []
    inf_cols: List[str] = []
    nan_cells = 0

    for col in feature_cols:
        s = tx_df[col]
        if pd.api.types.is_numeric_dtype(s):
            arr = s.to_numpy(dtype=np.float64)
        else:
            was_na = s.isna()
            parsed = pd.to_numeric(s, errors="coerce")
            newly_na = parsed.isna() & ~was_na
            if newly_na.any():
                non_numeric_cols.append(col)
                continue
            arr = parsed.to_numpy(dtype=np.float64)
        if np.isinf(arr).any():
            inf_cols.append(col)
            continue
        nan_cells += int(np.isnan(arr).sum())
        X_cols.append(arr)

    if non_numeric_cols:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "non_numeric_features",
                "message": f"{len(non_numeric_cols)} feature column(s) contain non-numeric entries.",
                "columns": non_numeric_cols[:15],
            },
        )
    if inf_cols:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "infinite_features",
                "message": f"{len(inf_cols)} feature column(s) contain +/-Inf values.",
                "columns": inf_cols[:15],
            },
        )

    X_raw_unsorted = np.column_stack(X_cols) if len(tx_df) > 0 else np.empty((0, 182), dtype=np.float64)

    # 4. Handle optional class column cleanly (strictly isolated from X_raw)
    has_optional_class = False
    y_unsorted = np.full(len(tx_df), -1, dtype=np.int64)

    if "class" in tx_df.columns:
        has_optional_class = True
        try:
            y_unsorted = np.array([_parse_class_value(v) for v in tx_df["class"].tolist()], dtype=np.int64)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"error": "invalid_class_value", "message": str(exc)})

    if class_bytes is not None and len(class_bytes) > 0:
        cls_df = _read_csv_bytes(class_bytes, "classes")
        if "txId" not in cls_df.columns or "class" not in cls_df.columns:
            raise HTTPException(
                status_code=422,
                detail={"error": "missing_class_columns", "message": "classes CSV must contain 'txId' and 'class' columns."},
            )
        cls_tx_ids = _parse_strict_int_series(cls_df["txId"], "txId", "classes")
        try:
            cls_vals = [_parse_class_value(v) for v in cls_df["class"].tolist()]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail={"error": "invalid_class_value", "message": str(exc)})
        cls_map = dict(zip(cls_tx_ids.tolist(), cls_vals))
        has_optional_class = True
        y_unsorted = np.array([cls_map.get(int(tid), -1) for tid in tx_ids], dtype=np.int64)

    # 5. Validate directed edges CSV (strictly NO silent dropping)
    if "txId1" not in edge_df.columns or "txId2" not in edge_df.columns:
        raise HTTPException(
            status_code=422,
            detail={"error": "missing_edge_columns", "message": "edges CSV must contain 'txId1' and 'txId2' columns."},
        )
    if len(edge_df) > MAX_EDGES_PER_UPLOAD:
        raise HTTPException(
            status_code=422,
            detail={"error": "edge_limit_exceeded", "message": f"Edges exceed {MAX_EDGES_PER_UPLOAD:,} row limit."},
        )

    src_arr = _parse_strict_int_series(edge_df["txId1"], "txId1", "edges")
    dst_arr = _parse_strict_int_series(edge_df["txId2"], "txId2", "edges")

    tx_to_step = dict(zip(tx_ids.tolist(), timesteps.tolist()))
    dangling_sample: List[Dict[str, int]] = []
    dangling_count = 0
    cross_sample: List[Dict[str, Any]] = []
    cross_count = 0

    edge_steps = np.empty(len(src_arr), dtype=np.int64)
    for i in range(len(src_arr)):
        s_id = int(src_arr[i])
        d_id = int(dst_arr[i])
        s_step = tx_to_step.get(s_id)
        d_step = tx_to_step.get(d_id)
        if s_step is None or d_step is None:
            dangling_count += 1
            if len(dangling_sample) < 10:
                dangling_sample.append({"txId1": s_id, "txId2": d_id})
            continue
        if s_step != d_step:
            cross_count += 1
            if len(cross_sample) < 10:
                cross_sample.append({"txId1": s_id, "step1": s_step, "txId2": d_id, "step2": d_step})
            continue
        edge_steps[i] = s_step

    if dangling_count > 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "dangling_edges",
                "message": (
                    f"{dangling_count} edge(s) reference txId endpoint(s) missing from transactions CSV. "
                    "Edges are never silently dropped."
                ),
                "dangling_edge_count": dangling_count,
                "sample": dangling_sample,
            },
        )
    if cross_count > 0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "cross_timestep_edges",
                "message": (
                    f"{cross_count} edge(s) span different Time steps; temporal snapshots require intra-timestep edges only. "
                    "Edges are never silently dropped."
                ),
                "cross_timestep_edge_count": cross_count,
                "sample": cross_sample,
            },
        )

    # 6. Canonicalize row ordering to match Stage 2 preprocess_graph.py: sorted by ['Time step', 'txId']
    meta_df = pd.DataFrame(
        {
            "node_order": np.arange(len(tx_df), dtype=np.int64),
            "txId": tx_ids,
            "Time step": timesteps,
            "_orig_idx": np.arange(len(tx_df), dtype=np.int64),
        }
    )
    meta_sorted = meta_df.sort_values(["Time step", "txId"], kind="mergesort").reset_index(drop=True)
    order_idx = meta_sorted["_orig_idx"].to_numpy(dtype=np.int64)
    X_raw = X_raw_unsorted[order_idx]
    y_optional = y_unsorted[order_idx]
    meta_sorted = meta_sorted.drop(columns=["_orig_idx"])

    # Deduplicate exact parallel directed edges while reporting count
    raw_edge_count = len(src_arr)
    edge_pairs_all = pd.DataFrame({"src_tx": src_arr, "dst_tx": dst_arr, "timestep": edge_steps})
    edge_pairs = edge_pairs_all.drop_duplicates(subset=["src_tx", "dst_tx"], keep="first").reset_index(drop=True)
    dup_edges = raw_edge_count - len(edge_pairs)

    warnings: List[str] = []
    if nan_cells > 0:
        warnings.append(f"{nan_cells} missing numeric feature value(s) will be median-imputed using Stage 2 train statistics.")
    if dup_edges > 0:
        warnings.append(f"{dup_edges} duplicate directed edge(s) were deduplicated after integrity checks.")
    if has_optional_class:
        warnings.append("Optional class column detected: retained strictly for post-hoc display and NEVER used in inference.")

    class_counts = {
        "illicit": int((y_optional == 1).sum()),
        "licit": int((y_optional == 0).sum()),
        "unknown": int((y_optional == -1).sum()),
    }
    unique_steps = sorted(set(timesteps.tolist()))

    return ValidatedDataset(
        node_count=len(meta_sorted),
        edge_count_raw=raw_edge_count,
        unique_edges_after_dedup=len(edge_pairs),
        duplicate_edges_removed=dup_edges,
        timesteps=unique_steps,
        imputed_nan_cells=nan_cells,
        has_optional_class_column=has_optional_class,
        class_counts=class_counts,
        warnings=warnings,
        tx_df=meta_sorted,
        feature_columns=feature_cols,
        X_raw=X_raw,
        y_optional=y_optional,
        edge_pairs=edge_pairs,
    )
