"""End-to-end frozen inference orchestration for validated Stage 6 CSV uploads."""

from datetime import datetime, timezone
import os
import time
from typing import Any, Dict, List, Optional
import uuid

import numpy as np
import torch

import app.backend.config as config
from app.backend.database import db
from app.backend.ml.frozen_model import assign_risk_category, get_model_bundle
from app.backend.schemas import AnalyzeResponse
from app.backend.services.csv_validator import validate_csv_uploads
from app.backend.services.subgraph_service import analyze_uploaded_snapshot_clusters
from src.stage5.stage5b_individual_explanations import gt_label_str


def run_full_analysis(
    tx_bytes: bytes,
    edge_bytes: bytes,
    class_bytes: Optional[bytes] = None,
    dataset_name: str = "uploaded_dataset.csv",
) -> AnalyzeResponse:
    t0 = time.perf_counter()
    validated = validate_csv_uploads(tx_bytes, edge_bytes, class_bytes)
    bundle = get_model_bundle()
    threshold = float(bundle.threshold)

    # Exact Stage 2 Imputer+Scaler -> Stage 4H RankGauss Pipeline
    prep = bundle.preprocess_raw_features(validated.X_raw)
    x_stage2_all = prep["x_stage2"]
    x_rg_all = prep["x_rg"]

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    created_epoch = time.time()
    created_at = datetime.fromtimestamp(created_epoch, tz=timezone.utc).isoformat()
    run_cache_dir = os.path.join(config.RUNS_CACHE_DIR, run_id)
    os.makedirs(run_cache_dir, exist_ok=True)

    tx_meta = validated.tx_df
    step_series = tx_meta["Time step"].to_numpy(dtype=np.int64)
    tx_id_series = tx_meta["txId"].to_numpy(dtype=np.int64)
    order_series = tx_meta["node_order"].to_numpy(dtype=np.int64)

    tx_rows: List[Dict[str, Any]] = []
    all_cluster_rows: List[Dict[str, Any]] = []
    all_edge_rows: List[Dict[str, Any]] = []
    per_step_summary: List[Dict[str, Any]] = []
    bridge_samples_all: List[Dict[str, Any]] = []

    total_flagged = 0
    total_clusters = 0
    total_candidates = 0
    total_bridges = 0

    for step in validated.timesteps:
        mask = step_series == step
        step_tx_ids = tx_id_series[mask]
        step_orders = order_series[mask]
        step_y = validated.y_optional[mask]
        step_x_s2 = x_stage2_all[mask]
        step_x_rg = x_rg_all[mask]

        # Build directed intra-snapshot edge_index in local [0 .. N_t - 1] indices
        tx_to_local = {int(tid): idx for idx, tid in enumerate(step_tx_ids.tolist())}
        e_sub = validated.edge_pairs[validated.edge_pairs["timestep"] == step]
        if len(e_sub) > 0:
            src_local = np.array([tx_to_local[int(s)] for s in e_sub["src_tx"].tolist()], dtype=np.int64)
            dst_local = np.array([tx_to_local[int(d)] for d in e_sub["dst_tx"].tolist()], dtype=np.int64)
            edge_index_cpu = torch.tensor(np.vstack([src_local, dst_local]), dtype=torch.long)
        else:
            src_local = np.empty(0, dtype=np.int64)
            dst_local = np.empty(0, dtype=np.int64)
            edge_index_cpu = torch.empty((2, 0), dtype=torch.long)

        pred_out = bundle.predict_snapshot(step_x_rg, edge_index_cpu)
        probs = pred_out["probs"]
        logits = pred_out["logits"]
        preds = pred_out["preds"]

        sub_out = analyze_uploaded_snapshot_clusters(
            timestep=int(step),
            tx_ids=step_tx_ids,
            src_local=src_local,
            dst_local=dst_local,
            probs=probs,
            logits=logits,
            y_opt=step_y,
            threshold=threshold,
        )

        # Persist ephemeral snapshot tensor cache for fast on-demand IG & 1-hop cluster graph inspection
        torch.save(
            {
                "timestep": int(step),
                "x_stage2": torch.from_numpy(step_x_s2.astype(np.float32)),
                "x_rg": torch.from_numpy(step_x_rg.astype(np.float32)),
                "edge_index": edge_index_cpu,
                "tx_ids": torch.from_numpy(step_tx_ids),
                "probs": torch.from_numpy(probs.astype(np.float32)),
                "logits": torch.from_numpy(logits.astype(np.float32)),
                "y_opt": torch.from_numpy(step_y),
            },
            os.path.join(run_cache_dir, f"step_{int(step)}.pt"),
        )

        in_deg = sub_out["in_deg"]
        out_deg = sub_out["out_deg"]
        node_to_cid = sub_out["node_to_cid"]

        for u in range(len(step_tx_ids)):
            p = float(probs[u])
            pred_cls = int(preds[u])
            y_val = int(step_y[u])
            tx_rows.append(
                {
                    "node_order": int(step_orders[u]),
                    "tx_id": int(step_tx_ids[u]),
                    "timestep": int(step),
                    "snapshot_node_idx": int(u),
                    "pred_prob": round(p, 6),
                    "logit": round(float(logits[u]), 6),
                    "pred_class": pred_cls,
                    "pred_label": "Illicit" if pred_cls == 1 else "Licit",
                    "risk_category": assign_risk_category(p, threshold),
                    "in_deg": int(in_deg[u]),
                    "out_deg": int(out_deg[u]),
                    "cluster_id": node_to_cid.get(u, ""),
                    "optional_gt_code": y_val,
                    "optional_gt_label": gt_label_str(y_val),
                }
            )

        step_flagged = int(preds.sum())
        total_flagged += step_flagged
        total_clusters += sub_out["direct_clusters_count"]
        total_candidates += sub_out["potential_candidates_count"]
        total_bridges += sub_out["shared_bridges_count"]
        all_cluster_rows.extend(sub_out["cluster_rows"])
        all_edge_rows.extend(sub_out["edge_rows"])
        bridge_samples_all.extend(sub_out["bridge_samples"][:10])

        per_step_summary.append(
            {
                "timestep": int(step),
                "node_count": int(len(step_tx_ids)),
                "edge_count": int(len(e_sub)),
                "flagged_count": step_flagged,
                "flagged_rate_pct": round(100.0 * step_flagged / max(len(step_tx_ids), 1), 2),
                "mean_pred_prob": round(float(probs.mean()), 6) if len(probs) else 0.0,
                "max_pred_prob": round(float(probs.max()), 6) if len(probs) else 0.0,
                "direct_clusters_count": sub_out["direct_clusters_count"],
                "potential_candidates_count": sub_out["potential_candidates_count"],
                "shared_bridges_count": sub_out["shared_bridges_count"],
            }
        )

    inference_time_sec = round(time.perf_counter() - t0, 4)
    flagged_rate_pct = round(100.0 * total_flagged / max(validated.node_count, 1), 2)

    probs_all = np.array([r["pred_prob"] for r in tx_rows], dtype=np.float64)
    risk_band_counts = {
        "HighPredictedRisk": int(sum(1 for r in tx_rows if r["risk_category"].startswith("High Predicted Risk"))),
        "ElevatedRisk": int(sum(1 for r in tx_rows if r["risk_category"].startswith("Elevated Risk"))),
        "ModerateMonitor": int(sum(1 for r in tx_rows if r["risk_category"].startswith("Moderate Monitor"))),
        "LowPredictedRisk": int(sum(1 for r in tx_rows if r["risk_category"].startswith("Low Predicted Risk"))),
    }

    # Optional post-hoc metrics ONLY when uploaded CSV contains known labels (0/1)
    post_hoc_eval: Optional[Dict[str, Any]] = None
    if validated.has_optional_class_column:
        ys_known = [r for r in tx_rows if r["optional_gt_code"] in (0, 1)]
        if ys_known:
            tp = sum(1 for r in ys_known if r["optional_gt_code"] == 1 and r["pred_class"] == 1)
            fp = sum(1 for r in ys_known if r["optional_gt_code"] == 0 and r["pred_class"] == 1)
            fn = sum(1 for r in ys_known if r["optional_gt_code"] == 1 and r["pred_class"] == 0)
            tn = sum(1 for r in ys_known if r["optional_gt_code"] == 0 and r["pred_class"] == 0)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            post_hoc_eval = {
                "labeled_count": len(ys_known),
                "unknown_count": validated.node_count - len(ys_known),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "note": "Post-hoc display only; optional class labels NEVER entered frozen inference.",
            }

    disclaimer = (
        "Risk scores and suspicious clusters are probabilistic model outputs at frozen threshold "
        f"{threshold:.2f} and require human compliance investigation; they are not proof of criminal fraud."
    )

    summary_payload = {
        "model_id": bundle.exp_id,
        "preprocessing_order": ["Stage2 SimpleImputer(median) + StandardScaler", "Stage4H RankGauss Pipeline"],
        "timesteps": validated.timesteps,
        "unique_edges_after_dedup": validated.unique_edges_after_dedup,
        "duplicate_edges_removed": validated.duplicate_edges_removed,
        "imputed_nan_cells": validated.imputed_nan_cells,
        "flagged_rate_pct": flagged_rate_pct,
        "mean_pred_prob": round(float(probs_all.mean()), 6) if len(probs_all) else 0.0,
        "max_pred_prob": round(float(probs_all.max()), 6) if len(probs_all) else 0.0,
        "risk_band_counts": risk_band_counts,
        "class_counts": validated.class_counts,
        "post_hoc_eval": post_hoc_eval,
        "per_timestep": per_step_summary,
        "bridge_samples": bridge_samples_all[:20],
        "warnings": validated.warnings,
        "disclaimer": disclaimer,
    }

    run_meta = {
        "run_id": run_id,
        "dataset_name": dataset_name or "uploaded_dataset.csv",
        "created_at": created_at,
        "created_epoch": created_epoch,
        "threshold_used": threshold,
        "inference_time_sec": inference_time_sec,
        "node_count": validated.node_count,
        "edge_count": validated.unique_edges_after_dedup,
        "flagged_count": total_flagged,
        "direct_clusters_count": total_clusters,
        "potential_candidates_count": total_candidates,
        "shared_bridges_count": total_bridges,
        "has_optional_class_column": validated.has_optional_class_column,
        "summary": summary_payload,
    }

    db.save_analysis_run(run_meta, tx_rows, all_cluster_rows, all_edge_rows)

    return AnalyzeResponse(
        run_id=run_id,
        dataset_name=run_meta["dataset_name"],
        status="completed",
        created_at=created_at,
        inference_time_sec=inference_time_sec,
        threshold_used=threshold,
        node_count=validated.node_count,
        edge_count=validated.unique_edges_after_dedup,
        timesteps=validated.timesteps,
        flagged_count=total_flagged,
        flagged_rate_pct=flagged_rate_pct,
        direct_clusters_count=total_clusters,
        potential_candidates_count=total_candidates,
        shared_bridges_count=total_bridges,
        warnings=validated.warnings,
        disclaimer=disclaimer,
    )
