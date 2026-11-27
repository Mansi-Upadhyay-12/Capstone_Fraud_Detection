"""On-demand Stage 5B 3-zone Composite Midpoint Integrated Gradients (M=200) explanation service."""

from datetime import datetime, timezone
import os
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import HTTPException

import app.backend.config as config
from app.backend.database import db
from app.backend.ml.frozen_model import FrozenFraudModelBundle, get_model_bundle
from src.stage5.stage5b_individual_explanations import gt_label_str


def _feature_group_from_index(idx: int) -> str:
    if idx < 93:
        return "Local93"
    if idx < 165:
        return "Aggregate72"
    return "Domain17"


def compute_composite_midpoint_ig(
    bundle: FrozenFraudModelBundle,
    x_rg_cpu: torch.Tensor,
    edge_index_cpu: torch.Tensor,
    node_idx: int,
) -> Dict[str, Any]:
    """
    Computes exact Stage 5B 3-zone Composite Midpoint Integrated Gradients (M=200):
      - Zone 1: 60 points over [0.000, 0.005]
      - Zone 2: 60 points over [0.005, 0.050]
      - Zone 3: 80 points over [0.050, 1.000]
    Baseline x_base = 0 in RankGauss feature space. Target scalar = raw logit[node_idx].
    """
    dev = bundle.device
    x_rg = x_rg_cpu.to(dev)
    edge_index = edge_index_cpu.to(dev)
    graph_bundle = bundle.build_adjacency_bundle(edge_index, x_rg.size(0))

    x_base = torch.zeros_like(x_rg)
    diff = x_rg - x_base

    a1 = torch.linspace(0.0, 0.005, 61, device=dev)
    a2 = torch.linspace(0.005, 0.05, 61, device=dev)
    a3 = torch.linspace(0.05, 1.0, 81, device=dev)
    m1 = 0.5 * (a1[:-1] + a1[1:])
    m2 = 0.5 * (a2[:-1] + a2[1:])
    m3 = 0.5 * (a3[:-1] + a3[1:])
    alphas = torch.cat([m1, m2, m3], dim=0)
    weights = torch.cat(
        [
            torch.full((60,), 0.005 / 60.0, dtype=torch.float32, device=dev),
            torch.full((60,), 0.045 / 60.0, dtype=torch.float32, device=dev),
            torch.full((80,), 0.95 / 80.0, dtype=torch.float32, device=dev),
        ]
    )

    accum = torch.zeros_like(x_rg)
    for a, w in zip(alphas, weights):
        xi = (x_base + a * diff).detach().requires_grad_(True)
        logits, _, _ = bundle.model.forward_step(xi, graph_bundle, None)
        logits[node_idx].backward()
        accum += w * xi.grad.detach()

    ig_mat = (diff * accum).detach().cpu().numpy()
    with torch.no_grad():
        lx = float(bundle.model.forward_step(x_rg, graph_bundle, None)[0][node_idx].item())
        l0 = float(bundle.model.forward_step(x_base, graph_bundle, None)[0][node_idx].item())

    delta = lx - l0
    ig_sum = float(ig_mat.sum())
    abs_err = abs(ig_sum - delta)
    rel_err = abs_err / max(abs(delta), 1e-12) * 100.0
    return {
        "ig_matrix": ig_mat,
        "graph_signed": ig_mat.sum(axis=0),
        "self_signed": ig_mat[node_idx],
        "logit_x": lx,
        "logit_0": l0,
        "logit_delta": delta,
        "ig_sum": ig_sum,
        "abs_err": abs_err,
        "rel_err_pct": rel_err,
        "all_finite": bool(np.isfinite(ig_mat).all()),
    }


def explain_transaction(run_id: str, tx_id: int) -> Dict[str, Any]:
    run_info = db.get_run(run_id)
    if run_info is None:
        raise HTTPException(status_code=404, detail={"error": "run_not_found", "message": f"Run '{run_id}' not found."})

    tx_rec = db.get_transaction(run_id, tx_id)
    if tx_rec is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "tx_not_found", "message": f"Transaction {tx_id} not found in run '{run_id}'."},
        )

    cached = db.get_cached_explanation(run_id, tx_id)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    t = int(tx_rec["timestep"])
    snap_path = os.path.join(config.RUNS_CACHE_DIR, run_id, f"step_{t}.pt")
    if not os.path.exists(snap_path):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "snapshot_cache_expired",
                "message": (
                    f"Tensor cache for run '{run_id}' timestep {t} has expired or was evicted "
                    f"(retention: {config.MAX_CACHED_RUNS} runs / {config.RETENTION_DAYS} days). "
                    "Re-upload the CSV to compute new IG attributions."
                ),
            },
        )

    snap = torch.load(snap_path, map_location="cpu", weights_only=False)
    node_idx = int(tx_rec["snapshot_node_idx"])
    x_rg_cpu = snap["x_rg"]
    edge_index_cpu = snap["edge_index"]
    tx_ids = snap["tx_ids"].numpy()
    probs = snap["probs"].numpy()
    y_opt = snap["y_opt"].numpy()

    bundle = get_model_bundle()
    threshold = float(bundle.threshold)
    feat_names = bundle.feature_columns

    ig_res = compute_composite_midpoint_ig(bundle, x_rg_cpu, edge_index_cpu, node_idx)
    ig_mat = ig_res["ig_matrix"]
    graph_signed = ig_res["graph_signed"]
    self_signed = ig_res["self_signed"]
    node_ig_totals = ig_mat.sum(axis=1)

    self_total = float(self_signed.sum())
    graph_total = float(ig_res["ig_sum"])
    neighbor_total = float(graph_total - self_total)
    total_abs = float(np.abs(node_ig_totals).sum())
    self_abs = float(np.abs(node_ig_totals[node_idx]))
    neighbor_abs = float(total_abs - self_abs)

    # Top positive (fraud-pushing) and negative (licit-pushing) RankGauss features
    pos_order = np.argsort(-graph_signed)[:10]
    neg_order = np.argsort(graph_signed)[:10]

    def _feat_rows(indices: np.ndarray) -> List[Dict[str, Any]]:
        out = []
        for idx in indices.tolist():
            rg_val = float(x_rg_cpu[node_idx, idx].item())
            out.append(
                {
                    "feature_index": idx,
                    "feature_name": feat_names[idx],
                    "feature_group": _feature_group_from_index(idx),
                    "target_rankgauss_value": round(rg_val, 6),
                    "graph_ig_attribution": round(float(graph_signed[idx]), 6),
                    "self_ig_attribution": round(float(self_signed[idx]), 6),
                    "neighbor_ig_attribution": round(float(graph_signed[idx] - self_signed[idx]), 6),
                }
            )
        return out

    # Feature group breakdown
    group_breakdown = []
    for grp_name, i0, i1 in (("Local93", 0, 93), ("Aggregate72", 93, 165), ("Domain17", 165, 182)):
        sl = graph_signed[i0:i1]
        group_breakdown.append(
            {
                "group": grp_name,
                "feature_count": int(i1 - i0),
                "signed_sum": round(float(sl.sum()), 6),
                "abs_sum": round(float(np.abs(sl).sum()), 6),
                "self_signed_sum": round(float(self_signed[i0:i1].sum()), 6),
            }
        )

    # Directed 1-hop & 2-hop neighborhood inspection (strictly intra-snapshot)
    src_np = edge_index_cpu[0].numpy()
    dst_np = edge_index_cpu[1].numpy()
    in_deg = np.bincount(dst_np, minlength=len(tx_ids))
    out_deg = np.bincount(src_np, minlength=len(tx_ids))

    in_nbrs = [int(src_np[e]) for e in np.where(dst_np == node_idx)[0]]
    out_nbrs = [int(dst_np[e]) for e in np.where(src_np == node_idx)[0]]

    nbr_rows: List[Dict[str, Any]] = []
    seen = set()
    for direction, nbr_list in (("incoming", in_nbrs), ("outgoing", out_nbrs)):
        for v in nbr_list:
            key = (direction, v)
            if key in seen:
                continue
            seen.add(key)
            vp = float(probs[v])
            vy = int(y_opt[v])
            nbr_rows.append(
                {
                    "direction": direction,
                    "edge_representation": (
                        f"{int(tx_ids[v])} -> {int(tx_ids[node_idx])}"
                        if direction == "incoming"
                        else f"{int(tx_ids[node_idx])} -> {int(tx_ids[v])}"
                    ),
                    "neighbor_tx_id": int(tx_ids[v]),
                    "neighbor_pred_prob": round(vp, 6),
                    "neighbor_pred_class": "Illicit" if vp >= threshold else "Licit",
                    "neighbor_gt_label": gt_label_str(vy),
                    "neighbor_in_deg": int(in_deg[v]),
                    "neighbor_out_deg": int(out_deg[v]),
                    "neighbor_signed_ig_logit": round(float(node_ig_totals[v]), 6),
                    "neighbor_abs_ig_logit": round(float(np.abs(ig_mat[v]).sum()), 6),
                }
            )
    nbr_rows.sort(key=lambda r: (-r["neighbor_abs_ig_logit"], r["direction"], r["neighbor_tx_id"]))

    unique_1hop = sorted((set(in_nbrs) | set(out_nbrs)) - {node_idx})
    two_hop_set = set()
    if unique_1hop:
        u1_arr = np.array(unique_1hop)
        m2 = np.isin(src_np, u1_arr) | np.isin(dst_np, u1_arr)
        cand2 = set(src_np[m2].tolist()) | set(dst_np[m2].tolist())
        two_hop_set = cand2 - set(unique_1hop) - {node_idx}

    if two_hop_set:
        idxs2 = list(two_hop_set)
        ps2 = probs[idxs2]
        ys2 = y_opt[idxs2]
        preds2 = ps2 >= threshold
        two_hop_summary = {
            "count": len(idxs2),
            "pred_illicit_count": int(preds2.sum()),
            "pred_licit_count": int(len(idxs2) - preds2.sum()),
            "pred_illicit_rate": round(float(preds2.mean()), 4),
            "mean_pred_prob": round(float(ps2.mean()), 4),
            "gt_illicit_count": int((ys2 == 1).sum()),
            "gt_licit_count": int((ys2 == 0).sum()),
            "gt_unknown_count": int((ys2 == -1).sum()),
        }
    else:
        two_hop_summary = {
            "count": 0,
            "pred_illicit_count": 0,
            "pred_licit_count": 0,
            "pred_illicit_rate": 0.0,
            "mean_pred_prob": 0.0,
            "gt_illicit_count": 0,
            "gt_licit_count": 0,
            "gt_unknown_count": 0,
        }

    created_at = datetime.now(timezone.utc).isoformat()
    method_name = "Composite_Midpoint_IG_3Zone_M200"
    payload = {
        "run_id": run_id,
        "tx_id": int(tx_id),
        "timestep": t,
        "created_at": created_at,
        "cache_hit": False,
        "method": method_name,
        "baseline": "RankGauss zero vector (training median baseline)",
        "model_id": bundle.exp_id,
        "threshold_used": threshold,
        "prediction": {
            "pred_prob": float(tx_rec["pred_prob"]),
            "logit": float(tx_rec["logit"]),
            "pred_label": tx_rec["pred_label"],
            "risk_category": tx_rec["risk_category"],
            "cluster_id": tx_rec["cluster_id"],
            "optional_gt_label": tx_rec["optional_gt_label"],
        },
        "completeness": {
            "logit_x": round(ig_res["logit_x"], 6),
            "logit_baseline": round(ig_res["logit_0"], 6),
            "logit_delta": round(ig_res["logit_delta"], 6),
            "ig_sum": round(ig_res["ig_sum"], 6),
            "abs_error": float(ig_res["abs_err"]),
            "rel_error_pct": float(ig_res["rel_err_pct"]),
            "all_finite": ig_res["all_finite"],
        },
        "self_vs_neighbor": {
            "self_signed_logit": round(self_total, 6),
            "neighbor_signed_logit": round(neighbor_total, 6),
            "self_abs_share": round(self_abs / max(total_abs, 1e-12), 4),
            "neighbor_abs_share": round(neighbor_abs / max(total_abs, 1e-12), 4),
            "one_hop_unique_count": len(unique_1hop),
            "incoming_count": len(in_nbrs),
            "outgoing_count": len(out_nbrs),
        },
        "feature_group_breakdown": group_breakdown,
        "top_positive_features": _feat_rows(pos_order),
        "top_negative_features": _feat_rows(neg_order),
        "one_hop_neighbors": nbr_rows,
        "two_hop_summary": two_hop_summary,
        "disclaimer": (
            "Integrated Gradients explains model logit sensitivity in RankGauss-normalized space relative to the "
            "median baseline. Flagged predictions and structural clusters are investigative risk signals, not legal proof."
        ),
    }

    db.save_explanation(
        run_id=run_id,
        tx_id=int(tx_id),
        created_at=created_at,
        method=method_name,
        abs_err=float(ig_res["abs_err"]),
        rel_err=float(ig_res["rel_err_pct"]),
        payload=payload,
    )
    return payload
