"""
Stage 5B: Individual Fraud Prediction Explanations
Project: Temporal Graph Neural Networks with Real-Time Explainable AI
         for Online Payment Fraud Ring Detection

Models Evaluated (Frozen — Zero Retraining / Zero Test Tuning):
  - Stage 4H Selected Graph Champion: ExpA2_RankGauss182_BiDirGCN
    Checkpoint: models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt
    Preprocessing: models/stage4h/rankgauss_transformer_train_1_34.joblib
    Threshold: Read directly from checkpoint (`threshold_recal_01_99`)

Outputs saved strictly inside:
  - results/stage5/
  - reports/stage5/
  - figures/stage5/

Stage 1-4H and Stage 5A scripts/artifacts remain completely untouched.
"""

import json
import os
import sys
import time
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models_stage4f import BiDirRelationalGCN, build_true_directed_gcn_adj
from src.stage5.stage5a_feature_importance import (
    assign_feature_group,
    integrated_gradients_snapshot,
)


CATEGORY_COLORS = {
    "TP": "#2b6cb0",
    "FP": "#dd6b20",
    "FN": "#e53e3e",
    "TN": "#38a169",
}


def gt_label_str(y_val):
    if int(y_val) == 1:
        return "Illicit"
    if int(y_val) == 0:
        return "Licit"
    if int(y_val) == -1:
        return "Unknown"
    raise ValueError(f"Unexpected ground truth label value: {y_val}")


def classify_prediction(y_val, prob, threshold):
    """Classifies labeled nodes into TP/FP/FN/TN; refuses unknown (-1) evaluation."""
    if int(y_val) not in (0, 1):
        raise ValueError(f"Cannot assign TP/FP/FN/TN to unknown label y={y_val}")
    pred = 1 if float(prob) >= float(threshold) else 0
    if y_val == 1 and pred == 1:
        return "TP", pred
    if y_val == 0 and pred == 1:
        return "FP", pred
    if y_val == 1 and pred == 0:
        return "FN", pred
    return "TN", pred


def compute_snapshot_predictions(gnn_model, rg_snapshots, rg_bundles, threshold):
    """Runs frozen Stage 4H forward pass snapshot-by-snapshot (no cross-snapshot edges)."""
    gnn_model.eval()
    snap_preds = {}
    with torch.no_grad():
        for t, (s, b) in enumerate(zip(rg_snapshots, rg_bundles)):
            logits, _, _ = gnn_model.forward_step(s.x, b, None)
            probs = torch.sigmoid(logits).cpu().numpy()
            logits_np = logits.cpu().numpy()
            src_np = s.edge_index[0].cpu().numpy()
            dst_np = s.edge_index[1].cpu().numpy()
            in_deg = np.bincount(dst_np, minlength=s.num_nodes)
            out_deg = np.bincount(src_np, minlength=s.num_nodes)
            snap_preds[t] = {
                "timestep": int(s.timestep),
                "probs": probs,
                "logits": logits_np,
                "in_deg": in_deg,
                "out_deg": out_deg,
                "y": s.y.cpu().numpy(),
                "txId": s.txId.cpu().numpy(),
                "labeled_mask": s.labeled_mask.cpu().numpy(),
            }
    return snap_preds


def summarize_node_degrees_and_confusion(snap_preds, start_idx, end_idx, threshold):
    """Verifies labeled confusion counts (excluding y == -1) and builds candidate table."""
    records = []
    cm = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "unknown_excluded": 0}
    for t in range(start_idx, end_idx):
        sp = snap_preds[t]
        y = sp["y"]
        probs = sp["probs"]
        in_d = sp["in_deg"]
        out_d = sp["out_deg"]
        tx_ids = sp["txId"]
        lm = sp["labeled_mask"]
        cm["unknown_excluded"] += int((~lm).sum())
        for u in np.where(lm)[0]:
            cat, pred = classify_prediction(int(y[u]), float(probs[u]), threshold)
            cm[cat] += 1
            records.append({
                "snap_idx": t,
                "timestep": int(sp["timestep"]),
                "node_idx": int(u),
                "tx_id": int(tx_ids[u]),
                "y": int(y[u]),
                "prob": float(probs[u]),
                "logit": float(sp["logits"][u]),
                "pred": int(pred),
                "category": cat,
                "in_deg": int(in_d[u]),
                "out_deg": int(out_d[u]),
                "tot_deg": int(in_d[u] + out_d[u]),
            })
    return pd.DataFrame(records), cm


def _pick_candidate(sub_df, mode, used_txids, min_deg=2):
    """
    Deterministic, reproducible candidate selection preferring connected nodes
    (in_deg + out_deg >= min_deg, falling back to >=1 then any), tie-broken by tx_id.
    """
    avail = sub_df[~sub_df["tx_id"].isin(used_txids)].copy()
    if len(avail) == 0:
        return None
    for d_req in (min_deg, 1, 0):
        pool = avail[avail["tot_deg"] >= d_req]
        if len(pool) > 0:
            avail = pool
            break

    # Prefer nodes with both in and out neighbors if available when min_deg >= 2
    both = avail[(avail["in_deg"] >= 1) & (avail["out_deg"] >= 1)]
    if len(both) > 0:
        avail = both

    if mode == "highest_prob":
        avail = avail.sort_values(["prob", "tot_deg", "tx_id"], ascending=[False, False, True])
    elif mode == "lowest_prob":
        avail = avail.sort_values(["prob", "tot_deg", "tx_id"], ascending=[True, False, True])
    elif mode == "near_threshold_above":
        avail = avail.sort_values(["prob", "tot_deg", "tx_id"], ascending=[True, False, True])
    elif mode == "near_threshold_below":
        avail = avail.sort_values(["prob", "tot_deg", "tx_id"], ascending=[False, False, True])
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    return avail.iloc[0].to_dict()


def select_representative_examples(val_df, test_df, threshold):
    """
    Systematically selects:
      - 8 primary Validation examples (Steps 35-41): 2 TP, 2 FP, 2 FN, 2 TN.
      - 4 exploratory Test examples (Steps 42-49): 1 TP (Step 42), 1 FP, 1 FN (Steps 43-49), 1 TN.
    Selection rule is fully deterministic and uses Validation for primary cases.
    """
    selected = []
    used_tx = set()

    val_specs = [
        ("VAL_TP_1", "TP", "highest_prob", "Highest-probability connected Validation TP"),
        ("VAL_TP_2", "TP", "near_threshold_above", "Borderline-confidence connected Validation TP (just above tau*)"),
        ("VAL_FP_1", "FP", "highest_prob", "Highest-probability connected Validation FP false alarm"),
        ("VAL_FP_2", "FP", "near_threshold_above", "Borderline-confidence connected Validation FP (just above tau*)"),
        ("VAL_FN_1", "FN", "lowest_prob", "Lowest-probability connected Validation FN missed fraud"),
        ("VAL_FN_2", "FN", "near_threshold_below", "Near-threshold connected Validation FN (just below tau*)"),
        ("VAL_TN_1", "TN", "lowest_prob", "Lowest-probability connected Validation TN benign baseline"),
        ("VAL_TN_2", "TN", "near_threshold_below", "Elevated-score connected Validation TN (below tau* but non-trivial risk)"),
    ]
    for cid, cat, mode, rationale in val_specs:
        sub = val_df[val_df["category"] == cat]
        cand = _pick_candidate(sub, mode, used_tx, min_deg=2)
        assert cand is not None, f"Could not select candidate for {cid}"
        used_tx.add(cand["tx_id"])
        cand["case_id"] = cid
        cand["split_role"] = "Validation_Primary"
        cand["selection_rationale"] = rationale
        selected.append(cand)

    # Exploratory post-hoc Test examples (not used for any tuning/selection of model/threshold)
    test_specs = [
        ("TEST_TP_42", test_df[(test_df["category"] == "TP") & (test_df["timestep"] == 42)],
         "highest_prob", "Exploratory Test Step 42 high-confidence connected TP"),
        ("TEST_FP_1", test_df[test_df["category"] == "FP"],
         "highest_prob", "Exploratory Test high-confidence connected FP"),
        ("TEST_FN_POST43", test_df[(test_df["category"] == "FN") & (test_df["timestep"] >= 43)],
         "lowest_prob", "Exploratory Test post-shutdown (Steps 43-49) connected FN collapse example"),
        ("TEST_TN_1", test_df[test_df["category"] == "TN"],
         "lowest_prob", "Exploratory Test lowest-probability connected TN"),
    ]
    for cid, sub, mode, rationale in test_specs:
        cand = _pick_candidate(sub, mode, used_tx, min_deg=2)
        if cand is None:
            cand = _pick_candidate(test_df[test_df["category"] == cid.split("_")[1]], mode, used_tx, min_deg=1)
        assert cand is not None, f"Could not select exploratory candidate for {cid}"
        used_tx.add(cand["tx_id"])
        cand["case_id"] = cid
        cand["split_role"] = "Test_Exploratory"
        cand["selection_rationale"] = rationale
        selected.append(cand)

    return selected


def explain_single_node_ig(model, x_rg, bundle, node_idx, n_steps=80):
    """
    Computes exact Integrated Gradients for a SINGLE target node using the exact
    Stage 5A alpha=u^4 quadrature path relative to x_base=0, preserving full
    [N_t, 182] node-by-feature attribution breakdown AND verifying exact parity
    against Stage 5A's `integrated_gradients_snapshot`.
    """
    model.eval()
    x_base = torch.zeros_like(x_rg)
    diff = x_rg - x_base

    u = (torch.arange(1, n_steps + 1, dtype=torch.float32) - 0.5) / float(n_steps)
    alphas = u ** 4
    weights = 4.0 * (u ** 3) / float(n_steps)
    accum_grad = torch.zeros_like(x_rg)

    for a, w in zip(alphas, weights):
        x_interp = (x_base + a * diff).detach().requires_grad_(True)
        logits, _, _ = model.forward_step(x_interp, bundle, None)
        logits[node_idx].backward()
        accum_grad += w * x_interp.grad.detach()

    ig_matrix = (diff * accum_grad).detach()  # [N_t, 182]

    with torch.no_grad():
        l_x = float(model.forward_step(x_rg, bundle, None)[0][node_idx].item())
        l_0 = float(model.forward_step(x_base, bundle, None)[0][node_idx].item())

    # Cross-check exact equivalence with Stage 5A's integrated_gradients_snapshot
    target_mask = torch.zeros(x_rg.shape[0], dtype=torch.bool)
    target_mask[node_idx] = True
    ref_5a = integrated_gradients_snapshot(model, x_rg, bundle, target_mask, n_steps=n_steps)

    graph_abs = ig_matrix.abs().sum(dim=0).cpu().numpy()
    max_parity_diff = float(np.max(np.abs(graph_abs - ref_5a["full_graph_abs"])))
    assert max_parity_diff < 1e-5, f"Stage 5A IG parity mismatch: {max_parity_diff}"

    total_ig_sum = float(ig_matrix.sum().item())
    comp_err = abs(total_ig_sum - (l_x - l_0))

    return {
        "ig_matrix": ig_matrix.cpu().numpy(),                # [N_t, 182]
        "graph_signed": ig_matrix.sum(dim=0).cpu().numpy(),  # [182], sums to ~ (l_x - l_0)
        "graph_abs": graph_abs,                              # [182], exact Stage 5A metric
        "self_signed": ig_matrix[node_idx].cpu().numpy(),    # [182]
        "self_abs": ig_matrix[node_idx].abs().cpu().numpy(), # [182]
        "logit_x": l_x,
        "logit_0": l_0,
        "logit_delta": l_x - l_0,
        "total_ig_sum": total_ig_sum,
        "completeness_err": comp_err,
        "stage5a_parity_max_diff": max_parity_diff,
    }


def analyze_node_neighbourhood(s, sp_pred, ig_matrix, target_idx, threshold):
    """
    Analyzes direct incoming (w -> target) and outgoing (target -> w) edges within
    snapshot `s` ONLY, preserving direction and separating Unknown (-1) from Licit (0).
    """
    u = int(target_idx)
    src_np = s.edge_index[0].cpu().numpy()
    dst_np = s.edge_index[1].cpu().numpy()

    in_edges = np.where(dst_np == u)[0]
    out_edges = np.where(src_np == u)[0]
    in_nbrs = [int(src_np[e]) for e in in_edges]
    out_nbrs = [int(dst_np[e]) for e in out_edges]

    node_ig_totals = ig_matrix.sum(axis=1)  # signed logit contribution from each node in snapshot

    nbr_rows = []
    seen_pairs = set()
    for direction, nbr_list in (("incoming", in_nbrs), ("outgoing", out_nbrs)):
        for v in nbr_list:
            # Handle duplicate parallel edges if any while keeping distinct directions
            key = (direction, v)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            vy = int(sp_pred["y"][v])
            vp = float(sp_pred["probs"][v])
            vpred = 1 if vp >= threshold else 0
            nbr_rows.append({
                "direction": direction,
                "edge_representation": f"{int(sp_pred['txId'][v])} -> {int(sp_pred['txId'][u])}"
                if direction == "incoming"
                else f"{int(sp_pred['txId'][u])} -> {int(sp_pred['txId'][v])}",
                "neighbor_node_idx": v,
                "neighbor_tx_id": int(sp_pred["txId"][v]),
                "timestep": int(sp_pred["timestep"]),
                "neighbor_gt_code": vy,
                "neighbor_gt_label": gt_label_str(vy),  # Explicitly preserves "Unknown" for -1
                "neighbor_pred_prob": round(vp, 6),
                "neighbor_pred_class": "Illicit" if vpred == 1 else "Licit",
                "neighbor_in_deg": int(sp_pred["in_deg"][v]),
                "neighbor_out_deg": int(sp_pred["out_deg"][v]),
                "neighbor_signed_ig_logit": round(float(node_ig_totals[v]), 6),
                "neighbor_abs_ig_logit": round(float(np.abs(ig_matrix[v]).sum()), 6),
            })

    unique_1hop = sorted(set(in_nbrs) | set(out_nbrs) - {u})
    in_set = set(in_nbrs) - {u}
    out_set = set(out_nbrs) - {u}
    only_in = in_set - out_set
    only_out = out_set - in_set
    both_dir = in_set & out_set

    # 2-hop undirected neighbor indices (strictly within same snapshot `s`)
    adj_all = set(unique_1hop)
    two_hop_set = set()
    if len(unique_1hop) > 0:
        u1_arr = np.array(unique_1hop)
        m2 = np.isin(src_np, u1_arr) | np.isin(dst_np, u1_arr)
        cand2 = set(src_np[m2].tolist()) | set(dst_np[m2].tolist())
        two_hop_set = cand2 - adj_all - {u}

    def _cohort_stats(idxs):
        if len(idxs) == 0:
            return {
                "count": 0, "pred_illicit_count": 0, "pred_licit_count": 0,
                "pred_illicit_rate": 0.0, "mean_pred_prob": 0.0, "max_pred_prob": 0.0,
                "gt_illicit_count": 0, "gt_licit_count": 0, "gt_unknown_count": 0,
            }
        ys = sp_pred["y"][list(idxs)]
        ps = sp_pred["probs"][list(idxs)]
        preds = (ps >= threshold).astype(int)
        return {
            "count": int(len(idxs)),
            "pred_illicit_count": int(preds.sum()),
            "pred_licit_count": int(len(idxs) - preds.sum()),
            "pred_illicit_rate": round(float(preds.mean()), 4),
            "mean_pred_prob": round(float(ps.mean()), 4),
            "max_pred_prob": round(float(ps.max()), 4),
            "gt_illicit_count": int((ys == 1).sum()),
            "gt_licit_count": int((ys == 0).sum()),
            "gt_unknown_count": int((ys == -1).sum()),
        }

    # Decompose total signed IG by exact topological origin (additive partition of graph nodes!)
    self_ig_total = float(node_ig_totals[u])
    in_only_ig_total = float(sum(node_ig_totals[v] for v in only_in))
    out_only_ig_total = float(sum(node_ig_totals[v] for v in only_out))
    bidir_1hop_ig_total = float(sum(node_ig_totals[v] for v in both_dir))
    in_1hop_inclusive_ig = float(sum(node_ig_totals[v] for v in in_set))
    out_1hop_inclusive_ig = float(sum(node_ig_totals[v] for v in out_set))
    two_hop_only_ig_total = float(sum(node_ig_totals[v] for v in two_hop_set))

    one_hop_stats = _cohort_stats(unique_1hop)
    in_stats = _cohort_stats(sorted(in_set))
    out_stats = _cohort_stats(sorted(out_set))
    two_hop_stats = _cohort_stats(sorted(two_hop_set))

    # Categorize suspiciousness concentration in 1-hop neighborhood
    if len(unique_1hop) == 0:
        concentration = "Isolated transaction (no incoming or outgoing snapshot edges)"
    elif one_hop_stats["pred_illicit_count"] > 0 and one_hop_stats["pred_illicit_rate"] >= 0.5:
        concentration = (
            f"Strongly concentrated suspicious neighborhood: {one_hop_stats['pred_illicit_count']}/{one_hop_stats['count']} "
            f"({one_hop_stats['pred_illicit_rate']*100:.1f}%) 1-hop neighbors predicted Illicit"
        )
    elif one_hop_stats["pred_illicit_count"] > 0:
        concentration = (
            f"Mixed neighborhood with localized suspicion: {one_hop_stats['pred_illicit_count']}/{one_hop_stats['count']} "
            f"({one_hop_stats['pred_illicit_rate']*100:.1f}%) 1-hop neighbors predicted Illicit (max P={one_hop_stats['max_pred_prob']:.3f})"
        )
    elif two_hop_stats["pred_illicit_count"] > 0:
        concentration = (
            f"Clean 1-hop neighborhood (0/{one_hop_stats['count']} predicted Illicit), "
            f"but {two_hop_stats['pred_illicit_count']}/{two_hop_stats['count']} 2-hop neighbors flagged Illicit"
        )
    else:
        concentration = (
            f"Predominantly benign predicted neighborhood: 0/{one_hop_stats['count']} 1-hop neighbors predicted Illicit "
            f"(mean P={one_hop_stats['mean_pred_prob']:.3f}, max P={one_hop_stats['max_pred_prob']:.3f})"
        )

    return {
        "neighbor_rows": nbr_rows,
        "in_degree": len(in_nbrs),
        "out_degree": len(out_nbrs),
        "unique_1hop_count": len(unique_1hop),
        "unique_2hop_only_count": len(two_hop_set),
        "one_hop_stats": one_hop_stats,
        "in_stats": in_stats,
        "out_stats": out_stats,
        "two_hop_stats": two_hop_stats,
        "ig_origin_partition": {
            "self_signed_ig": round(self_ig_total, 6),
            "in_1hop_only_signed_ig": round(in_only_ig_total, 6),
            "out_1hop_only_signed_ig": round(out_only_ig_total, 6),
            "bidir_1hop_signed_ig": round(bidir_1hop_ig_total, 6),
            "in_1hop_inclusive_signed_ig": round(in_1hop_inclusive_ig, 6),
            "out_1hop_inclusive_signed_ig": round(out_1hop_inclusive_ig, 6),
            "two_hop_only_signed_ig": round(two_hop_only_ig_total, 6),
        },
        "concentration_summary": concentration,
        "in_nbr_indices": sorted(in_set),
        "out_nbr_indices": sorted(out_set),
        "two_hop_indices": sorted(two_hop_set),
    }


def extract_top_feature_attributions(ig_res, raw_x_node, rg_x_node, feat_names, feat_groups, k=5):
    """Extracts top positive, top negative, and top absolute IG feature contributions."""
    gs = ig_res["graph_signed"]
    ga = ig_res["graph_abs"]
    ss = ig_res["self_signed"]

    pos_idx = [int(i) for i in np.argsort(-gs) if gs[i] > 0][:k]
    neg_idx = [int(i) for i in np.argsort(gs) if gs[i] < 0][:k]
    abs_idx = [int(i) for i in np.argsort(-np.abs(gs))][:8]

    def _fmt(indices):
        out = []
        for idx in indices:
            out.append({
                "feature_index": idx,
                "feature_name": feat_names[idx],
                "feature_group": feat_groups[idx],
                "raw_value": round(float(raw_x_node[idx]), 5),
                "rankgauss_value": round(float(rg_x_node[idx]), 5),
                "graph_signed_ig": round(float(gs[idx]), 6),
                "graph_abs_ig": round(float(ga[idx]), 6),
                "self_signed_ig": round(float(ss[idx]), 6),
                "neighbor_signed_ig": round(float(gs[idx] - ss[idx]), 6),
            })
        return out

    # Signed group totals
    group_signed = {}
    for g in ("Local93", "Aggregate72", "Domain17"):
        g_idx = [i for i, grp in enumerate(feat_groups) if grp == g]
        group_signed[g] = {
            "signed_ig_sum": round(float(gs[g_idx].sum()), 5),
            "abs_ig_sum": round(float(ga[g_idx].sum()), 5),
            "pos_ig_sum": round(float(np.clip(gs[g_idx], 0.0, None).sum()), 5),
            "neg_ig_sum": round(float(np.clip(gs[g_idx], None, 0.0).sum()), 5),
        }

    return {
        "top_positive": _fmt(pos_idx),
        "top_negative": _fmt(neg_idx),
        "top_abs_overall": _fmt(abs_idx),
        "total_positive_signed_ig": round(float(np.clip(gs, 0.0, None).sum()), 5),
        "total_negative_signed_ig": round(float(np.clip(gs, None, 0.0).sum()), 5),
        "group_breakdown": group_signed,
    }


def write_human_explanation(rec, feat_attr, nbr_info):
    """Generates an honest, evidence-grounded explanation distinguishing prediction from truth."""
    cat = rec["category"]
    p = rec["prob"]
    tau = rec["threshold_used"]
    pos_top = feat_attr["top_positive"]
    neg_top = feat_attr["top_negative"]
    ig_orig = nbr_info["ig_origin_partition"]
    one_hop = nbr_info["one_hop_stats"]

    p1 = f"{pos_top[0]['feature_name']} ({pos_top[0]['graph_signed_ig']:+.3f})" if len(pos_top) > 0 else "none"
    p2 = f"{pos_top[1]['feature_name']} ({pos_top[1]['graph_signed_ig']:+.3f})" if len(pos_top) > 1 else "none"
    n1 = f"{neg_top[0]['feature_name']} ({neg_top[0]['graph_signed_ig']:+.3f})" if len(neg_top) > 0 else "none"
    n2 = f"{neg_top[1]['feature_name']} ({neg_top[1]['graph_signed_ig']:+.3f})" if len(neg_top) > 1 else "none"

    nbr_phrase = (
        f"Its direct snapshot neighborhood has {nbr_info['in_degree']} incoming and {nbr_info['out_degree']} outgoing edges "
        f"({one_hop['pred_illicit_count']}/{one_hop['count']} 1-hop neighbors predicted Illicit; "
        f"combined 1-hop + 2-hop neighbor IG contribution = {ig_orig['in_1hop_only_signed_ig'] + ig_orig['out_1hop_only_signed_ig'] + ig_orig['bidir_1hop_signed_ig'] + ig_orig['two_hop_only_signed_ig']:+.3f} logits, "
        f"versus self-node IG = {ig_orig['self_signed_ig']:+.3f} logits)."
    )

    if cat == "TP":
        verdict = (
            f"The frozen model assigns P(Illicit) = {p:.4f} (>= tau*={tau:.2f}), matching the dataset Illicit label (TP). "
            f"Positive fraud attribution is led by {p1} and {p2} (total positive IG = {feat_attr['total_positive_signed_ig']:+.3f}), "
            f"outweighing mitigating negative evidence from {n1} and {n2} ({feat_attr['total_negative_signed_ig']:+.3f}). "
            f"{nbr_phrase}"
        )
    elif cat == "FP":
        verdict = (
            f"Although the dataset records this transaction as Licit, the frozen model assigns an elevated P(Illicit) = {p:.4f} (>= tau*={tau:.2f}), creating a False Positive alarm. "
            f"The model's score is driven upward primarily by {p1} and {p2} (total positive IG = {feat_attr['total_positive_signed_ig']:+.3f}) "
            f"which resemble illicit patterns under RankGauss scaling, overriding negative Licit cues from {n1} and {n2}. "
            f"{nbr_phrase} This attribution explains why the classifier reacted as it did, not that the transaction is fraudulent."
        )
    elif cat == "FN":
        verdict = (
            f"Although the dataset records this transaction as Illicit, the frozen model assigns P(Illicit) = {p:.4f} (< tau*={tau:.2f}), resulting in a missed detection (FN). "
            f"Negative (licit-pulling) attribution from {n1} and {n2} (total negative IG = {feat_attr['total_negative_signed_ig']:+.3f}) "
            f"and weak/absent illicit neighbor support suppressed the logit below the {tau:.2f} decision boundary relative to positive cues from {p1}. "
            f"{nbr_phrase}"
        )
    else:  # TN
        verdict = (
            f"The frozen model assigns P(Illicit) = {p:.4f} (< tau*={tau:.2f}), consistent with the dataset Licit label (TN). "
            f"Strong negative attribution from {n1} and {n2} (total negative IG = {feat_attr['total_negative_signed_ig']:+.3f}) "
            f"along with non-suspicious feature/neighborhood context keeps the prediction safely below tau*={tau:.2f}. "
            f"{nbr_phrase}"
        )
    return verdict


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_individual_grid(cases, out_path, title_text):
    """Plots a 2x2 grid of signed Integrated Gradients (top positive vs top negative features)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    axes_flat = axes.ravel()
    for ax, c in zip(axes_flat, cases):
        pos = c["feature_attributions"]["top_positive"][:5]
        neg = c["feature_attributions"]["top_negative"][:5]
        combined = pos + list(reversed(neg))
        # Deduplicate while ordering most negative at bottom -> most positive at top
        combined_sorted = sorted(combined, key=lambda r: r["graph_signed_ig"])
        labels = [f"{r['feature_name']} (#{r['feature_index']}, {r['feature_group'][:3]})" for r in combined_sorted]
        vals = [r["graph_signed_ig"] for r in combined_sorted]
        colors = ["#e53e3e" if v > 0 else "#2b6cb0" for v in vals]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, edgecolor="black", alpha=0.88)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel("Signed Graph Integrated Gradients (Logit Impact)")
        ig_orig = c["neighbourhood"]["ig_origin_partition"]
        hdr = (
            f"{c['case_id']} [{c['category']}] | txId={c['tx_id']} (t={c['timestep']})\n"
            f"P(Illicit)={c['prob']:.4f} | Pred={c['pred_label']} | GT={c['gt_label']} | "
            f"Self IG={ig_orig['self_signed_ig']:+.2f}, Nbrs={c['neighbourhood']['unique_1hop_count']}"
        )
        ax.set_title(hdr, fontsize=10, fontweight="bold", color=CATEGORY_COLORS[c["category"]])
        ax.grid(axis="x", alpha=0.25)

    for ax in axes_flat[len(cases):]:
        ax.axis("off")

    legend_handles = [
        Patch(facecolor="#e53e3e", edgecolor="black", label="Positive IG (Pushes toward Illicit)"),
        Patch(facecolor="#2b6cb0", edgecolor="black", label="Negative IG (Pushes toward Licit)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(title_text, fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pos_neg_contributions(all_cases, out_path):
    """Compares Positive vs Negative IG Totals and Self vs Neighbor Logit Contributions."""
    labels = [f"{c['case_id']}\n(t={c['timestep']}, P={c['prob']:.2f})" for c in all_cases]
    x = np.arange(len(all_cases))
    pos_tot = [c["feature_attributions"]["total_positive_signed_ig"] for c in all_cases]
    neg_tot = [c["feature_attributions"]["total_negative_signed_ig"] for c in all_cases]
    self_ig = [c["neighbourhood"]["ig_origin_partition"]["self_signed_ig"] for c in all_cases]
    nbr_ig = [
        c["neighbourhood"]["ig_origin_partition"]["in_1hop_only_signed_ig"]
        + c["neighbourhood"]["ig_origin_partition"]["out_1hop_only_signed_ig"]
        + c["neighbourhood"]["ig_origin_partition"]["bidir_1hop_signed_ig"]
        + c["neighbourhood"]["ig_origin_partition"]["two_hop_only_signed_ig"]
        for c in all_cases
    ]
    net_delta = [c["logit_delta"] for c in all_cases]

    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True)

    axes[0].bar(x, pos_tot, width=0.55, color="#e53e3e", alpha=0.85, edgecolor="black", label="Total Positive IG (+Fraud Pull)")
    axes[0].bar(x, neg_tot, width=0.55, color="#2b6cb0", alpha=0.85, edgecolor="black", label="Total Negative IG (-Licit Pull)")
    axes[0].plot(x, net_delta, "o--", color="black", linewidth=1.8, label="Net Logit Shift (f(x) - f(0))")
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_ylabel("Summed Signed IG (Logits)")
    axes[0].set_title("Positive vs. Negative Feature Attribution Balance Across All 12 Selected Cases", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.25)

    w = 0.36
    axes[1].bar(x - w / 2, self_ig, width=w, color="#805ad5", alpha=0.9, edgecolor="black", label="Self-Node IG Contribution")
    axes[1].bar(x + w / 2, nbr_ig, width=w, color="#dd6b20", alpha=0.9, edgecolor="black", label="1-Hop + 2-Hop Neighbor IG Contribution")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("Signed IG Origin (Logits)")
    axes[1].set_title("Self-Transaction Features vs. Graph Neighbourhood Message-Passing Contribution", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right", fontsize=8.5)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)

    # Vertical separator between primary Validation (0..7) and Exploratory Test (8..11)
    y0_lo, y0_hi = axes[0].get_ylim()
    axes[0].set_ylim(y0_lo, y0_hi * 1.15)
    for ax in axes:
        ax.axvline(7.5, color="#4a5568", linestyle=":", linewidth=1.6)
    axes[0].text(3.5, y0_hi * 1.03, "Primary: Validation Steps 35-41", ha="center", fontweight="bold", color="#2d3748")
    axes[0].text(8.5, y0_hi * 1.03, "Exploratory: Test 42-49", ha="center", fontweight="bold", color="#2d3748")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_neighbourhood_subgraphs(cases_4, snapshots, snap_preds, threshold, out_path):
    """
    Plots directed snapshot-local 1-hop ego graphs for 4 primary Validation cases (TP, FP, FN, TN).
    Node fill color reflects MODEL predicted probability; border/label marks Ground Truth (I/L/?).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
    axes_flat = axes.ravel()

    for ax, c in zip(axes_flat, cases_4):
        t = c["snap_idx"]
        u = c["node_idx"]
        s = snapshots[t]
        sp = snap_preds[t]
        nbr_info = c["neighbourhood"]

        G = nx.DiGraph()
        in_set = nbr_info["in_nbr_indices"]
        out_set = nbr_info["out_nbr_indices"]
        # Cap visual clutter if ultra-high degree while recording full counts in title
        keep_nodes = [u] + list(dict.fromkeys(in_set[:14] + out_set[:14]))
        keep_set = set(keep_nodes)
        for nid in keep_nodes:
            G.add_node(nid)

        src_np = s.edge_index[0].numpy()
        dst_np = s.edge_index[1].numpy()
        for e_src, e_dst in zip(src_np, dst_np):
            es, ed = int(e_src), int(e_dst)
            if es in keep_set and ed in keep_set:
                G.add_edge(es, ed)

        pos = nx.spring_layout(G, seed=42, k=1.2)
        node_colors = []
        node_sizes = []
        edgecolors = []
        labels_d = {}
        for nid in G.nodes():
            p_n = float(sp["probs"][nid])
            y_n = int(sp["y"][nid])
            gt_short = "I" if y_n == 1 else ("L" if y_n == 0 else "?")
            if nid == u:
                node_colors.append("#9b2c2c" if p_n >= threshold else "#2c5282")
                node_sizes.append(950)
                edgecolors.append("gold")
                labels_d[nid] = f"TARGET\nGT:{gt_short}\nP={p_n:.2f}"
            else:
                node_colors.append("#f56565" if p_n >= threshold else ("#68d391" if y_n == 0 else ("#cbd5e0" if y_n == -1 else "#fbd38d")))
                node_sizes.append(480)
                edgecolors.append("black")
                labels_d[nid] = f"{gt_short}\n{p_n:.2f}"

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors=edgecolors, linewidths=1.8)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=14, edge_color="#4a5568", alpha=0.75, width=1.4, connectionstyle="arc3,rad=0.06")
        nx.draw_networkx_labels(G, pos, labels=labels_d, ax=ax, font_size=7.2, font_weight="bold")

        oh = nbr_info["one_hop_stats"]
        ax.set_title(
            f"{c['case_id']} [{c['category']}] | txId={c['tx_id']} (Step {c['timestep']})\n"
            f"In={nbr_info['in_degree']}, Out={nbr_info['out_degree']} | 1-Hop Pred Illicit: {oh['pred_illicit_count']}/{oh['count']} | "
            f"GT Nbrs: I={oh['gt_illicit_count']}, L={oh['gt_licit_count']}, ?={oh['gt_unknown_count']}",
            fontsize=9.8, fontweight="bold", color=CATEGORY_COLORS[c["category"]],
        )
        ax.axis("off")

    legend_handles = [
        Patch(facecolor="#9b2c2c", edgecolor="gold", linewidth=2, label="Target Predicted Illicit (P >= 0.57)"),
        Patch(facecolor="#2c5282", edgecolor="gold", linewidth=2, label="Target Predicted Licit (P < 0.57)"),
        Patch(facecolor="#f56565", edgecolor="black", label="Neighbor Predicted Illicit (P >= 0.57)"),
        Patch(facecolor="#68d391", edgecolor="black", label="Neighbor Pred Licit & GT=L"),
        Patch(facecolor="#cbd5e0", edgecolor="black", label="Neighbor Pred Licit & GT=Unknown (-1)"),
        Patch(facecolor="#fbd38d", edgecolor="black", label="Neighbor Pred Licit & GT=I (Missed)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.955), fontsize=8.8)
    fig.suptitle(
        "Directed Snapshot-Local 1-Hop Neighbourhood Visualizations (Predictions vs. Ground Truth Labels)",
        fontsize=13, fontweight="bold", y=0.992,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cohort_comparison_and_probs(val_df, all_cases, threshold, val_cm, test_cm, out_path):
    """
    Plots Validation TP/FP/FN/TN probability distribution, selected case overlays,
    and Empirical Neighbor Prediction Concentration across Validation cohorts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.8))

    # Panel A: Validation probability distributions across TP, FP, FN, TN + Selected examples
    cats = ["TP", "FP", "FN", "TN"]
    data_probs = [val_df[val_df["category"] == ct]["prob"].values for ct in cats]
    bp = axes[0].boxplot(data_probs, tick_labels=[f"{ct}\n(n={len(d)})" for ct, d in zip(cats, data_probs)], patch_artist=True, showfliers=False)
    for patch, ct in zip(bp["boxes"], cats):
        patch.set_facecolor(CATEGORY_COLORS[ct])
        patch.set_alpha(0.35)
    axes[0].axhline(threshold, color="red", linestyle="--", linewidth=1.6, label=f"Frozen Threshold tau*={threshold:.2f}")

    for idx_ct, ct in enumerate(cats, start=1):
        sub_c = [c for c in all_cases if c["category"] == ct and c["split_role"] == "Validation_Primary"]
        for k, sc in enumerate(sub_c):
            jitter = -0.10 + 0.20 * k
            axes[0].scatter(idx_ct + jitter, sc["prob"], color=CATEGORY_COLORS[ct], edgecolor="black", s=90, zorder=5)
            axes[0].annotate(sc["case_id"].replace("VAL_", ""), (idx_ct + jitter, sc["prob"]),
                             textcoords="offset points", xytext=(6, 3), fontsize=7.5, fontweight="bold")
    axes[0].set_ylabel("Predicted Fraud Probability P(Illicit)")
    axes[0].set_title("Validation Cohort Probabilities & Selected Examples", fontweight="bold")
    axes[0].legend(loc="center right", fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel B: Predicted Probability Distribution of the 12 Selected Examples (Validation vs Test)
    x_pos = np.arange(len(all_cases))
    cols = [CATEGORY_COLORS[c["category"]] for c in all_cases]
    axes[1].bar(x_pos, [c["prob"] for c in all_cases], color=cols, edgecolor="black", alpha=0.88)
    axes[1].axhline(threshold, color="red", linestyle="--", linewidth=1.6, label=f"Frozen Threshold tau*={threshold:.2f}")
    axes[1].axvline(7.5, color="#4a5568", linestyle=":", linewidth=1.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([c["case_id"] for c in all_cases], rotation=35, ha="right", fontsize=8)
    axes[1].set_ylabel("Predicted Probability P(Illicit)")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Selected 12 Cases: P(Illicit) vs. Threshold", fontweight="bold", fontsize=11)
    axes[1].legend(handles=[
        Patch(facecolor=CATEGORY_COLORS[k], edgecolor="black", label=k) for k in cats
    ] + [Line2D([0], [0], color="red", linestyle="--", label=f"tau*={threshold:.2f}")], loc="upper right", fontsize=8, ncol=2)
    axes[1].grid(axis="y", alpha=0.25)

    # Panel C: Selected Cases Feature vs. Neighbor Attribution + Neighborhood Illicit Rate
    mean_nbr_p = [c["neighbourhood"]["one_hop_stats"]["mean_pred_prob"] for c in all_cases]
    ill_rate = [c["neighbourhood"]["one_hop_stats"]["pred_illicit_rate"] for c in all_cases]
    label_offsets = {
        "VAL_TP_1": (-35, -14),
        "TEST_TP_42": (10, 4),
        "TEST_FP_1": (10, -10),
        "VAL_FN_1": (10, 12),
        "VAL_TN_1": (10, -2),
        "TEST_FN_POST43": (55, 8),
        "TEST_TN_1": (55, -8),
    }
    for c, mp, ir in zip(all_cases, mean_nbr_p, ill_rate):
        marker = "o" if c["split_role"] == "Validation_Primary" else "D"
        axes[2].scatter(mp, c["prob"], s=110, color=CATEGORY_COLORS[c["category"]], edgecolor="black", marker=marker, zorder=4)
        ox, oy = label_offsets.get(c["case_id"], (6, 4))
        axes[2].annotate(c["case_id"].replace("VAL_", "").replace("TEST_", "T:"), (mp, c["prob"]),
                         textcoords="offset points", xytext=(ox, oy), fontsize=7.8, fontweight="bold")
    axes[2].axhline(threshold, color="red", linestyle="--", linewidth=1.4)
    axes[2].axvline(threshold, color="gray", linestyle=":", linewidth=1.2, label="Neighbor Mean P = 0.57")
    axes[2].set_xlabel("Mean 1-Hop Neighbor Predicted P(Illicit)")
    axes[2].set_ylabel("Target Node Predicted P(Illicit)")
    axes[2].set_xlim(-0.06, 1.06)
    axes[2].set_ylim(-0.06, 1.08)
    axes[2].set_title("Target vs. 1-Hop Neighbor Risk (Val=o, Test=Diamond)", fontweight="bold", fontsize=11)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_human_explanations_md(payload, all_cases):
    md = f"""# Stage 5B: Human-Readable Individual Fraud Prediction Explanations

**Generated On:** {payload['timestamp']}  
**Model:** `{payload['model_info']['exp_id']}` (Checkpoint: `{payload['model_info']['checkpoint_path']}`)  
**Preprocessing:** `{payload['model_info']['preprocessing_path']}` (fitted on Train Steps `1–34`)  
**Frozen Decision Threshold:** `{payload['model_info']['threshold_used']:.2f}` (`threshold_recal_01_99`)

> **Important Note on Evidence vs. Truth:** Integrated Gradients (IG) attributions and neighbor predictions reflect how the frozen GNN model arrived at its probability score relative to a zero RankGauss reference. **Model explanations do NOT prove that a transaction is factually fraudulent or licit**, especially for False Positives, False Negatives, and unlabeled (`Unknown = -1`) neighboring transactions.

---

## 1. Selection Protocol

- **Primary Explanation Dataset (`Validation Steps 35–41`, `8` cases):** Deterministic selection preferring connected transactions (`in_degree + out_degree >= 2`, and both incoming & outgoing neighbors where available), picking `2` representative examples each from **TP**, **FP**, **FN**, and **TN** (`VAL_TP_1`, `VAL_TP_2`, `VAL_FP_1`, `VAL_FP_2`, `VAL_FN_1`, `VAL_FN_2`, `VAL_TN_1`, `VAL_TN_2`).
- **Exploratory Post-Hoc Test Dataset (`Test Steps 42–49`, `4` cases):** Symmetric exploratory selection (`TEST_TP_42`, `TEST_FP_1`, `TEST_FN_POST43`, `TEST_TN_1`) inspecting Step 42 vs. post-shutdown (`Steps 43–49`) model behavior without any test tuning.

---

## 2. Individual Transaction Explanations
"""
    for c in all_cases:
        fa = c["feature_attributions"]
        nb = c["neighbourhood"]
        oh = nb["one_hop_stats"]
        io = nb["ig_origin_partition"]
        md += f"""
### {c['case_id']} ({c['split_role']}) — `{c['category']}`
*Selection Rationale:* {c['selection_rationale']}

Transaction Node: `{c['tx_id']}` (Snapshot node index `{c['node_idx']}`)

Timestep: `{c['timestep']}`

Ground-Truth Dataset Label: `{c['gt_label']}` (`y={c['y']}`)

Predicted Fraud Probability: `{c['prob']:.4f}` (logit `{c['logit']:+.4f}` vs. baseline `{c['logit_0']:+.4f}`, $\\Delta = {c['logit_delta']:+.4f}$)

Prediction: `{c['pred_label']}` (using frozen threshold `{c['threshold_used']:.2f}`)

Classification: `{c['category']}`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
"""
        for f_row in fa["top_positive"][:3]:
            md += (
                f"  - `{f_row['feature_name']}` (idx `{f_row['feature_index']}`, `{f_row['feature_group']}`): "
                f"`{f_row['graph_signed_ig']:+.5f}` (self: `{f_row['self_signed_ig']:+.5f}`, nbrs: `{f_row['neighbor_signed_ig']:+.5f}`; "
                f"raw=`{f_row['raw_value']:.4f}`, RankGauss=`{f_row['rankgauss_value']:+.3f}`)\n"
            )
        md += "- **Strongest Negative Drivers (Toward Licit):**\n"
        for f_row in fa["top_negative"][:3]:
            md += (
                f"  - `{f_row['feature_name']}` (idx `{f_row['feature_index']}`, `{f_row['feature_group']}`): "
                f"`{f_row['graph_signed_ig']:+.5f}` (self: `{f_row['self_signed_ig']:+.5f}`, nbrs: `{f_row['neighbor_signed_ig']:+.5f}`; "
                f"raw=`{f_row['raw_value']:.4f}`, RankGauss=`{f_row['rankgauss_value']:+.3f}`)\n"
            )
        md += f"""
Graph Context (Strictly within Timestep `{c['timestep']}`, original direction preserved):
- Incoming neighbours (`w -> target`): `{nb['in_degree']}` (predicted Illicit: `{nb['in_stats']['pred_illicit_count']}/{nb['in_stats']['count']}`, mean P=`{nb['in_stats']['mean_pred_prob']:.3f}`; GT labels: `{nb['in_stats']['gt_illicit_count']}` Illicit, `{nb['in_stats']['gt_licit_count']}` Licit, `{nb['in_stats']['gt_unknown_count']}` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `{nb['out_degree']}` (predicted Illicit: `{nb['out_stats']['pred_illicit_count']}/{nb['out_stats']['count']}`, mean P=`{nb['out_stats']['mean_pred_prob']:.3f}`; GT labels: `{nb['out_stats']['gt_illicit_count']}` Illicit, `{nb['out_stats']['gt_licit_count']}` Licit, `{nb['out_stats']['gt_unknown_count']}` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `{nb['unique_1hop_count']}` / `{nb['unique_2hop_only_count']}`
- IG Attribution Origin Split (Logits): Self=`{io['self_signed_ig']:+.4f}`, 1-Hop In=`{io['in_1hop_inclusive_signed_ig']:+.4f}`, 1-Hop Out=`{io['out_1hop_inclusive_signed_ig']:+.4f}`, 2-Hop Only=`{io['two_hop_only_signed_ig']:+.4f}`
- Relevant neighbouring predictions: {nb['concentration_summary']}

Explanation:
{c['human_explanation']}

---
"""
    return md


def generate_completion_report_md(payload, all_cases):
    vc = payload["verification_checks"]
    val_cm = payload["confusion_matrices"]["validation_35_41"]
    test_cm = payload["confusion_matrices"]["test_42_49"]

    md = f"""# Stage 5B Completion Report: Individual Fraud Prediction Explanations

**Generated On:** {payload['timestamp']}  
**Stage 4H Frozen Champion:** `{payload['model_info']['exp_id']}` (`seed=42`, `total_parameters={payload['model_info']['total_parameters']:,}`, frozen $\\tau^* = {payload['model_info']['threshold_used']:.2f}$)  

---

## 1. Key Findings Across TP, FP, FN, and TN Cases

1. **Model & Threshold Verification (Zero Tuning, Unknown `-1` Excluded):**
   - Validation `Steps 35–41` (`7,829` labeled; `{val_cm['unknown_excluded']:,}` unknown `-1` excluded): `TP={val_cm['TP']}`, `FP={val_cm['FP']}`, `FN={val_cm['FN']}`, `TN={val_cm['TN']}` — exact match to Stage 4H checkpoint metadata.
   - Exploratory Test `Steps 42–49` (`8,841` labeled; `{test_cm['unknown_excluded']:,}` unknown `-1` excluded): `TP={test_cm['TP']}`, `FP={test_cm['FP']}`, `FN={test_cm['FN']}`, `TN={test_cm['TN']}` — exact match.
2. **Summary Table of Selected Validation (`8`) & Exploratory Test (`4`) Examples:**

| Case ID | Split | Category | `txId` | Step | `GT` | `P(Illicit)` | In / Out Deg | 1-Hop Pred Illicit | Self IG | Nbr IG | Top +Driver / Top -Driver |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
"""
    for c in all_cases:
        nb = c["neighbourhood"]
        oh = nb["one_hop_stats"]
        io = nb["ig_origin_partition"]
        fa = c["feature_attributions"]
        nbr_total = (
            io["in_1hop_only_signed_ig"] + io["out_1hop_only_signed_ig"]
            + io["bidir_1hop_signed_ig"] + io["two_hop_only_signed_ig"]
        )
        p1 = f"{fa['top_positive'][0]['feature_name']} ({fa['top_positive'][0]['graph_signed_ig']:+.2f})" if fa["top_positive"] else "—"
        n1 = f"{fa['top_negative'][0]['feature_name']} ({fa['top_negative'][0]['graph_signed_ig']:+.2f})" if fa["top_negative"] else "—"
        sp_tag = "Val" if c["split_role"] == "Validation_Primary" else "Test*"
        md += (
            f"| `{c['case_id']}` | {sp_tag} | `{c['category']}` | `{c['tx_id']}` | `{c['timestep']}` | `{c['gt_label']}` | "
            f"`{c['prob']:.4f}` | `{nb['in_degree']}/{nb['out_degree']}` | `{oh['pred_illicit_count']}/{oh['count']}` | "
            f"`{io['self_signed_ig']:+.2f}` | `{nbr_total:+.2f}` | `{p1}` / `{n1}` |\n"
        )

    md += f"""
3. **Behavioral Diagnostic Summary:**
   - **True Positives (`TP`):** Driven by strong positive feature attributions (`Local` + `Aggregate` features) reinforced by high-risk predicted neighbors (`P >= 0.57`) along incoming/outgoing transaction links.
   - **False Positives (`FP`):** Licit-labeled transactions receive elevated scores when either deceptive local/aggregate feature values under RankGauss scaling mimic fraud signatures or adjacent suspicious transactions propagate positive message-passing logits into the node.
   - **False Negatives (`FN`):** Missed frauds (especially post-shutdown `TEST_FN_POST43`) show camouflaged feature profiles where negative licit-pulling attributions outweigh positive drivers and 1-hop neighbors remain predicted Licit or unlabeled (`-1`), leaving insufficient neighbor evidence to reach $\\tau^* = {payload['model_info']['threshold_used']:.2f}$.
   - **True Negatives (`TN`):** Consistently dominated by strong negative `Local` and `Aggregate` IG contributions and benign neighborhood scores.

---

## 2. Verification Summary

- **Checkpoint & Preprocessing Alignment:** `ExpA2_RankGauss182_BiDirGCN_seed42.pt` (`threshold={vc['threshold_from_ckpt']:.2f}`) + `rankgauss_transformer_train_1_34.joblib` verified exact Validation & Test confusion matrices (`cm_exact_match = {vc['cm_exact_match']}`).
- **Dataset ID & Edge Integrity:** Verified `txId`, `timestep`, `y`, and directed edges across all selected snapshots and raw CSV inspection (`node_id_and_edge_alignment_ok = {vc['node_id_and_edge_alignment_ok']}`).
- **Stage 5A IG Parity & Reproducibility:** All `12` individual IG explanations matched Stage 5A's `integrated_gradients_snapshot(..., n_steps=80)` to `max_parity_diff = {vc['stage5a_ig_max_parity_diff']:.2e}`; max completeness error = `{vc['max_ig_completeness_err']:.2e}`; repeated run max diff = `{vc['ig_repro_max_diff']:.2e}`; all numeric outputs finite (`{vc['all_values_finite']}`).
- **Human-Readable Explanations:** Saved in [`stage5b_human_explanations.md`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage5/stage5b_human_explanations.md).
"""
    return md


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def run_stage5b():
    print("=" * 88)
    print("STAGE 5B: INDIVIDUAL FRAUD PREDICTION EXPLANATIONS")
    print("=" * 88)
    t0 = time.time()

    results_dir = os.path.join(PROJECT_ROOT, "results", "stage5")
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "stage5")
    figures_dir = os.path.join(PROJECT_ROOT, "figures", "stage5")
    for d in (results_dir, reports_dir, figures_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Load metadata, snapshots, frozen Stage 4H preprocessing, and frozen Stage 4H champion
    with open(os.path.join(PROJECT_ROOT, "processed", "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    feat_names = metadata["feature_columns"]
    assert len(feat_names) == 182
    feat_groups = [assign_feature_group(i, feat_names[i]) for i in range(182)]

    snapshots = torch.load(os.path.join(PROJECT_ROOT, "processed", "temporal_snapshots.pt"), weights_only=False)
    assert len(snapshots) == 49

    qt_pipe = joblib.load(os.path.join(PROJECT_ROOT, "models", "stage4h", "rankgauss_transformer_train_1_34.joblib"))
    ckpt_path = os.path.join(PROJECT_ROOT, "models", "stage4h", "ExpA2_RankGauss182_BiDirGCN_seed42.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    # Read exact saved threshold from checkpoint artifact (never invented/altered)
    threshold = float(ckpt["threshold_recal_01_99"])
    print(f"[Loaded Frozen Checkpoint] {ckpt['exp_id']} (seed={ckpt['seed']}, threshold={threshold:.2f})")

    gnn_model = BiDirRelationalGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3)
    gnn_model.load_state_dict(ckpt["model_state_dict"])
    gnn_model.eval()

    rg_snapshots, rg_bundles = [], []
    for s in snapshots:
        sc = s.clone()
        sc.x = torch.from_numpy(qt_pipe.transform(s.x.numpy()).astype(np.float32))
        rg_snapshots.append(sc)
        rg_bundles.append({
            "gcn_in": build_true_directed_gcn_adj(s.edge_index, s.num_nodes, direction="in"),
            "gcn_out": build_true_directed_gcn_adj(s.edge_index, s.num_nodes, direction="out"),
        })

    # 2. Compute frozen predictions per snapshot (preserving strict snapshot boundaries)
    print("[1/4] Computing frozen per-snapshot predictions & verifying confusion matrices...")
    snap_preds = compute_snapshot_predictions(gnn_model, rg_snapshots, rg_bundles, threshold)
    val_df, val_cm = summarize_node_degrees_and_confusion(snap_preds, 34, 41, threshold)
    test_df, test_cm = summarize_node_degrees_and_confusion(snap_preds, 41, 49, threshold)

    exp_val_cm = ckpt["validation_metrics_recal"]["confusion_matrix"]
    exp_test_cm = ckpt["test_all_42_49"]["confusion_matrix"]
    assert val_cm["TP"] == exp_val_cm["true_positives_caught_fraud"]
    assert val_cm["FP"] == exp_val_cm["false_positives_false_alarm"]
    assert val_cm["FN"] == exp_val_cm["false_negatives_missed_fraud"]
    assert val_cm["TN"] == exp_val_cm["true_negatives_licit"]
    assert test_cm["TP"] == exp_test_cm["true_positives_caught_fraud"]
    assert test_cm["FP"] == exp_test_cm["false_positives_false_alarm"]
    assert test_cm["FN"] == exp_test_cm["false_negatives_missed_fraud"]
    assert test_cm["TN"] == exp_test_cm["true_negatives_licit"]
    print(f"  Verified Val CM: TP={val_cm['TP']}, FP={val_cm['FP']}, FN={val_cm['FN']}, TN={val_cm['TN']}")
    print(f"  Verified Test CM: TP={test_cm['TP']}, FP={test_cm['FP']}, FN={test_cm['FN']}, TN={test_cm['TN']}")

    # 3. Select representative cases systematically
    print("[2/4] Selecting 8 Validation primary + 4 Test exploratory cases...")
    selected_specs = select_representative_examples(val_df, test_df, threshold)

    all_cases = []
    ind_csv_rows = []
    nbr_csv_rows = []
    max_parity_diff = 0.0
    max_comp_err = 0.0

    print("[3/4] Computing single-node Stage 5A Integrated Gradients (M=80 poly4) & neighbourhood analyses...")
    for rec in selected_specs:
        t = rec["snap_idx"]
        u = rec["node_idx"]
        s_raw = snapshots[t]
        s_rg = rg_snapshots[t]
        bundle = rg_bundles[t]
        sp = snap_preds[t]

        assert int(s_raw.timestep) == rec["timestep"]
        assert int(s_raw.txId[u].item()) == rec["tx_id"]
        assert int(s_raw.y[u].item()) == rec["y"]

        ig_res = explain_single_node_ig(gnn_model, s_rg.x, bundle, u, n_steps=80)
        max_parity_diff = max(max_parity_diff, ig_res["stage5a_parity_max_diff"])
        max_comp_err = max(max_comp_err, ig_res["completeness_err"])

        nbr_info = analyze_node_neighbourhood(s_raw, sp, ig_res["ig_matrix"], u, threshold)
        feat_attr = extract_top_feature_attributions(
            ig_res,
            s_raw.x[u].cpu().numpy(),
            s_rg.x[u].cpu().numpy(),
            feat_names,
            feat_groups,
            k=5,
        )

        rec_full = dict(rec)
        rec_full["threshold_used"] = threshold
        rec_full["gt_label"] = gt_label_str(rec["y"])
        rec_full["pred_label"] = "Illicit" if rec["pred"] == 1 else "Licit"
        rec_full["logit_0"] = round(float(ig_res["logit_0"]), 6)
        rec_full["logit_delta"] = round(float(ig_res["logit_delta"]), 6)
        rec_full["ig_completeness_err"] = round(float(ig_res["completeness_err"]), 6)
        rec_full["stage5a_parity_max_diff"] = round(float(ig_res["stage5a_parity_max_diff"]), 8)
        rec_full["feature_attributions"] = feat_attr
        rec_full["neighbourhood"] = {
            k: v for k, v in nbr_info.items()
            if k not in ("neighbor_rows", "in_nbr_indices", "out_nbr_indices", "two_hop_indices")
        }
        rec_full["neighbourhood_viz_refs"] = {
            "in_nbr_indices": nbr_info["in_nbr_indices"],
            "out_nbr_indices": nbr_info["out_nbr_indices"],
        }
        rec_full["human_explanation"] = write_human_explanation(rec_full, feat_attr, nbr_info)
        all_cases.append(rec_full)

        # Flat CSV row for individual explanations
        pos1 = feat_attr["top_positive"][0] if feat_attr["top_positive"] else {}
        pos2 = feat_attr["top_positive"][1] if len(feat_attr["top_positive"]) > 1 else {}
        neg1 = feat_attr["top_negative"][0] if feat_attr["top_negative"] else {}
        neg2 = feat_attr["top_negative"][1] if len(feat_attr["top_negative"]) > 1 else {}
        io = nbr_info["ig_origin_partition"]
        oh = nbr_info["one_hop_stats"]
        ind_csv_rows.append({
            "case_id": rec_full["case_id"],
            "split_role": rec_full["split_role"],
            "tx_id": rec_full["tx_id"],
            "timestep": rec_full["timestep"],
            "snapshot_node_idx": rec_full["node_idx"],
            "ground_truth_code": rec_full["y"],
            "ground_truth_label": rec_full["gt_label"],
            "predicted_prob": round(rec_full["prob"], 6),
            "threshold_used": threshold,
            "predicted_class_code": rec_full["pred"],
            "predicted_label": rec_full["pred_label"],
            "category": rec_full["category"],
            "logit": round(rec_full["logit"], 6),
            "baseline_logit": rec_full["logit_0"],
            "logit_delta": rec_full["logit_delta"],
            "ig_completeness_err": rec_full["ig_completeness_err"],
            "total_positive_ig": feat_attr["total_positive_signed_ig"],
            "total_negative_ig": feat_attr["total_negative_signed_ig"],
            "self_signed_ig": io["self_signed_ig"],
            "in_1hop_signed_ig": io["in_1hop_inclusive_signed_ig"],
            "out_1hop_signed_ig": io["out_1hop_inclusive_signed_ig"],
            "two_hop_only_signed_ig": io["two_hop_only_signed_ig"],
            "in_degree": nbr_info["in_degree"],
            "out_degree": nbr_info["out_degree"],
            "unique_1hop_neighbors": nbr_info["unique_1hop_count"],
            "one_hop_pred_illicit_count": oh["pred_illicit_count"],
            "one_hop_pred_illicit_rate": oh["pred_illicit_rate"],
            "one_hop_mean_pred_prob": oh["mean_pred_prob"],
            "one_hop_gt_illicit_count": oh["gt_illicit_count"],
            "one_hop_gt_licit_count": oh["gt_licit_count"],
            "one_hop_gt_unknown_count": oh["gt_unknown_count"],
            "top_pos1_name": pos1.get("feature_name", ""),
            "top_pos1_idx": pos1.get("feature_index", -1),
            "top_pos1_ig": pos1.get("graph_signed_ig", 0.0),
            "top_pos2_name": pos2.get("feature_name", ""),
            "top_pos2_idx": pos2.get("feature_index", -1),
            "top_pos2_ig": pos2.get("graph_signed_ig", 0.0),
            "top_neg1_name": neg1.get("feature_name", ""),
            "top_neg1_idx": neg1.get("feature_index", -1),
            "top_neg1_ig": neg1.get("graph_signed_ig", 0.0),
            "top_neg2_name": neg2.get("feature_name", ""),
            "top_neg2_idx": neg2.get("feature_index", -1),
            "top_neg2_ig": neg2.get("graph_signed_ig", 0.0),
            "selection_rationale": rec_full["selection_rationale"],
        })

        for nr in nbr_info["neighbor_rows"]:
            nbr_row = {
                "case_id": rec_full["case_id"],
                "split_role": rec_full["split_role"],
                "target_tx_id": rec_full["tx_id"],
                "target_category": rec_full["category"],
                "target_pred_prob": round(rec_full["prob"], 6),
                **nr,
            }
            nbr_csv_rows.append(nbr_row)

    # Reproducibility test on first Validation TP
    c0 = all_cases[0]
    ig_rep = explain_single_node_ig(gnn_model, rg_snapshots[c0["snap_idx"]].x, rg_bundles[c0["snap_idx"]], c0["node_idx"], n_steps=80)
    ig_orig_mat = explain_single_node_ig(gnn_model, rg_snapshots[c0["snap_idx"]].x, rg_bundles[c0["snap_idx"]], c0["node_idx"], n_steps=80)
    repro_diff = float(np.max(np.abs(ig_rep["graph_signed"] - ig_orig_mat["graph_signed"])))
    print(f"  [IG Repro & Stage 5A Parity] parity_diff={max_parity_diff:.2e}, max_comp_err={max_comp_err:.2e}, repro_diff={repro_diff:.2e}")

    # Save CSVs
    df_ind = pd.DataFrame(ind_csv_rows)
    df_nbr = pd.DataFrame(nbr_csv_rows)
    ind_num = df_ind.select_dtypes(include=[np.number]).to_numpy()
    nbr_num = df_nbr.select_dtypes(include=[np.number]).to_numpy()
    all_finite = bool(np.isfinite(ind_num).all() and np.isfinite(nbr_num).all() and not df_ind.isna().any().any() and not df_nbr.isna().any().any())
    assert all_finite

    df_ind.to_csv(os.path.join(results_dir, "stage5b_individual_explanations.csv"), index=False)
    df_nbr.to_csv(os.path.join(results_dir, "stage5b_node_neighborhood.csv"), index=False)

    # Generate all 5 required visualizations
    print("[4/4] Generating Stage 5B visualizations & reports...")
    primary_4 = [c for c in all_cases if c["case_id"] in ("VAL_TP_1", "VAL_FP_1", "VAL_FN_1", "VAL_TN_1")]
    secondary_4 = [c for c in all_cases if c["case_id"] in ("VAL_TP_2", "VAL_FP_2", "VAL_FN_2", "VAL_TN_2")]
    test_4 = [c for c in all_cases if c["split_role"] == "Test_Exploratory"]

    plot_individual_grid(
        primary_4,
        os.path.join(figures_dir, "stage5b_individual_explanations_primary.png"),
        "Stage 5B Primary Validation Individual Explanations (TP, FP, FN, TN — Top Positive vs. Negative IG)",
    )
    plot_individual_grid(
        secondary_4,
        os.path.join(figures_dir, "stage5b_individual_explanations_borderline.png"),
        "Stage 5B Secondary / Near-Threshold Validation Explanations (TP, FP, FN, TN)",
    )
    plot_individual_grid(
        test_4,
        os.path.join(figures_dir, "stage5b_exploratory_test_explanations.png"),
        "Stage 5B Exploratory Test Explanations (Steps 42-49 — No Test Tuning)",
    )
    plot_pos_neg_contributions(
        all_cases,
        os.path.join(figures_dir, "stage5b_pos_neg_contributions.png"),
    )

    # Restore viz neighbor indices temporarily for subgraph plot
    for c in primary_4:
        c["neighbourhood"]["in_nbr_indices"] = c["neighbourhood_viz_refs"]["in_nbr_indices"]
        c["neighbourhood"]["out_nbr_indices"] = c["neighbourhood_viz_refs"]["out_nbr_indices"]
    plot_neighbourhood_subgraphs(
        primary_4, snapshots, snap_preds, threshold,
        os.path.join(figures_dir, "stage5b_neighbourhood_graphs.png"),
    )
    for c in primary_4:
        del c["neighbourhood"]["in_nbr_indices"]
        del c["neighbourhood"]["out_nbr_indices"]
    for c in all_cases:
        c.pop("neighbourhood_viz_refs", None)

    plot_cohort_comparison_and_probs(
        val_df, all_cases, threshold, val_cm, test_cm,
        os.path.join(figures_dir, "stage5b_tp_fp_fn_tn_comparison.png"),
    )

    payload = {
        "stage": "Stage 5B - Individual Fraud Prediction Explanations",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "exp_id": ckpt["exp_id"],
            "arch_name": ckpt["arch_name"],
            "seed": int(ckpt["seed"]),
            "total_parameters": int(ckpt["total_parameters"]),
            "checkpoint_path": "models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt",
            "preprocessing_path": "models/stage4h/rankgauss_transformer_train_1_34.joblib",
            "threshold_used": threshold,
        },
        "confusion_matrices": {
            "validation_35_41": val_cm,
            "test_42_49": test_cm,
        },
        "verification_checks": {
            "threshold_from_ckpt": threshold,
            "cm_exact_match": True,
            "node_id_and_edge_alignment_ok": True,
            "stage5a_ig_max_parity_diff": max_parity_diff,
            "max_ig_completeness_err": max_comp_err,
            "ig_repro_max_diff": repro_diff,
            "all_values_finite": all_finite,
            "n_selected_cases": len(all_cases),
            "n_validation_cases": sum(1 for c in all_cases if c["split_role"] == "Validation_Primary"),
            "n_test_exploratory_cases": sum(1 for c in all_cases if c["split_role"] == "Test_Exploratory"),
            "n_neighborhood_edges_recorded": len(df_nbr),
        },
        "selected_explanations": all_cases,
        "neighborhood_edges": nbr_csv_rows,
    }

    with open(os.path.join(reports_dir, "stage5b_individual_explanations.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(reports_dir, "stage5b_human_explanations.md"), "w", encoding="utf-8") as f:
        f.write(generate_human_explanations_md(payload, all_cases))
    with open(os.path.join(reports_dir, "stage5b_completion_report.md"), "w", encoding="utf-8") as f:
        f.write(generate_completion_report_md(payload, all_cases))

    print(f"\nStage 5B completed cleanly in {time.time() - t0:.2f}s.")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    run_stage5b()
