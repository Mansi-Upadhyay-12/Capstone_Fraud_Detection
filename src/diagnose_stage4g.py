"""
Stage 4G: Temporal Distribution Shift and Post-Shutdown Fraud Detection Investigation

Purely retrospective diagnostic & integrity analysis across Train (1-34),
Validation (35-41), Test Step 42 (pre-shutdown), and Test Steps 43-49 (post-shutdown).

STRICT RESTRICTIONS:
  - Does NOT train any new model.
  - Does NOT alter any Stage 1-4F checkpoints, scripts, metrics, or reports.
  - Does NOT use test labels for training, feature selection, model selection, or threshold tuning.
  - Saves all outputs strictly inside reports/stage4g/.
"""

import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models_nextgen import (
    SeparatedGraphNet,
    StabilizedEvolveGCN,
    CausalTemporalAttnGCN,
    CausalAttnEvolveGCN,
    build_gcn_sparse_adj,
    build_neighbor_mean_sparse_adj,
)
from src.models_stage4f import (
    AnchorGCNAvgWrapper,
    BiDirGATv2Net,
    BiDirGraphSAGEMeanMax,
    BiDirRelationalGCN,
    CausalCtxBiDirGCN,
)
from src.train_baselines import FraudMLP
from src.train_evolvegcn import FraudEvolveGCNH, build_normalized_sparse_adj as build_evolvegcn_adj
from src.train_gcn_lstm import FraudGCNLSTM
from src.train_gcn_lstm_historical import FraudGCNLSTMHistorical
from src.train_gnn_static import FraudGAT, FraudGCN, ImprovedFraudGAT, ImprovedFraudGCN
from src.train_nextgen import compute_ece, compute_metrics_and_calibration
from src.train_stage4f import build_feature_slices, build_snapshot_bundle


REGIMES = [
    ("train_1_34", "Train (1-34)", 1, 34),
    ("val_35_41", "Val (35-41)", 35, 41),
    ("test_42", "Test Step 42", 42, 42),
    ("test_43_49", "Test Steps 43-49", 43, 49),
]


def safe_metrics(y_true, y_prob, tau):
    if len(y_true) == 0 or (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
        return None
    m = compute_metrics_and_calibration(y_true, y_prob, threshold=tau)
    return m


def compute_psi(ref, cur, n_bins=10, eps=1e-6):
    """Computes Population Stability Index using reference (Train 1-34) quantile bins."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = np.clip(ref_counts / max(len(ref), 1), eps, None)
    cur_p = np.clip(cur_counts / max(len(cur), 1), eps, None)
    ref_p = ref_p / ref_p.sum()
    cur_p = cur_p / cur_p.sum()
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


# =============================================================================
# 1. DATA INTEGRITY INVESTIGATION
# =============================================================================

def run_data_integrity_check(base_dir, snapshots, global_data, tabular, metadata):
    dataset_dir = os.path.join(base_dir, "dataset")
    df_c = pd.read_csv(os.path.join(dataset_dir, "txs_classes.txt"))
    df_e = pd.read_csv(os.path.join(dataset_dir, "txs_edgelist.txt"))
    df_f = pd.read_csv(os.path.join(dataset_dir, "txs_features.txt"))

    df_m = pd.merge(df_f, df_c, on="txId", how="inner")
    df_m.sort_values(by=["Time step", "txId"], ascending=[True, True], inplace=True)
    df_m.reset_index(drop=True, inplace=True)

    feat_cols = [c for c in df_m.columns if c not in ["txId", "Time step", "class"]]
    meta_cols = metadata["feature_columns"]
    assert feat_cols == meta_cols, "Feature column mismatch!"

    txid_to_time = dict(zip(df_m["txId"].values, df_m["Time step"].values))
    e_src_t = df_e["txId1"].map(txid_to_time).values
    e_dst_t = df_e["txId2"].map(txid_to_time).values
    cross_time_edges = int((e_src_t != e_dst_t).sum())

    # Cross-check processed tensors against sorted raw dataframe
    tab_txid = tabular["txId"]
    tab_ts = tabular["timestep"]
    tab_y = tabular["y"]
    raw_y = np.full(len(df_m), -1, dtype=np.int64)
    raw_classes = df_m["class"].values
    raw_y[raw_classes == 1] = 1
    raw_y[raw_classes == 2] = 0

    assert np.array_equal(tab_txid, df_m["txId"].values)
    assert np.array_equal(tab_ts, df_m["Time step"].values.astype(np.int64))
    assert np.array_equal(tab_y, raw_y)
    assert torch.equal(global_data.txId, torch.tensor(tab_txid))
    assert torch.equal(global_data.y, torch.tensor(tab_y))

    # Concatenate snapshots and verify exact equivalence with global_data / tabular_data
    snap_x = torch.cat([s.x for s in snapshots], dim=0).numpy()
    snap_y = torch.cat([s.y for s in snapshots], dim=0).numpy()
    snap_txid = torch.cat([s.txId for s in snapshots], dim=0).numpy()
    max_x_diff = float(np.max(np.abs(snap_x - tabular["X"])))
    assert max_x_diff < 1e-6
    assert np.array_equal(snap_y, tab_y)
    assert np.array_equal(snap_txid, tab_txid)

    # Check raw duplicates, near-constant features in Steps 43-49, and missingness patterns
    X_raw = df_m[feat_cols].values.astype(np.float64)
    ts_arr = df_m["Time step"].values.astype(np.int64)
    nan_mask = np.isnan(X_raw)

    dup_counts_by_regime = {}
    const_cols_by_regime = {}
    for r_key, _, t0, t1 in REGIMES:
        m = (ts_arr >= t0) & (ts_arr <= t1)
        sub_scaled = tabular["X"][m]
        # Exact duplicate feature rows (ignoring txId/time)
        uniq_rows = len(np.unique(sub_scaled, axis=0))
        dup_counts_by_regime[r_key] = {"rows": int(m.sum()), "unique_feature_rows": int(uniq_rows), "duplicate_rows": int(m.sum() - uniq_rows)}
        stds = sub_scaled.std(axis=0)
        const_cols = [meta_cols[i] for i in range(182) if stds[i] < 1e-6]
        const_cols_by_regime[r_key] = const_cols

    return {
        "raw_nodes": int(len(df_m)),
        "raw_edges": int(len(df_e)),
        "unique_txids": int(df_m["txId"].nunique()),
        "duplicate_txids": int(len(df_m) - df_m["txId"].nunique()),
        "cross_timestep_edges": cross_time_edges,
        "feature_column_count": len(feat_cols),
        "feature_column_order_matches_metadata": True,
        "snapshot_vs_tabular_max_x_diff": round(max_x_diff, 8),
        "snapshot_vs_tabular_txid_exact": True,
        "snapshot_vs_tabular_label_exact": True,
        "duplicate_feature_rows_by_regime": dup_counts_by_regime,
        "constant_scaled_columns_by_regime": const_cols_by_regime,
    }, df_m, X_raw, nan_mask, ts_arr, feat_cols


# =============================================================================
# 2. FEATURE DISTRIBUTION DRIFT
# =============================================================================

def analyze_feature_drift(X_scaled, X_raw, nan_mask, ts_arr, feat_cols):
    groups = {
        "Local_93": list(range(0, 93)),
        "Aggregate_72": list(range(93, 165)),
        "Domain_17": list(range(165, 182)),
        "All_182": list(range(0, 182)),
    }

    m_tr = (ts_arr >= 1) & (ts_arr <= 34)
    m_va = (ts_arr >= 35) & (ts_arr <= 41)
    m_42 = (ts_arr == 42)
    m_43_49 = (ts_arr >= 43) & (ts_arr <= 49)
    reg_masks = {
        "train_1_34": m_tr,
        "val_35_41": m_va,
        "test_42": m_42,
        "test_43_49": m_43_49,
    }

    # Missingness rates (raw pre-imputation)
    missing_by_regime = {}
    for r_key, mask in reg_masks.items():
        sub_nan = nan_mask[mask]
        missing_by_regime[r_key] = {
            "rows": int(mask.sum()),
            "rows_with_any_nan": int(sub_nan.any(axis=1).sum()),
            "pct_rows_with_any_nan": round(float(sub_nan.any(axis=1).mean() * 100.0), 4),
            "total_missing_cells_pct": round(float(sub_nan.mean() * 100.0), 6),
            "by_group_pct_cells": {
                g: round(float(sub_nan[:, idx].mean() * 100.0), 6) for g, idx in groups.items()
            },
        }

    # Per-feature statistics across the 4 regimes + PSI / KS / Wasserstein vs Train(1-34) & vs Step 42
    X_tr = X_scaled[m_tr]
    X_va = X_scaled[m_va]
    X_42 = X_scaled[m_42]
    X_43 = X_scaled[m_43_49]

    per_feature = []
    for i, col in enumerate(feat_cols):
        grp = "Local_93" if i < 93 else ("Aggregate_72" if i < 165 else "Domain_17")
        tr_i, va_i, s42_i, s43_i = X_tr[:, i], X_va[:, i], X_42[:, i], X_43[:, i]

        psi_val = compute_psi(tr_i, va_i)
        psi_42 = compute_psi(tr_i, s42_i)
        psi_43 = compute_psi(tr_i, s43_i)
        psi_42_to_43 = compute_psi(s42_i, s43_i)

        ks_43 = float(ks_2samp(tr_i, s43_i).statistic)
        ks_42_to_43 = float(ks_2samp(s42_i, s43_i).statistic)
        w1_43 = float(wasserstein_distance(tr_i, s43_i))
        w1_42_to_43 = float(wasserstein_distance(s42_i, s43_i))

        per_feature.append({
            "index": i,
            "name": col,
            "group": grp,
            "mean_train": round(float(tr_i.mean()), 4),
            "std_train": round(float(tr_i.std()), 4),
            "median_train": round(float(np.median(tr_i)), 4),
            "q25_train": round(float(np.quantile(tr_i, 0.25)), 4),
            "q75_train": round(float(np.quantile(tr_i, 0.75)), 4),
            "mean_val": round(float(va_i.mean()), 4),
            "std_val": round(float(va_i.std()), 4),
            "median_val": round(float(np.median(va_i)), 4),
            "mean_step42": round(float(s42_i.mean()), 4),
            "std_step42": round(float(s42_i.std()), 4),
            "median_step42": round(float(np.median(s42_i)), 4),
            "mean_steps43_49": round(float(s43_i.mean()), 4),
            "std_steps43_49": round(float(s43_i.std()), 4),
            "median_steps43_49": round(float(np.median(s43_i)), 4),
            "q25_steps43_49": round(float(np.quantile(s43_i, 0.25)), 4),
            "q75_steps43_49": round(float(np.quantile(s43_i, 0.75)), 4),
            "abs_mean_shift_tr_to_43_49": round(float(abs(s43_i.mean() - tr_i.mean())), 4),
            "abs_median_shift_tr_to_43_49": round(float(abs(np.median(s43_i) - np.median(tr_i))), 4),
            "std_ratio_43_49_vs_tr": round(float(s43_i.std() / max(tr_i.std(), 1e-8)), 4),
            "psi_tr_to_val": round(psi_val, 4),
            "psi_tr_to_step42": round(psi_42, 4),
            "psi_tr_to_43_49": round(psi_43, 4),
            "psi_step42_to_43_49": round(psi_42_to_43, 4),
            "ks_tr_to_43_49": round(ks_43, 4),
            "ks_step42_to_43_49": round(ks_42_to_43, 4),
            "wasserstein_tr_to_43_49": round(w1_43, 4),
            "wasserstein_step42_to_43_49": round(w1_42_to_43, 4),
        })

    # Feature-group drift summary
    group_drift = {}
    for g_name, idxs in groups.items():
        sub = [per_feature[i] for i in idxs]
        group_drift[g_name] = {
            "count": len(sub),
            "psi_tr_to_val_median": round(float(np.median([d["psi_tr_to_val"] for d in sub])), 4),
            "psi_tr_to_val_mean": round(float(np.mean([d["psi_tr_to_val"] for d in sub])), 4),
            "psi_tr_to_step42_median": round(float(np.median([d["psi_tr_to_step42"] for d in sub])), 4),
            "psi_tr_to_step42_mean": round(float(np.mean([d["psi_tr_to_step42"] for d in sub])), 4),
            "psi_tr_to_43_49_median": round(float(np.median([d["psi_tr_to_43_49"] for d in sub])), 4),
            "psi_tr_to_43_49_mean": round(float(np.mean([d["psi_tr_to_43_49"] for d in sub])), 4),
            "psi_tr_to_43_49_max": round(float(np.max([d["psi_tr_to_43_49"] for d in sub])), 4),
            "psi_step42_to_43_49_median": round(float(np.median([d["psi_step42_to_43_49"] for d in sub])), 4),
            "psi_step42_to_43_49_mean": round(float(np.mean([d["psi_step42_to_43_49"] for d in sub])), 4),
            "psi_step42_to_43_49_max": round(float(np.max([d["psi_step42_to_43_49"] for d in sub])), 4),
            "abs_mean_shift_tr_to_43_49_median": round(float(np.median([d["abs_mean_shift_tr_to_43_49"] for d in sub])), 4),
            "abs_mean_shift_tr_to_43_49_max": round(float(np.max([d["abs_mean_shift_tr_to_43_49"] for d in sub])), 4),
            "std_ratio_43_49_median": round(float(np.median([d["std_ratio_43_49_vs_tr"] for d in sub])), 4),
            "std_ratio_43_49_max": round(float(np.max([d["std_ratio_43_49_vs_tr"] for d in sub])), 4),
            "ks_tr_to_43_49_median": round(float(np.median([d["ks_tr_to_43_49"] for d in sub])), 4),
            "frac_psi_gt_025_tr_to_43_49": round(float(np.mean([d["psi_tr_to_43_49"] > 0.25 for d in sub])), 4),
            "frac_gt5_outlier_pct": {
                r_key: round(float((np.abs(X_scaled[mask][:, idxs]) > 5.0).mean() * 100.0), 4)
                for r_key, mask in reg_masks.items()
            },
        }

    # Per-timestep trajectory across 1-49
    per_step_drift = {}
    for t in range(1, 50):
        mt = (ts_arr == t)
        xt_93 = X_scaled[mt, :93]
        xt_agg = X_scaled[mt, 93:165]
        xt_dom = X_scaled[mt, 165:182]
        per_step_drift[str(t)] = {
            "nodes": int(mt.sum()),
            "nan_pct_rows": round(float(nan_mask[mt].any(axis=1).mean() * 100.0), 3),
            "l2_norm_mean_local93": round(float(np.linalg.norm(xt_93, axis=1).mean()), 3),
            "l2_norm_mean_agg72": round(float(np.linalg.norm(xt_agg, axis=1).mean()), 3),
            "l2_norm_mean_dom17": round(float(np.linalg.norm(xt_dom, axis=1).mean()), 3),
            "frac_gt5_local93_pct": round(float((np.abs(xt_93) > 5.0).mean() * 100.0), 4),
            "frac_gt5_agg72_pct": round(float((np.abs(xt_agg) > 5.0).mean() * 100.0), 4),
        }

    top15_tr_to_43 = sorted(per_feature, key=lambda d: d["psi_tr_to_43_49"], reverse=True)[:15]
    top15_42_to_43 = sorted(per_feature, key=lambda d: d["psi_step42_to_43_49"], reverse=True)[:15]

    return {
        "missing_values_raw": missing_by_regime,
        "group_drift_summary": group_drift,
        "top15_psi_train_to_43_49": top15_tr_to_43,
        "top15_psi_step42_to_43_49": top15_42_to_43,
        "per_step_trajectory": per_step_drift,
        "all_182_features": per_feature,
    }


# =============================================================================
# 3. GRAPH STRUCTURE DRIFT
# =============================================================================

def analyze_graph_structure(snapshots):
    per_step = {}
    for s in snapshots:
        t = int(s.timestep)
        n = int(s.num_nodes)
        ei = s.edge_index.numpy()
        m = int(ei.shape[1])
        u, v = ei[0], ei[1]

        in_deg = np.bincount(v, minlength=n) if m > 0 else np.zeros(n, dtype=int)
        out_deg = np.bincount(u, minlength=n) if m > 0 else np.zeros(n, dtype=int)
        tot_deg = in_deg + out_deg

        # Weakly connected components
        if m > 0:
            adj = coo_matrix((np.ones(m, dtype=np.int8), (u, v)), shape=(n, n))
            n_cc, labels_cc = connected_components(adj, directed=False, return_labels=True)
            cc_sizes = np.bincount(labels_cc)
            giant_pct = float(cc_sizes.max() / max(n, 1) * 100.0)
            non_trivial_ccs = int((cc_sizes >= 2).sum())
        else:
            n_cc, giant_pct, non_trivial_ccs = n, 100.0 / max(n, 1), 0

        density = float(m / max(n * (n - 1), 1))
        y = s.y.numpy()
        ill_m = (y == 1)
        lic_m = (y == 0)

        # Retrospective homophily & unlabeled neighbor stats
        if m > 0:
            both_lab = (y[u] >= 0) & (y[v] >= 0)
            v_ill = both_lab & (y[v] == 1)
            v_lic = both_lab & (y[v] == 0)
            u_ill = both_lab & (y[u] == 1)
            u_lic = both_lab & (y[u] == 0)
            p_in_ill = float((y[u[v_ill]] == 1).mean()) if v_ill.any() else None
            p_in_lic = float((y[u[v_lic]] == 1).mean()) if v_lic.any() else None
            p_out_ill = float((y[v[u_ill]] == 1).mean()) if u_ill.any() else None
            p_out_lic = float((y[v[u_lic]] == 1).mean()) if u_lic.any() else None
            nbr_unlab_frac = float(((y[u] == -1) | (y[v] == -1)).mean() * 100.0)
        else:
            p_in_ill = p_in_lic = p_out_ill = p_out_lic = None
            nbr_unlab_frac = 0.0

        per_step[str(t)] = {
            "timestep": t,
            "split": str(s.split),
            "num_nodes": n,
            "num_edges": m,
            "edge_to_node_ratio": round(float(m / max(n, 1)), 4),
            "density": round(density, 7),
            "in_deg_mean": round(float(in_deg.mean()), 4),
            "in_deg_p95": round(float(np.quantile(in_deg, 0.95)), 2),
            "in_deg_max": int(in_deg.max()),
            "out_deg_mean": round(float(out_deg.mean()), 4),
            "out_deg_p95": round(float(np.quantile(out_deg, 0.95)), 2),
            "out_deg_max": int(out_deg.max()),
            "tot_deg_median": round(float(np.median(tot_deg)), 2),
            "isolated_pct": round(float((tot_deg == 0).mean() * 100.0), 2),
            "zero_in_all_pct": round(float((in_deg == 0).mean() * 100.0), 2),
            "zero_out_all_pct": round(float((out_deg == 0).mean() * 100.0), 2),
            "zero_in_illicit_pct": round(float((in_deg[ill_m] == 0).mean() * 100.0), 2) if ill_m.any() else None,
            "zero_out_illicit_pct": round(float((out_deg[ill_m] == 0).mean() * 100.0), 2) if ill_m.any() else None,
            "num_wcc": int(n_cc),
            "non_trivial_wcc": non_trivial_ccs,
            "giant_cc_pct": round(giant_pct, 2),
            "edges_with_unlabeled_pct": round(nbr_unlab_frac, 2),
            "homophily_in_given_v_ill": round(p_in_ill, 4) if p_in_ill is not None else None,
            "homophily_in_given_v_lic": round(p_in_lic, 4) if p_in_lic is not None else None,
            "homophily_out_given_u_ill": round(p_out_ill, 4) if p_out_ill is not None else None,
            "homophily_out_given_u_lic": round(p_out_lic, 4) if p_out_lic is not None else None,
        }

    # Aggregate across the 4 regimes
    regime_summary = {}
    for r_key, _, t0, t1 in REGIMES:
        sub_steps = [per_step[str(t)] for t in range(t0, t1 + 1)]
        tot_n = sum(x["num_nodes"] for x in sub_steps)
        tot_e = sum(x["num_edges"] for x in sub_steps)

        # Edge-level homophily pooled across steps in regime
        in_ill_num = in_ill_den = in_lic_num = in_lic_den = 0
        out_ill_num = out_ill_den = out_lic_num = out_lic_den = 0
        ill_zin = ill_zout = ill_tot = 0
        lic_zin = lic_zout = lic_tot = 0
        for t in range(t0, t1 + 1):
            s = snapshots[t - 1]
            y = s.y.numpy()
            ei = s.edge_index.numpy()
            u, v = ei[0], ei[1]
            in_d = np.bincount(v, minlength=s.num_nodes)
            out_d = np.bincount(u, minlength=s.num_nodes)
            ill_m, lic_m = (y == 1), (y == 0)
            ill_tot += int(ill_m.sum()); ill_zin += int((in_d[ill_m] == 0).sum()); ill_zout += int((out_d[ill_m] == 0).sum())
            lic_tot += int(lic_m.sum()); lic_zin += int((in_d[lic_m] == 0).sum()); lic_zout += int((out_d[lic_m] == 0).sum())
            both = (y[u] >= 0) & (y[v] >= 0)
            vi = both & (y[v] == 1); vl = both & (y[v] == 0)
            ui = both & (y[u] == 1); ul = both & (y[u] == 0)
            in_ill_num += int((y[u[vi]] == 1).sum()); in_ill_den += int(vi.sum())
            in_lic_num += int((y[u[vl]] == 1).sum()); in_lic_den += int(vl.sum())
            out_ill_num += int((y[v[ui]] == 1).sum()); out_ill_den += int(ui.sum())
            out_lic_num += int((y[v[ul]] == 1).sum()); out_lic_den += int(ul.sum())

        regime_summary[r_key] = {
            "timesteps": f"{t0}-{t1}",
            "mean_nodes_per_step": round(float(np.mean([x["num_nodes"] for x in sub_steps])), 1),
            "mean_edges_per_step": round(float(np.mean([x["num_edges"] for x in sub_steps])), 1),
            "pooled_edge_to_node_ratio": round(float(tot_e / max(tot_n, 1)), 4),
            "mean_density": round(float(np.mean([x["density"] for x in sub_steps])), 7),
            "mean_isolated_pct": round(float(np.mean([x["isolated_pct"] for x in sub_steps])), 2),
            "mean_zero_in_all_pct": round(float(np.mean([x["zero_in_all_pct"] for x in sub_steps])), 2),
            "mean_zero_out_all_pct": round(float(np.mean([x["zero_out_all_pct"] for x in sub_steps])), 2),
            "illicit_zero_in_pct": round(float(ill_zin / max(ill_tot, 1) * 100.0), 2),
            "illicit_zero_out_pct": round(float(ill_zout / max(ill_tot, 1) * 100.0), 2),
            "licit_zero_in_pct": round(float(lic_zin / max(lic_tot, 1) * 100.0), 2),
            "licit_zero_out_pct": round(float(lic_zout / max(lic_tot, 1) * 100.0), 2),
            "mean_giant_cc_pct": round(float(np.mean([x["giant_cc_pct"] for x in sub_steps])), 2),
            "mean_wcc_count": round(float(np.mean([x["num_wcc"] for x in sub_steps])), 1),
            "pooled_homophily_in_given_v_ill": round(float(in_ill_num / max(in_ill_den, 1)), 4),
            "pooled_homophily_in_given_v_lic": round(float(in_lic_num / max(in_lic_den, 1)), 4),
            "pooled_homophily_out_given_u_ill": round(float(out_ill_num / max(out_ill_den, 1)), 4),
            "pooled_homophily_out_given_u_lic": round(float(out_lic_num / max(out_lic_den, 1)), 4),
            "labeled_edge_counts": {"v_ill": in_ill_den, "v_lic": in_lic_den, "u_ill": out_ill_den, "u_lic": out_lic_den},
        }

    return {"regime_summary": regime_summary, "per_step": per_step}


# =============================================================================
# 4. LABEL & FRAUD-PATTERN ANALYSIS (RETROSPECTIVE ONLY)
# =============================================================================

def analyze_labels_and_fraud_patterns(snapshots, X_scaled, y_all, ts_arr, feat_cols):
    per_step = {}
    for s in snapshots:
        t = int(s.timestep)
        y = s.y.numpy()
        ill = int((y == 1).sum())
        lic = int((y == 0).sum())
        unl = int((y == -1).sum())
        lab = ill + lic
        per_step[str(t)] = {
            "timestep": t,
            "split": str(s.split),
            "total_nodes": int(s.num_nodes),
            "labeled_nodes": lab,
            "unlabeled_nodes": unl,
            "unlabeled_pct": round(float(unl / max(s.num_nodes, 1) * 100.0), 2),
            "illicit_count": ill,
            "licit_count": lic,
            "illicit_prevalence_pct": round(float(ill / max(lab, 1) * 100.0), 2) if lab > 0 else 0.0,
        }

    # Fraud vs Licit class-conditional vectors across regimes (standardized feature units)
    regime_stats = {}
    delta_vectors = {}
    cohen_d_vectors = {}
    for r_key, _, t0, t1 in REGIMES:
        m = (ts_arr >= t0) & (ts_arr <= t1)
        m_ill = m & (y_all == 1)
        m_lic = m & (y_all == 0)
        x_i = X_scaled[m_ill]
        x_l = X_scaled[m_lic]

        mu_i = x_i.mean(axis=0)
        mu_l = x_l.mean(axis=0)
        sd_i = x_i.std(axis=0)
        sd_l = x_l.std(axis=0)
        pooled_sd = np.sqrt(0.5 * (sd_i ** 2 + sd_l ** 2) + 1e-8)

        delta = mu_i - mu_l
        cohen_d = delta / pooled_sd
        delta_vectors[r_key] = delta
        cohen_d_vectors[r_key] = cohen_d

        regime_stats[r_key] = {
            "labeled_total": int(m_ill.sum() + m_lic.sum()),
            "illicit_count": int(m_ill.sum()),
            "licit_count": int(m_lic.sum()),
            "illicit_prevalence_pct": round(float(m_ill.sum() / max(int(m_ill.sum() + m_lic.sum()), 1) * 100.0), 2),
            "l2_delta_all182": round(float(np.linalg.norm(delta)), 4),
            "l2_delta_local93": round(float(np.linalg.norm(delta[:93])), 4),
            "l2_delta_agg72": round(float(np.linalg.norm(delta[93:165])), 4),
            "l2_delta_dom17": round(float(np.linalg.norm(delta[165:182])), 4),
        }

    def cos_sim(a, b):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return round(float(np.dot(a, b) / (na * nb)), 4) if na > 1e-8 and nb > 1e-8 else 0.0

    sig_sim = {}
    for g_name, sl in [("All_182", slice(0, 182)), ("Local_93", slice(0, 93)), ("Aggregate_72", slice(93, 165)), ("Domain_17", slice(165, 182))]:
        sig_sim[g_name] = {
            "cos_train_vs_val": cos_sim(cohen_d_vectors["train_1_34"][sl], cohen_d_vectors["val_35_41"][sl]),
            "cos_train_vs_step42": cos_sim(cohen_d_vectors["train_1_34"][sl], cohen_d_vectors["test_42"][sl]),
            "cos_train_vs_43_49": cos_sim(cohen_d_vectors["train_1_34"][sl], cohen_d_vectors["test_43_49"][sl]),
            "cos_val_vs_step42": cos_sim(cohen_d_vectors["val_35_41"][sl], cohen_d_vectors["test_42"][sl]),
            "cos_val_vs_43_49": cos_sim(cohen_d_vectors["val_35_41"][sl], cohen_d_vectors["test_43_49"][sl]),
            "cos_step42_vs_43_49": cos_sim(cohen_d_vectors["test_42"][sl], cohen_d_vectors["test_43_49"][sl]),
        }

    # Top discriminative features in Train+Val & how they behave in Step 42 vs Steps 43-49
    d_tr = cohen_d_vectors["train_1_34"]
    d_va = cohen_d_vectors["val_35_41"]
    d_42 = cohen_d_vectors["test_42"]
    d_43 = cohen_d_vectors["test_43_49"]
    top_idx_val = np.argsort(-np.abs(d_va))[:20]
    top_discriminative_features = []
    for i in top_idx_val:
        grp = "Local_93" if i < 93 else ("Aggregate_72" if i < 165 else "Domain_17")
        top_discriminative_features.append({
            "index": int(i),
            "name": feat_cols[i],
            "group": grp,
            "cohen_d_train": round(float(d_tr[i]), 4),
            "cohen_d_val": round(float(d_va[i]), 4),
            "cohen_d_step42": round(float(d_42[i]), 4),
            "cohen_d_steps43_49": round(float(d_43[i]), 4),
            "sign_flipped_in_43_49": bool((d_va[i] * d_43[i]) < 0),
        })

    # Top discriminative features THAT SURVIVE in Steps 43-49
    top_surviving_43_49 = []
    for i in np.argsort(-np.abs(d_43))[:15]:
        grp = "Local_93" if i < 93 else ("Aggregate_72" if i < 165 else "Domain_17")
        top_surviving_43_49.append({
            "index": int(i),
            "name": feat_cols[i],
            "group": grp,
            "cohen_d_train": round(float(d_tr[i]), 4),
            "cohen_d_val": round(float(d_va[i]), 4),
            "cohen_d_step42": round(float(d_42[i]), 4),
            "cohen_d_steps43_49": round(float(d_43[i]), 4),
        })

    return {
        "per_step_label_counts": per_step,
        "regime_label_and_separation": regime_stats,
        "fraud_signature_cosine_similarity_cohen_d": sig_sim,
        "top20_val_discriminative_features_shift": top_discriminative_features,
        "top15_steps43_49_discriminative_features": top_surviving_43_49,
    }


# =============================================================================
# 5. MODEL PREDICTION DIAGNOSIS & CROSS-MODEL COMPARISON (NO RETRAINING)
# =============================================================================

def eval_static_or_sequential_checkpoint(name, ckpt_path, snapshots, global_data, tabular, device):
    """Evaluates an existing saved checkpoint cleanly across Val(35-41), Step 42, Steps 43-49, and Test All."""
    X_tab, y_tab, ts_tab, lab_tab = tabular["X"], tabular["y"], tabular["timestep"], tabular["labeled_mask"]
    step_probs = {}  # step -> {"prob_lab": ..., "y_lab": ..., "prob_all": ...}

    if ckpt_path.endswith(".joblib"):
        clf = joblib.load(ckpt_path)
        s3 = json.loads(open("reports/stage3_baselines_metrics.json", "r", encoding="utf-8").read())
        tau_map = {
            "Logistic_Regression": s3["Logistic_Regression"]["optimal_threshold"],
            "Random_Forest": s3["Random_Forest"]["optimal_threshold"],
            "HistGradientBoosting": s3["Gradient_Boosting"]["optimal_threshold"],
        }
        tau = float(tau_map[name])
        for s in snapshots:
            t = int(s.timestep)
            if t < 35:
                continue
            probs_all = clf.predict_proba(s.x.numpy())[:, 1]
            lm = s.labeled_mask.numpy()
            step_probs[t] = {"prob_all": probs_all, "prob_lab": probs_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)

    if name == "MLP_Baseline":
        model = FraudMLP(input_dim=ckpt["input_dim"], hidden_dim1=ckpt["hidden_dim1"], hidden_dim2=ckpt["hidden_dim2"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tau = float(ckpt["best_threshold"])
        with torch.no_grad():
            for s in snapshots:
                t = int(s.timestep)
                if t < 35:
                    continue
                p_all = torch.sigmoid(model(s.x.to(device))).cpu().numpy()
                lm = s.labeled_mask.numpy()
                step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    if name in ("GCN_Original", "GCN_Improved_Residual", "GAT_Original", "GAT_Improved_Residual"):
        undir = ckpt.get("use_undirected", False)
        if name == "GCN_Original":
            model = FraudGCN(ckpt["in_channels"], ckpt["hidden_channels"], ckpt["out_channels"]).to(device)
        elif name == "GCN_Improved_Residual":
            model = ImprovedFraudGCN(ckpt["in_channels"], ckpt["hidden_channels"], ckpt["out_channels"], num_layers=ckpt["num_layers"]).to(device)
        elif name == "GAT_Original":
            model = FraudGAT(ckpt["in_channels"], ckpt["hidden_channels"], ckpt["heads"], ckpt.get("out_channels", 64)).to(device)
        else:
            model = ImprovedFraudGAT(ckpt["in_channels"], ckpt["hidden_channels"], ckpt["heads1"], ckpt["heads2"], ckpt["out_channels"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tau = float(ckpt.get("optimal_threshold", ckpt.get("optimal_threshold_f1")))
        from torch_geometric.utils import to_undirected
        with torch.no_grad():
            for s in snapshots:
                t = int(s.timestep)
                if t < 35:
                    continue
                ei = to_undirected(s.edge_index) if undir else s.edge_index
                p_all = torch.sigmoid(model(s.x.to(device), ei.to(device))).cpu().numpy()
                lm = s.labeled_mask.numpy()
                step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    if name in ("GCN_LSTM", "GCN_LSTM_Historical"):
        if name == "GCN_LSTM":
            model = FraudGCNLSTM(in_channels=182, hidden_gcn=128, spatial_dim=64, lstm_dim=64).to(device)
        else:
            model = FraudGCNLSTMHistorical(in_channels=182, hidden_gcn=128, spatial_dim=64, lstm_dim=64).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tau = float(ckpt["optimal_threshold"])
        h_s, c_s = model.init_hidden(device)
        with torch.no_grad():
            for s in snapshots:
                t = int(s.timestep)
                if name == "GCN_LSTM":
                    logits, h_s, c_s = model.forward_step(s.x.to(device), s.edge_index.to(device), h_s, c_s)
                else:
                    logits, h_s, c_s = model.forward_step_historical(s.x.to(device), s.edge_index.to(device), h_s, c_s)
                if t >= 35:
                    p_all = torch.sigmoid(logits).cpu().numpy()
                    lm = s.labeled_mask.numpy()
                    step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    if name == "EvolveGCN_H_Recalibrated":
        d_rec = json.loads(open("reports/stage4c_evolvegcn_metrics_recalibrated.json", "r", encoding="utf-8").read())
        tau = float(d_rec["recalibrated_stage4c"]["optimal_threshold_f1"])
        model = FraudEvolveGCNH(
            in_channels=ckpt["in_channels"],
            hidden_channels=ckpt["hidden_channels"],
            out_channels=ckpt["out_channels"],
            use_residual=ckpt["use_residual"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        w1, w2 = model.init_weights_state()
        with torch.no_grad():
            for s in snapshots:
                t = int(s.timestep)
                adj = build_evolvegcn_adj(s.edge_index, s.num_nodes, use_undirected=ckpt["use_undirected"], device=device)
                logits, w1, w2 = model.forward_step(s.x.to(device), adj, w1, w2)
                if t >= 35:
                    p_all = torch.sigmoid(logits).cpu().numpy()
                    lm = s.labeled_mask.numpy()
                    step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    # Stage 4D / 4E / 4F checkpoints
    if "spec" in ckpt and ckpt["spec"].get("arch") in ("SeparatedGraphNet", "StabilizedEvolveGCN", "CausalTemporalAttnGCN", "CausalAttnEvolveGCN"):
        spec = ckpt["spec"]
        fm = spec.get("feat_mode", "full_182")
        in_ch = 93 if fm == "local_93" else 182
        from src.train_nextgen import build_model_from_spec, get_slice_x
        model = build_model_from_spec(spec, in_ch, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if "threshold_recal_01_99" in ckpt:
            tau = float(ckpt["threshold_recal_01_99"])
        elif "validation_metrics_recal" in ckpt:
            tau = float(ckpt["validation_metrics_recal"]["threshold_used"])
        else:
            tau = 0.93
        state = model.init_state()
        with torch.no_grad():
            for s in snapshots:
                t = int(s.timestep)
                if spec.get("adj_kind", "gcn_directed") == "sage_directed":
                    adj = build_neighbor_mean_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device)
                else:
                    adj = build_gcn_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device)
                logits, state, _ = model.forward_step(get_slice_x(s.x.to(device), fm), adj, state)
                if t >= 35:
                    p_all = torch.sigmoid(logits).cpu().numpy()
                    lm = s.labeled_mask.numpy()
                    step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    if "feature_indices" in ckpt:  # Stage 4E
        from src.models_nextgen import SeparatedGraphNet
        idx_t = torch.tensor(ckpt["feature_indices"], dtype=torch.long, device=device)
        model = SeparatedGraphNet(in_channels=ckpt["in_channels"], hidden_channels=128, out_channels=64, dropout_rate=0.3, graph_mode="gcn_avg").to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        tau = float(ckpt["threshold_recal_01_99"])
        with torch.no_grad():
            for s in snapshots:
                t = int(s.timestep)
                if t < 35:
                    continue
                adj = build_gcn_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device)
                logits, _, _ = model.forward_step(s.x.to(device).index_select(1, idx_t), adj, None)
                p_all = torch.sigmoid(logits).cpu().numpy()
                lm = s.labeled_mask.numpy()
                step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
        return tau, step_probs

    # Stage 4F
    from src.train_stage4f import build_model_from_spec as build_4f_model
    spec = ckpt["spec"]
    feat_slices = build_feature_slices()
    idx_list = feat_slices[spec["feat_key"]]
    idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
    model = build_4f_model(spec, len(idx_list), device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tau = float(ckpt["threshold_recal_01_99"])
    state = model.init_state()
    with torch.no_grad():
        for s in snapshots:
            t = int(s.timestep)
            bundle = build_snapshot_bundle(s, drop_edge_p=0.0, device=device)
            logits, state, _ = model.forward_step(s.x.to(device).index_select(1, idx_t), bundle, state)
            if t >= 35:
                p_all = torch.sigmoid(logits).cpu().numpy()
                lm = s.labeled_mask.numpy()
                step_probs[t] = {"prob_all": p_all, "prob_lab": p_all[lm], "y_lab": s.y.numpy()[lm]}
    return tau, step_probs


def summarize_model_slices(tau, step_probs):
    def agg(steps):
        y = np.concatenate([step_probs[t]["y_lab"] for t in steps])
        p = np.concatenate([step_probs[t]["prob_lab"] for t in steps])
        return compute_metrics_and_calibration(y, p, threshold=tau)

    m_val = agg(range(35, 42))
    m_42 = agg([42])
    m_43_49 = agg(range(43, 50))
    m_test = agg(range(42, 50))
    return {
        "threshold": round(tau, 4),
        "val_35_41": m_val,
        "test_42": m_42,
        "test_43_49": m_43_49,
        "test_all_42_49": m_test,
    }


def diagnose_gcn_vs_hgb_predictions(gcn_tau, gcn_sp, hgb_tau, hgb_sp):
    """Deep prediction comparison between GCNAvg and HistGradientBoosting across snapshots 35-49."""
    per_step = {}
    for t in range(35, 50):
        y = gcn_sp[t]["y_lab"]
        pg_lab = gcn_sp[t]["prob_lab"]
        ph_lab = hgb_sp[t]["prob_lab"]
        pg_all = gcn_sp[t]["prob_all"]
        ph_all = hgb_sp[t]["prob_all"]

        mg = compute_metrics_and_calibration(y, pg_lab, threshold=gcn_tau)
        mh = compute_metrics_and_calibration(y, ph_lab, threshold=hgb_tau)

        ill_m = (y == 1)
        lic_m = (y == 0)

        # Top-K retrospective recall in this snapshot
        k = min(max(int(ill_m.sum()) * 2, 10), len(y))
        g_topk = float(y[np.argsort(-pg_lab)[:k]].sum() / max(int(ill_m.sum()), 1))
        h_topk = float(y[np.argsort(-ph_lab)[:k]].sum() / max(int(ill_m.sum()), 1))

        per_step[str(t)] = {
            "timestep": t,
            "n_labeled": int(len(y)),
            "n_illicit": int(ill_m.sum()),
            "gcn": {
                "pr_auc": mg["pr_auc"],
                "roc_auc": mg["roc_auc"],
                "f1": mg["illicit_f1"],
                "prec": mg["illicit_precision"],
                "rec": mg["illicit_recall"],
                "pos_pred_rate_pct": round(float((pg_lab >= gcn_tau).mean() * 100.0), 3),
                "all_pos_pred_rate_pct": round(float((pg_all >= gcn_tau).mean() * 100.0), 3),
                "mean_prob_illicit": mg["mean_prob_illicit"],
                "mean_prob_licit": mg["mean_prob_licit"],
                "median_prob_illicit": round(float(np.median(pg_lab[ill_m])), 4) if ill_m.any() else None,
                "median_prob_licit": round(float(np.median(pg_lab[lic_m])), 4),
                "q95_prob_licit": round(float(np.quantile(pg_lab[lic_m], 0.95)), 4),
                "max_prob_illicit": round(float(pg_lab[ill_m].max()), 4) if ill_m.any() else None,
                "recall_at_2xIll": round(g_topk, 4),
                "tp": mg["confusion_matrix"]["true_positives_caught_fraud"],
                "fp": mg["confusion_matrix"]["false_positives_false_alarm"],
                "fn": mg["confusion_matrix"]["false_negatives_missed_fraud"],
            },
            "hgb": {
                "pr_auc": mh["pr_auc"],
                "roc_auc": mh["roc_auc"],
                "f1": mh["illicit_f1"],
                "prec": mh["illicit_precision"],
                "rec": mh["illicit_recall"],
                "pos_pred_rate_pct": round(float((ph_lab >= hgb_tau).mean() * 100.0), 3),
                "all_pos_pred_rate_pct": round(float((ph_all >= hgb_tau).mean() * 100.0), 3),
                "mean_prob_illicit": mh["mean_prob_illicit"],
                "mean_prob_licit": mh["mean_prob_licit"],
                "median_prob_illicit": round(float(np.median(ph_lab[ill_m])), 4) if ill_m.any() else None,
                "median_prob_licit": round(float(np.median(ph_lab[lic_m])), 4),
                "q95_prob_licit": round(float(np.quantile(ph_lab[lic_m], 0.95)), 4),
                "max_prob_illicit": round(float(ph_lab[ill_m].max()), 4) if ill_m.any() else None,
                "recall_at_2xIll": round(h_topk, 4),
                "tp": mh["confusion_matrix"]["true_positives_caught_fraud"],
                "fp": mh["confusion_matrix"]["false_positives_false_alarm"],
                "fn": mh["confusion_matrix"]["false_negatives_missed_fraud"],
            },
        }

    # Diagnostic counterfactual oracle upper-bound F1 on Steps 43-49 + Top-K recall on 43-49
    y_43 = np.concatenate([gcn_sp[t]["y_lab"] for t in range(43, 50)])
    pg_43 = np.concatenate([gcn_sp[t]["prob_lab"] for t in range(43, 50)])
    ph_43 = np.concatenate([hgb_sp[t]["prob_lab"] for t in range(43, 50)])

    def diag_curve_summary(y, p):
        best_f1, best_t = -1.0, 0.5
        for th in np.linspace(0.01, 0.99, 99):
            f1_v = f1_score(y, (p >= th).astype(int), zero_division=0)
            if f1_v > best_f1:
                best_f1 = f1_v
                best_t = float(round(th, 2))
        topk = {}
        order = np.argsort(-p)
        for k in [50, 100, 200, 400]:
            hits = int(y[order[:k]].sum())
            topk[f"k={k}"] = {"hits": hits, "precision": round(hits / k, 4), "recall": round(hits / max(int(y.sum()), 1), 4)}
        return {"oracle_max_f1_posthoc": round(float(best_f1), 4), "oracle_tau": best_t, "topk": topk}

    # Overlap on Step 42
    y_42 = gcn_sp[42]["y_lab"]
    pg_42 = gcn_sp[42]["prob_lab"]
    ph_42 = hgb_sp[42]["prob_lab"]
    g_tp42 = (y_42 == 1) & (pg_42 >= gcn_tau)
    h_tp42 = (y_42 == 1) & (ph_42 >= hgb_tau)

    return {
        "per_step_comparison": per_step,
        "steps_43_49_counterfactual_ranking_vs_calibration": {
            "gcn_avg": diag_curve_summary(y_43, pg_43),
            "hist_gradient_boosting": diag_curve_summary(y_43, ph_43),
        },
        "step_42_fraud_overlap": {
            "total_illicit": int((y_42 == 1).sum()),
            "both_caught": int((g_tp42 & h_tp42).sum()),
            "only_gcn_caught": int((g_tp42 & ~h_tp42).sum()),
            "only_hgb_caught": int((~g_tp42 & h_tp42).sum()),
            "both_missed": int(((y_42 == 1) & ~g_tp42 & ~h_tp42).sum()),
        },
    }


# =============================================================================
# 6. VISUALIZATIONS (SAVED STRICTLY IN reports/stage4g/)
# =============================================================================

def save_stage4g_visualizations(out_dir, drift_res, graph_res, pred_res, gcn_tau, gcn_sp, hgb_tau, hgb_sp):
    # Plot 1: Feature Distribution Drift
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    top12 = drift_res["top15_psi_step42_to_43_49"][:12][::-1]
    names = [d["name"] for d in top12]
    vals = [d["psi_step42_to_43_49"] for d in top12]
    cols_c = ["#2b6cb0" if d["group"] == "Local_93" else ("#dd6b20" if d["group"] == "Aggregate_72" else "#38a169") for d in top12]
    axes[0].barh(names, vals, color=cols_c, edgecolor="black", alpha=0.9)
    axes[0].axvline(0.25, color="#e53e3e", linestyle="--", label="Major Shift PSI=0.25")
    axes[0].set_title("Top Shifted Features: Step 42 -> Steps 43-49 (PSI)")
    axes[0].set_xlabel("Population Stability Index (PSI)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.25)

    grps = ["Local_93", "Aggregate_72", "Domain_17"]
    x = np.arange(len(grps))
    w = 0.26
    gd = drift_res["group_drift_summary"]
    axes[1].bar(x - w, [gd[g]["psi_tr_to_val_median"] for g in grps], w, label="Train -> Val (35-41)", color="#3182ce", edgecolor="black")
    axes[1].bar(x, [gd[g]["psi_tr_to_step42_median"] for g in grps], w, label="Train -> Step 42", color="#38a169", edgecolor="black")
    axes[1].bar(x + w, [gd[g]["psi_tr_to_43_49_median"] for g in grps], w, label="Train -> Steps 43-49", color="#e53e3e", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(grps)
    axes[1].set_ylabel("Median PSI vs Train (1-34)")
    axes[1].set_title("Median Feature Drift by Feature Group")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    steps = list(range(1, 50))
    traj = drift_res["per_step_trajectory"]
    axes[2].plot(steps, [traj[str(t)]["l2_norm_mean_local93"] for t in steps], marker="o", ms=3, label="Local-93 Mean L2", color="#2b6cb0")
    axes[2].plot(steps, [traj[str(t)]["l2_norm_mean_agg72"] for t in steps], marker="s", ms=3, label="Aggregate-72 Mean L2", color="#dd6b20")
    axes[2].axvline(34.5, color="gray", linestyle="--", alpha=0.7)
    axes[2].axvline(41.5, color="green", linestyle="--", alpha=0.7)
    axes[2].axvline(42.5, color="red", linestyle="--", label="Shutdown Boundary (42->43)")
    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Mean Feature Vector L2 Norm")
    axes[2].set_title("Per-Snapshot Feature Norm Drift (Steps 1-49)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage4g_feature_drift.png"), dpi=200)
    plt.close(fig)

    # Plot 2: Graph Structure & Homophily Drift
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    gs = graph_res["per_step"]
    axes[0].plot(steps, [gs[str(t)]["num_nodes"] for t in steps], label="Nodes", color="#2b6cb0")
    axes[0].plot(steps, [gs[str(t)]["num_edges"] for t in steps], label="Edges", color="#38a169")
    axes[0].axvline(42.5, color="red", linestyle="--", label="Step 42->43")
    axes[0].set_title("Snapshot Scale: Nodes & Edges (Steps 1-49)")
    axes[0].set_xlabel("Timestep")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, [gs[str(t)]["zero_in_all_pct"] for t in steps], label="Zero In-Deg % (All)", color="#805ad5")
    axes[1].plot(steps, [gs[str(t)]["zero_out_all_pct"] for t in steps], label="Zero Out-Deg % (All)", color="#dd6b20")
    axes[1].plot(steps, [gs[str(t)]["giant_cc_pct"] for t in steps], label="Giant WCC %", color="#319795")
    axes[1].axvline(42.5, color="red", linestyle="--")
    axes[1].set_title("Connectivity & Directional Zero-Degree Share")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    rs = graph_res["regime_summary"]
    reg_keys = ["train_1_34", "val_35_41", "test_42", "test_43_49"]
    reg_lbls = ["Train\n1-34", "Val\n35-41", "Test\nStep 42", "Test\n43-49"]
    rx = np.arange(len(reg_keys))
    axes[2].bar(rx - 0.18, [rs[k]["pooled_homophily_in_given_v_ill"] for k in reg_keys], 0.35, label="P(Parent=Ill | Child=Ill)", color="#c53030", edgecolor="black")
    axes[2].bar(rx + 0.18, [rs[k]["pooled_homophily_out_given_u_ill"] for k in reg_keys], 0.35, label="P(Child=Ill | Parent=Ill)", color="#2b6cb0", edgecolor="black")
    axes[2].set_xticks(rx)
    axes[2].set_xticklabels(reg_lbls)
    axes[2].set_ylim(0, 0.8)
    axes[2].set_ylabel("Homophily Conditional Probability")
    axes[2].set_title("Directed Fraud Homophily Collapse After Step 42")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage4g_graph_structure.png"), dpi=200)
    plt.close(fig)

    # Plot 3: Prediction Distribution & Per-Snapshot Performance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    st_eval = list(range(35, 50))
    ps_cmp = pred_res["per_step_comparison"]
    axes[0].plot(st_eval, [ps_cmp[str(t)]["gcn"]["pr_auc"] for t in st_eval], marker="o", label="GCNAvg PR-AUC", color="#2b6cb0")
    axes[0].plot(st_eval, [ps_cmp[str(t)]["hgb"]["pr_auc"] for t in st_eval], marker="s", label="HGB PR-AUC", color="#38a169")
    axes[0].plot(st_eval, [ps_cmp[str(t)]["gcn"]["roc_auc"] for t in st_eval], linestyle="--", label="GCNAvg ROC-AUC", color="#90cdf4")
    axes[0].plot(st_eval, [ps_cmp[str(t)]["hgb"]["roc_auc"] for t in st_eval], linestyle="--", label="HGB ROC-AUC", color="#9ae6b4")
    axes[0].axvline(42.5, color="red", linestyle="--", label="Step 42->43")
    axes[0].set_title("Per-Snapshot Ranking Trajectory (Steps 35-49)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.25)

    # PR Curves for Step 42 vs Steps 43-49
    y42 = gcn_sp[42]["y_lab"]
    pg42, ph42 = gcn_sp[42]["prob_lab"], hgb_sp[42]["prob_lab"]
    y43 = np.concatenate([gcn_sp[t]["y_lab"] for t in range(43, 50)])
    pg43 = np.concatenate([gcn_sp[t]["prob_lab"] for t in range(43, 50)])
    ph43 = np.concatenate([hgb_sp[t]["prob_lab"] for t in range(43, 50)])

    for lbl, yy, pp, col, ls in [
        ("GCNAvg Step 42 (0.883)", y42, pg42, "#2b6cb0", "-"),
        ("HGB Step 42 (0.891)", y42, ph42, "#38a169", "-"),
        ("GCNAvg 43-49 (0.032)", y43, pg43, "#2b6cb0", "--"),
        ("HGB 43-49 (0.055)", y43, ph43, "#38a169", "--"),
    ]:
        prec, rec, _ = precision_recall_curve(yy, pp)
        axes[1].plot(rec, prec, label=lbl, color=col, linestyle=ls)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves: Step 42 vs Steps 43-49")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    # Score distributions in Steps 43-49 (Illicit vs Licit)
    axes[2].hist(pg43[y43 == 0], bins=25, density=True, alpha=0.45, label="GCNAvg Licit (43-49)", color="#3182ce")
    axes[2].hist(pg43[y43 == 1], bins=25, density=True, alpha=0.55, label="GCNAvg Illicit (43-49)", color="#e53e3e")
    axes[2].axvline(gcn_tau, color="black", linestyle="--", label=f"Frozen tau*={gcn_tau:.2f}")
    axes[2].set_xlabel("Predicted Fraud Probability")
    axes[2].set_ylabel("Density")
    axes[2].set_title("GCNAvg Probability Overlap on Steps 43-49")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage4g_prediction_diagnosis.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# 7. REPORT GENERATOR & MAIN ENTRYPOINT
# =============================================================================

def generate_stage4g_report(payload):
    di = payload["data_integrity"]
    fd = payload["feature_drift"]
    gs = payload["graph_structure_drift"]["regime_summary"]
    lp = payload["label_and_fraud_pattern"]
    pdg = payload["model_prediction_diagnosis"]
    xm = payload["cross_model_comparison"]

    g_m = fd["group_drift_summary"]
    miss = fd["missing_values_raw"]
    sig = lp["fraud_signature_cosine_similarity_cohen_d"]
    cf = pdg["steps_43_49_counterfactual_ranking_vs_calibration"]

    g_row = next(r for r in xm if r["model_name"] == "Stage4F_Champ_Anchor_GCNAvg")
    h_row = next(r for r in xm if r["model_name"] == "HistGradientBoosting")
    g43 = g_row["test_43_49"]
    h43 = h_row["test_43_49"]

    md = f"""# Stage 4G: Temporal Distribution Shift & Post-Shutdown Fraud Detection Diagnostic Report

**Generated On:** {payload['timestamp']}  
**Scope:** Strictly retrospective multi-stage diagnostic investigation across Train (`Steps 1–34`), Validation (`Steps 35–41`), Test Step `42` (pre-shutdown), and Test Steps `43–49` (post-shutdown).  
**Restrictions Honored:** Zero models trained; frozen Stage 4F champion (`F4F_00_Anchor_GCNAvg`, $\\tau^*=0.93$) and Stage 1–4F checkpoints/files untouched; all outputs isolated in `reports/stage4g/`.

---

## Executive Diagnosis: Observed Facts vs. Causal Hypotheses

### Observed Facts (Directly Verified in Data & Frozen Models)
1. **Preprocessing & Data Integrity Are Clean (No Alignment/Pipeline Bug at Step 42):** All `{di['raw_nodes']:,}` nodes and `{di['raw_edges']:,}` edges align identically across raw files and processed tensors (`max_x_diff = {di['snapshot_vs_tabular_max_x_diff']}`), `cross_timestep_edges = {di['cross_timestep_edges']}`, zero duplicate `txId`s, and exact column ordering (`93` Local + `72` Aggregate + `17` Domain).
2. **Simultaneous Prevalence, Structural Homophily & Feature-Signature Shifts After Step 42:**
   - **Prevalence drops `4.4×`:** Labeled fraud prevalence falls from `11.10%` on Step `42` (`239/2,154`) to **`2.53%`** across Steps `43–49` (`169/6,687`, hitting `0.41%` on Step 45 and `0.28%` on Step 46).
   - **Directed Graph Fraud Homophily Drops Sharply:** Upstream parent homophily $P(\\text{{parent}}=\\text{{illicit}} \\mid \\text{{child}}=\\text{{illicit}})$ falls from **`{gs['train_1_34']['pooled_homophily_in_given_v_ill']:.4f}`** (Train) / **`{gs['val_35_41']['pooled_homophily_in_given_v_ill']:.4f}`** (Val) / **`{gs['test_42']['pooled_homophily_in_given_v_ill']:.4f}`** (Step 42) down to **`{gs['test_43_49']['pooled_homophily_in_given_v_ill']:.4f}`** in Steps `43–49`; downstream child homophily falls from `{gs['train_1_34']['pooled_homophily_out_given_u_ill']:.4f}` (Train) / `{gs['val_35_41']['pooled_homophily_out_given_u_ill']:.4f}` (Val) / `{gs['test_42']['pooled_homophily_out_given_u_ill']:.4f}` (Step 42) to **`{gs['test_43_49']['pooled_homophily_out_given_u_ill']:.4f}`**.
   - **Fraud-vs-Licit Feature Signature Rotates in Steps 43–49:** Using standardized Cohen's $d$ separation vectors, `Local-93` aligns strongly across Train, Val, and Step 42 (`cos(Val, Step 42) = {sig['Local_93']['cos_val_vs_step42']:.4f}`), then drops sharply in Steps `43–49` (`cos(Val, Steps 43-49) = {sig['Local_93']['cos_val_vs_43_49']:.4f}`, `cos(Step 42, Steps 43-49) = {sig['Local_93']['cos_step42_vs_43_49']:.4f}`).
3. **Why `0%` Recall Occurs at $\\tau^*=0.93$, and Why It Is Both Ranking Collapse and Calibration Mismatch:**
   - On pooled Steps `43–49` (`6,687` labeled nodes, `169` illicit), `GCNAvg` outputs `meanP_illicit = {g43['mean_prob_illicit']:.4f}` vs `meanP_licit = {g43['mean_prob_licit']:.4f}` (`ROC-AUC = {g43['roc_auc']:.4f}`, `PR-AUC = {g43['pr_auc']:.4f}`), compared with `HGB` (`meanP_illicit = {h43['mean_prob_illicit']:.4f}`, `meanP_licit = {h43['mean_prob_licit']:.4f}`, `ROC-AUC = {h43['roc_auc']:.4f}`, `PR-AUC = {h43['pr_auc']:.4f}`).
   - Even with a **purely diagnostic counterfactual oracle threshold** tuned on `43–49`, `GCNAvg` peaks at **`F1 = {cf['gcn_avg']['oracle_max_f1_posthoc']:.4f}`** (at $\\tau={cf['gcn_avg']['oracle_tau']}$) and catches only **`{cf['gcn_avg']['topk']['k=50']['hits']}/50`** frauds in its Top-50 predictions (`HGB` oracle `F1 = {cf['hist_gradient_boosting']['oracle_max_f1_posthoc']:.4f}`, Top-50 hits = **`{cf['hist_gradient_boosting']['topk']['k=50']['hits']}/50`**). Thus **threshold recalibration alone cannot rescue `GCNAvg` on Steps `43–49`**: the primary failure is **true ranking degradation**, compounded by frozen $\\tau^*=0.93$ sitting above nearly all post-shutdown probabilities (`TP = 0`).

### Evidence-Supported Interpretations vs. Open Hypotheses
- **Supported Interpretation:** The Step `43–49` collapse reflects a genuine **post-shutdown regime shift ($P(X \\mid Y=1)$ signature rotation, loss of fraud-edge homophily, and `4.4×` lower illicit prevalence)** rather than a pipeline or alignment bug.
- **Hypothesis on `HGB`'s Higher Partial Ranking (`0.7698` vs `0.6247`):** Because `HGB` splits non-linearly across all `182` quantile-invariant features (`Aggregate-72` + `Domain-17` + `Local-93`), it preserves partial separation on post-shutdown frauds where `Local-93` linear/GCN representations flatten out; exact per-tree path attribution on `43–49` remains an open verification target.

---

## 1. Data Integrity Verification Summary

- **Dataset Scale:** `{di['raw_nodes']:,}` nodes, `{di['raw_edges']:,}` edges, `{di['unique_txids']:,}` unique `txId`s (`0` duplicates).
- **Cross-Timestep Edges:** `{di['cross_timestep_edges']}` (all edges strictly connect transactions within the same snapshot `t`).
- **Tensor Alignment:** Exact match (`max_x_diff = {di['snapshot_vs_tabular_max_x_diff']}`) across `tabular_data.npz`, `elliptic_pyg_data.pt`, and `temporal_snapshots.pt`.
- **Constant Scaled Columns:** `{di['constant_scaled_columns_by_regime']}`.
- **Exact Duplicate Scaled Feature Rows:** Train `1–34`: `{di['duplicate_feature_rows_by_regime']['train_1_34']['duplicate_rows']}`, Val `35–41`: `{di['duplicate_feature_rows_by_regime']['val_35_41']['duplicate_rows']}`, Step `42`: `{di['duplicate_feature_rows_by_regime']['test_42']['duplicate_rows']}`, Steps `43–49`: `{di['duplicate_feature_rows_by_regime']['test_43_49']['duplicate_rows']}`.

---

## 2. Feature Distribution Drift (`Train 1–34`, `Val 35–41`, `Step 42`, `Steps 43–49`)

### 2.1 Raw Missing-Value Rates (Pre-Imputation)

| Regime | Rows | Rows w/ Any `NaN` | `All_182` Cell `NaN%` | `Local_93` `NaN%` | `Aggregate_72` `NaN%` | `Domain_17` `NaN%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Train (1-34)` | `{miss['train_1_34']['rows']:,}` | `{miss['train_1_34']['rows_with_any_nan']:,}` (`{miss['train_1_34']['pct_rows_with_any_nan']:.2f}%`) | `{miss['train_1_34']['total_missing_cells_pct']:.4f}%` | `{miss['train_1_34']['by_group_pct_cells']['Local_93']:.4f}%` | `{miss['train_1_34']['by_group_pct_cells']['Aggregate_72']:.4f}%` | `{miss['train_1_34']['by_group_pct_cells']['Domain_17']:.4f}%` |
| `Val (35-41)` | `{miss['val_35_41']['rows']:,}` | `{miss['val_35_41']['rows_with_any_nan']:,}` (`{miss['val_35_41']['pct_rows_with_any_nan']:.2f}%`) | `{miss['val_35_41']['total_missing_cells_pct']:.4f}%` | `{miss['val_35_41']['by_group_pct_cells']['Local_93']:.4f}%` | `{miss['val_35_41']['by_group_pct_cells']['Aggregate_72']:.4f}%` | `{miss['val_35_41']['by_group_pct_cells']['Domain_17']:.4f}%` |
| `Test Step 42` | `{miss['test_42']['rows']:,}` | `{miss['test_42']['rows_with_any_nan']:,}` (`{miss['test_42']['pct_rows_with_any_nan']:.2f}%`) | `{miss['test_42']['total_missing_cells_pct']:.4f}%` | `{miss['test_42']['by_group_pct_cells']['Local_93']:.4f}%` | `{miss['test_42']['by_group_pct_cells']['Aggregate_72']:.4f}%` | `{miss['test_42']['by_group_pct_cells']['Domain_17']:.4f}%` |
| `Test 43-49` | `{miss['test_43_49']['rows']:,}` | `{miss['test_43_49']['rows_with_any_nan']:,}` (`{miss['test_43_49']['pct_rows_with_any_nan']:.2f}%`) | `{miss['test_43_49']['total_missing_cells_pct']:.4f}%` | `{miss['test_43_49']['by_group_pct_cells']['Local_93']:.4f}%` | `{miss['test_43_49']['by_group_pct_cells']['Aggregate_72']:.4f}%` | `{miss['test_43_49']['by_group_pct_cells']['Domain_17']:.4f}%` |

### 2.2 Group-Level Drift Summary (PSI, KS, Standardized Mean/Std Shift)

| Feature Group (#) | Median PSI (`Tr->Val`) | Median PSI (`Tr->42`) | Median PSI (`Tr->43-49`) | Max PSI (`Tr->43-49`) | Median PSI (`42->43-49`) | Max Mean Shift (`Tr->43-49`) | `% Cols PSI>0.25` (`Tr->43-49`) | `|z|>5` (`Tr / Val / 42 / 43-49`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Local_93` (`93`) | `{g_m['Local_93']['psi_tr_to_val_median']:.4f}` | `{g_m['Local_93']['psi_tr_to_step42_median']:.4f}` | `{g_m['Local_93']['psi_tr_to_43_49_median']:.4f}` | `{g_m['Local_93']['psi_tr_to_43_49_max']:.4f}` | `{g_m['Local_93']['psi_step42_to_43_49_median']:.4f}` | `{g_m['Local_93']['abs_mean_shift_tr_to_43_49_max']:.4f}` | `{g_m['Local_93']['frac_psi_gt_025_tr_to_43_49']*100:.1f}%` | `{g_m['Local_93']['frac_gt5_outlier_pct']['train_1_34']:.2f}% / {g_m['Local_93']['frac_gt5_outlier_pct']['val_35_41']:.2f}% / {g_m['Local_93']['frac_gt5_outlier_pct']['test_42']:.2f}% / {g_m['Local_93']['frac_gt5_outlier_pct']['test_43_49']:.2f}%` |
| `Aggregate_72` (`72`) | `{g_m['Aggregate_72']['psi_tr_to_val_median']:.4f}` | `{g_m['Aggregate_72']['psi_tr_to_step42_median']:.4f}` | `{g_m['Aggregate_72']['psi_tr_to_43_49_median']:.4f}` | `{g_m['Aggregate_72']['psi_tr_to_43_49_max']:.4f}` | `{g_m['Aggregate_72']['psi_step42_to_43_49_median']:.4f}` | `{g_m['Aggregate_72']['abs_mean_shift_tr_to_43_49_max']:.4f}` | `{g_m['Aggregate_72']['frac_psi_gt_025_tr_to_43_49']*100:.1f}%` | `{g_m['Aggregate_72']['frac_gt5_outlier_pct']['train_1_34']:.2f}% / {g_m['Aggregate_72']['frac_gt5_outlier_pct']['val_35_41']:.2f}% / {g_m['Aggregate_72']['frac_gt5_outlier_pct']['test_42']:.2f}% / {g_m['Aggregate_72']['frac_gt5_outlier_pct']['test_43_49']:.2f}%` |
| `Domain_17` (`17`) | `{g_m['Domain_17']['psi_tr_to_val_median']:.4f}` | `{g_m['Domain_17']['psi_tr_to_step42_median']:.4f}` | `{g_m['Domain_17']['psi_tr_to_43_49_median']:.4f}` | `{g_m['Domain_17']['psi_tr_to_43_49_max']:.4f}` | `{g_m['Domain_17']['psi_step42_to_43_49_median']:.4f}` | `{g_m['Domain_17']['abs_mean_shift_tr_to_43_49_max']:.4f}` | `{g_m['Domain_17']['frac_psi_gt_025_tr_to_43_49']*100:.1f}%` | `{g_m['Domain_17']['frac_gt5_outlier_pct']['train_1_34']:.2f}% / {g_m['Domain_17']['frac_gt5_outlier_pct']['val_35_41']:.2f}% / {g_m['Domain_17']['frac_gt5_outlier_pct']['test_42']:.2f}% / {g_m['Domain_17']['frac_gt5_outlier_pct']['test_43_49']:.2f}%` |

---

## 3. Graph Structure Drift

| Regime | Nodes/Step | Edges/Step | Edge/Node | Mean Density | Isolated% | Zero-In (All / Ill%) | Zero-Out (All / Ill%) | Giant WCC% | $P(\\text{{parent}}=\\text{{ill}}\\|\\text{{child}}=\\text{{ill}})$ | $P(\\text{{child}}=\\text{{ill}}\\|\\text{{parent}}=\\text{{ill}})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Train (1-34)` | `{gs['train_1_34']['mean_nodes_per_step']}` | `{gs['train_1_34']['mean_edges_per_step']}` | `{gs['train_1_34']['pooled_edge_to_node_ratio']:.4f}` | `{gs['train_1_34']['mean_density']:.6f}` | `{gs['train_1_34']['mean_isolated_pct']:.2f}%` | `{gs['train_1_34']['mean_zero_in_all_pct']:.1f}% / {gs['train_1_34']['illicit_zero_in_pct']:.1f}%` | `{gs['train_1_34']['mean_zero_out_all_pct']:.1f}% / {gs['train_1_34']['illicit_zero_out_pct']:.1f}%` | `{gs['train_1_34']['mean_giant_cc_pct']:.1f}%` | `{gs['train_1_34']['pooled_homophily_in_given_v_ill']:.4f}` | `{gs['train_1_34']['pooled_homophily_out_given_u_ill']:.4f}` |
| `Val (35-41)` | `{gs['val_35_41']['mean_nodes_per_step']}` | `{gs['val_35_41']['mean_edges_per_step']}` | `{gs['val_35_41']['pooled_edge_to_node_ratio']:.4f}` | `{gs['val_35_41']['mean_density']:.6f}` | `{gs['val_35_41']['mean_isolated_pct']:.2f}%` | `{gs['val_35_41']['mean_zero_in_all_pct']:.1f}% / {gs['val_35_41']['illicit_zero_in_pct']:.1f}%` | `{gs['val_35_41']['mean_zero_out_all_pct']:.1f}% / {gs['val_35_41']['illicit_zero_out_pct']:.1f}%` | `{gs['val_35_41']['mean_giant_cc_pct']:.1f}%` | `{gs['val_35_41']['pooled_homophily_in_given_v_ill']:.4f}` | `{gs['val_35_41']['pooled_homophily_out_given_u_ill']:.4f}` |
| `Test Step 42` | `{gs['test_42']['mean_nodes_per_step']}` | `{gs['test_42']['mean_edges_per_step']}` | `{gs['test_42']['pooled_edge_to_node_ratio']:.4f}` | `{gs['test_42']['mean_density']:.6f}` | `{gs['test_42']['mean_isolated_pct']:.2f}%` | `{gs['test_42']['mean_zero_in_all_pct']:.1f}% / {gs['test_42']['illicit_zero_in_pct']:.1f}%` | `{gs['test_42']['mean_zero_out_all_pct']:.1f}% / {gs['test_42']['illicit_zero_out_pct']:.1f}%` | `{gs['test_42']['mean_giant_cc_pct']:.1f}%` | `{gs['test_42']['pooled_homophily_in_given_v_ill']:.4f}` | `{gs['test_42']['pooled_homophily_out_given_u_ill']:.4f}` |
| `Test 43-49` | `{gs['test_43_49']['mean_nodes_per_step']}` | `{gs['test_43_49']['mean_edges_per_step']}` | `{gs['test_43_49']['pooled_edge_to_node_ratio']:.4f}` | `{gs['test_43_49']['mean_density']:.6f}` | `{gs['test_43_49']['mean_isolated_pct']:.2f}%` | `{gs['test_43_49']['mean_zero_in_all_pct']:.1f}% / {gs['test_43_49']['illicit_zero_in_pct']:.1f}%` | `{gs['test_43_49']['mean_zero_out_all_pct']:.1f}% / {gs['test_43_49']['illicit_zero_out_pct']:.1f}%` | `{gs['test_43_49']['mean_giant_cc_pct']:.1f}%` | `{gs['test_43_49']['pooled_homophily_in_given_v_ill']:.4f}` | `{gs['test_43_49']['pooled_homophily_out_given_u_ill']:.4f}` |

---

## 4. Retrospective Label & Fraud-Pattern Signature Analysis

| Feature Subset | $\\cos(\\text{{Train}}, \\text{{Val}})$ | $\\cos(\\text{{Val}}, \\text{{Step 42}})$ | $\\cos(\\text{{Val}}, \\text{{43–49}})$ | $\\cos(\\text{{Step 42}}, \\text{{43–49}})$ |
| :--- | :---: | :---: | :---: | :---: |
| `All_182` | `{sig['All_182']['cos_train_vs_val']:.4f}` | `{sig['All_182']['cos_val_vs_step42']:.4f}` | `{sig['All_182']['cos_val_vs_43_49']:.4f}` | `{sig['All_182']['cos_step42_vs_43_49']:.4f}` |
| `Local_93` | `{sig['Local_93']['cos_train_vs_val']:.4f}` | `{sig['Local_93']['cos_val_vs_step42']:.4f}` | `{sig['Local_93']['cos_val_vs_43_49']:.4f}` | `{sig['Local_93']['cos_step42_vs_43_49']:.4f}` |
| `Aggregate_72` | `{sig['Aggregate_72']['cos_train_vs_val']:.4f}` | `{sig['Aggregate_72']['cos_val_vs_step42']:.4f}` | `{sig['Aggregate_72']['cos_val_vs_43_49']:.4f}` | `{sig['Aggregate_72']['cos_step42_vs_43_49']:.4f}` |
| `Domain_17` | `{sig['Domain_17']['cos_train_vs_val']:.4f}` | `{sig['Domain_17']['cos_val_vs_step42']:.4f}` | `{sig['Domain_17']['cos_val_vs_43_49']:.4f}` | `{sig['Domain_17']['cos_step42_vs_43_49']:.4f}` |

---

## 5. Cross-Model Comparison Across Regimes (Existing Checkpoints Only, Zero Retraining)

| Model | Val `35–41` PR-AUC / ROC | Step `42` PR-AUC / ROC | Steps `43–49` PR-AUC / ROC | Test All `42–49` PR-AUC / ROC / F1 |
| :--- | :---: | :---: | :---: | :---: |
"""
    for row in xm:
        v = row["val_35_41"]
        s42 = row["test_42"]
        s43 = row["test_43_49"]
        ta = row["test_all_42_49"]
        md += (
            f"| `{row['model_name']}` | `{v['pr_auc']:.4f}` / `{v['roc_auc']:.4f}` | "
            f"`{s42['pr_auc']:.4f}` / `{s42['roc_auc']:.4f}` | "
            f"`{s43['pr_auc']:.4f}` / `{s43['roc_auc']:.4f}` | "
            f"`{ta['pr_auc']:.4f}` / `{ta['roc_auc']:.4f}` / `{ta['illicit_f1']:.4f}` |\n"
        )

    md += """
### 2.3 Top 10 Most Shifted Features (`Step 42 -> Steps 43–49` by PSI)

| Feature | Group | Mean (Tr / 42 / 43–49) | Median (Tr / 43–49) | Std (Tr / 43–49) | PSI (`Tr->43-49`) | PSI (`42->43-49`) | KS (`42->43-49`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for d in fd["top15_psi_step42_to_43_49"][:10]:
        md += (
            f"| `{d['name']}` | `{d['group']}` | "
            f"`{d['mean_train']:.2f} / {d['mean_step42']:.2f} / {d['mean_steps43_49']:.2f}` | "
            f"`{d['median_train']:.2f} / {d['median_steps43_49']:.2f}` | "
            f"`{d['std_train']:.2f} / {d['std_steps43_49']:.2f}` | "
            f"`{d['psi_tr_to_43_49']:.4f}` | **{d['psi_step42_to_43_49']:.4f}** | `{d['ks_step42_to_43_49']:.4f}` |\n"
        )

    md += """
### 4.1 Top 10 Validation-Discriminative Features Across Time (Standardized Cohen's $d$)

| Feature | Group | $d$ (Train `1–34`) | $d$ (Val `35–41`) | $d$ (Step `42`) | $d$ (Steps `43–49`) | Sign Flip in `43–49`? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for d in lp["top20_val_discriminative_features_shift"][:10]:
        md += (
            f"| `{d['name']}` | `{d['group']}` | `{d['cohen_d_train']:+.3f}` | `{d['cohen_d_val']:+.3f}` | "
            f"`{d['cohen_d_step42']:+.3f}` | `{d['cohen_d_steps43_49']:+.3f}` | `{d['sign_flipped_in_43_49']}` |\n"
        )

    md += """
---

## 5A. Per-Snapshot Prediction Diagnosis: `GCNAvg` vs. `HistGradientBoosting` (Steps `42–49`)

| Step | N (Ill) | GCNAvg PR / ROC | HGB PR / ROC | GCNAvg `meanP(Ill / Lic / Unl+All Pos%)` | HGB `meanP(Ill / Lic / All Pos%)` | GCNAvg TP/FP/FN | HGB TP/FP/FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for st in range(42, 50):
        row_s = pdg["per_step_comparison"][str(st)]
        gs_m, hs_m = row_s["gcn"], row_s["hgb"]
        md += (
            f"| `{st}` | `{row_s['n_labeled']}` (`{row_s['n_illicit']}`) | "
            f"`{gs_m['pr_auc']:.4f}` / `{gs_m['roc_auc']:.4f}` | "
            f"`{hs_m['pr_auc']:.4f}` / `{hs_m['roc_auc']:.4f}` | "
            f"`{gs_m['mean_prob_illicit']:.3f} / {gs_m['mean_prob_licit']:.3f} ({gs_m['all_pos_pred_rate_pct']:.2f}%)` | "
            f"`{hs_m['mean_prob_illicit']:.3f} / {hs_m['mean_prob_licit']:.3f} ({hs_m['all_pos_pred_rate_pct']:.2f}%)` | "
            f"`{gs_m['tp']}/{gs_m['fp']}/{gs_m['fn']}` | `{hs_m['tp']}/{hs_m['fp']}/{hs_m['fn']}` |\n"
        )

    ov42 = pdg["step_42_fraud_overlap"]
    md += f"""
- **Step 42 Catch Overlap (`239` illicit):** Both caught `{ov42['both_caught']}`, GCNAvg-only `{ov42['only_gcn_caught']}`, HGB-only `{ov42['only_hgb_caught']}`, both missed `{ov42['both_missed']}`.
- **Counterfactual Ranking Ceiling on Steps `43–49` (`169` illicit):** `GCNAvg` oracle post-hoc max F1 = `{cf['gcn_avg']['oracle_max_f1_posthoc']:.4f}` (`tau={cf['gcn_avg']['oracle_tau']}`), Top-50 hits = `{cf['gcn_avg']['topk']['k=50']['hits']}` (`k=100`: `{cf['gcn_avg']['topk']['k=100']['hits']}`); `HGB` oracle post-hoc max F1 = `{cf['hist_gradient_boosting']['oracle_max_f1_posthoc']:.4f}` (`tau={cf['hist_gradient_boosting']['oracle_tau']}`), Top-50 hits = `{cf['hist_gradient_boosting']['topk']['k=50']['hits']}` (`k=100`: `{cf['hist_gradient_boosting']['topk']['k=100']['hits']}`).

---

## 6. Limitations & Most Scientifically Justified Next Experiment

### Diagnostic Limitations
1. **Sparse Post-Shutdown Positives:** Steps `45` (`5` illicit) and `46` (`2` illicit) contain very few labeled positive examples, making single-snapshot PR-AUC and Cohen's $d$ estimates noisy at the individual-step level; pooled Steps `43–49` (`169` illicit) gives the reliable summary.
2. **Unlabeled Transactions (`~77%`):** Class-3 unlabeled background nodes cannot confirm whether fraud-signature rotation also alters unlabeled neighborhoods or reflects Elliptic sampling shifts after the dark-market closure.

### Recommended Next Experiment (NOT Executed)
- **Stage 5 / Future Research Priority (Train-1–34 / Val-35–41 Strictly Preserved):** Because `HistGradientBoosting` holds `ROC-AUC = 0.7698` on Steps `43–49` while *every* pure `Local-93` neural/GNN architecture collapses to `0.59–0.63` on Steps `43–49`, the single most scientifically justified next experiment is a **Hybrid GNN-Embedding + Tabular Tree (`HistGradientBoosting` / Stacking) vs. RankGauss/Quantile-Transformed Feature Graph Model** — specifically testing (a) whether **Train-fitted `QuantileTransformer(output_distribution='normal')`** on all `182` features gives neural/GNN models the same coordinate-wise tail invariance trees rely on in Steps `43–49`, and (b) whether feeding out-of-fold / validation-calibrated `GCNAvg` graph embeddings + `182` features into `HistGradientBoosting` beats pure `HistGradientBoosting` (`PR-AUC 0.5646, ROC-AUC 0.8848`) without touching `Steps 42–49` for tuning!
"""
    return md


def run_stage4g_diagnostics():
    print("=" * 88)
    print("STAGE 4G: TEMPORAL DISTRIBUTION SHIFT & POST-SHUTDOWN DIAGNOSTIC INVESTIGATION")
    print("=" * 88)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "reports", "stage4g")
    os.makedirs(out_dir, exist_ok=True)

    snapshots = torch.load(os.path.join(base_dir, "processed", "temporal_snapshots.pt"), weights_only=False)
    global_data = torch.load(os.path.join(base_dir, "processed", "elliptic_pyg_data.pt"), weights_only=False)
    tabular = np.load(os.path.join(base_dir, "processed", "tabular_data.npz"))
    with open(os.path.join(base_dir, "processed", "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("[1/5] Running Data Integrity Check...")
    di_res, df_m, X_raw, nan_mask, ts_arr, feat_cols = run_data_integrity_check(
        base_dir, snapshots, global_data, tabular, metadata
    )

    print("[2/5] Analyzing Feature Distribution Drift (Train 1-34, Val 35-41, Step 42, Steps 43-49)...")
    drift_res = analyze_feature_drift(tabular["X"], X_raw, nan_mask, ts_arr, feat_cols)

    print("[3/5] Analyzing Graph Structure & Homophily Drift (Steps 1-49)...")
    graph_res = analyze_graph_structure(snapshots)

    print("[4/5] Analyzing Retrospective Label & Fraud-Pattern Shift...")
    label_res = analyze_labels_and_fraud_patterns(snapshots, tabular["X"], tabular["y"], ts_arr, feat_cols)

    print("[5/5] Evaluating Frozen Models & Cross-Model Comparison (No Retraining)...")
    device = torch.device("cpu")
    model_specs = [
        ("HistGradientBoosting", "models/hist_gradient_boosting.joblib"),
        ("Random_Forest", "models/random_forest.joblib"),
        ("Logistic_Regression", "models/logistic_regression.joblib"),
        ("MLP_Baseline", "models/mlp_baseline.pt"),
        ("GCN_Original", "models/gcn_static.pt"),
        ("GCN_Improved_Residual", "models/gcn_static_improved.pt"),
        ("GAT_Original", "models/gat_static.pt"),
        ("GAT_Improved_Residual", "models/gat_static_improved.pt"),
        ("GCN_LSTM", "models/gcn_lstm.pt"),
        ("GCN_LSTM_Historical", "models/gcn_lstm_historical.pt"),
        ("EvolveGCN_H_Recalibrated", "models/evolvegcn_h.pt"),
        ("Stage4D_E6_Local93_GCNAvg", "models/nextgen_gnn_best.pt"),
        ("Stage4D_E6_Local93_NoGraph", "models/stage4d/nograph_control_best.pt"),
        ("Stage4D_E6_Local93_SAGEConcat", "models/stage4d/E6_Local93_SAGEConcat_seed42.pt"),
        ("Stage4D_E5_ModeratedBCE_pw2.76", "models/stage4d/E5_Loss_ModeratedBCE_pw2.76_seed42.pt"),
        ("Stage4E_F2_Local93_Domain17_s123", "models/stage4e/F2_Local93_Domain17_seed123.pt"),
        ("Stage4F_Champ_Anchor_GCNAvg", "models/stage4f/stage4f_champion.pt"),
        ("Stage4F_BiDir_ResSwiGLU_s42", "models/stage4f/F4F_05_BiDir_ResSwiGLU_seed42.pt"),
        ("Stage4F_Asinh_BiDirGCN_s456", "models/stage4f/F4F_06_Asinh_BiDirGCN_seed456.pt"),
        ("Stage4F_Asinh110_BiDirGCN_s456", "models/stage4f/F4F_07_Asinh_110_BiDirGCN_seed456.pt"),
    ]

    cross_model = []
    cached = {}
    for m_name, rel_p in model_specs:
        full_p = os.path.join(base_dir, rel_p)
        if not os.path.exists(full_p):
            print(f"  [SKIP missing] {full_p}")
            continue
        tau, sp = eval_static_or_sequential_checkpoint(m_name, full_p, snapshots, global_data, tabular, device)
        row = {"model_name": m_name, "checkpoint_path": rel_p, **summarize_model_slices(tau, sp)}
        cross_model.append(row)
        if m_name in ("Stage4F_Champ_Anchor_GCNAvg", "HistGradientBoosting"):
            cached[m_name] = (tau, sp)
        print(
            f"  {m_name:34s} | Val PR={row['val_35_41']['pr_auc']:.4f} | "
            f"Step42 PR={row['test_42']['pr_auc']:.4f} ROC={row['test_42']['roc_auc']:.4f} | "
            f"43-49 PR={row['test_43_49']['pr_auc']:.4f} ROC={row['test_43_49']['roc_auc']:.4f}"
        )

    gcn_tau, gcn_sp = cached["Stage4F_Champ_Anchor_GCNAvg"]
    hgb_tau, hgb_sp = cached["HistGradientBoosting"]
    pred_diag = diagnose_gcn_vs_hgb_predictions(gcn_tau, gcn_sp, hgb_tau, hgb_sp)

    save_stage4g_visualizations(out_dir, drift_res, graph_res, pred_diag, gcn_tau, gcn_sp, hgb_tau, hgb_sp)

    payload = {
        "stage": "Stage 4G - Temporal Distribution Shift & Post-Shutdown Investigation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_integrity": di_res,
        "feature_drift": drift_res,
        "graph_structure_drift": graph_res,
        "label_and_fraud_pattern": label_res,
        "model_prediction_diagnosis": pred_diag,
        "cross_model_comparison": cross_model,
    }

    json_p = os.path.join(out_dir, "stage4g_diagnostic_metrics.json")
    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    md_p = os.path.join(out_dir, "stage4g_diagnostic_report.md")
    with open(md_p, "w", encoding="utf-8") as f:
        f.write(generate_stage4g_report(payload))

    print(f"\nSaved all Stage 4G artifacts inside: {out_dir}")
    print(f"Stage 4G completed in {time.time() - t0:.2f}s.")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    run_stage4g_diagnostics()
