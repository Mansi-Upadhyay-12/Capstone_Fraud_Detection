"""
Stage 4C: EvolveGCN-H (Evolving Graph Convolutional Network) for Elliptic++ Fraud Detection
Controlled Comparison: Directed vs. Undirected Graphs & Residual Skip Connections
Capstone Project: Temporal Graph Neural Networks with Real-Time Explainable AI
for Online Payment Fraud Ring Detection
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, to_undirected
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 1. NORMALIZED SPARSE ADJACENCY UTILITY (PER-SNAPSHOT, ZERO LEAKAGE)
# =============================================================================

def build_normalized_sparse_adj(edge_index, num_nodes, use_undirected=False, device=None):
    """
    Constructs symmetrically normalized sparse adjacency A_hat = D^{-1/2} (A + I) D^{-1/2}
    for a single isolated temporal snapshot.
    """
    ei = to_undirected(edge_index, num_nodes=num_nodes) if use_undirected else edge_index
    ei, _ = add_self_loops(ei, num_nodes=num_nodes)

    row, col = ei[0], ei[1]
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        ei, edge_weight, size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()
    if device is not None:
        adj = adj.to(device)
    return adj


# =============================================================================
# 2. EVOLVEGCN-H ARCHITECTURE (Pareja et al., AAAI 2020)
# =============================================================================

class TopKSummarizer(nn.Module):
    """
    Compresses a variable-size snapshot node matrix H_t in R^{N_t x d_in}
    into a fixed-size summary matrix Z_t in R^{k x d_in} (k = d_out) for GRU weight evolution.
    """
    def __init__(self, in_channels, k):
        super().__init__()
        self.k = k
        self.proj = nn.Parameter(torch.empty(in_channels, 1))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, h):
        num_nodes = h.shape[0]
        norm = torch.norm(self.proj, p=2) + 1e-8
        scores = (h @ self.proj).squeeze(-1) / norm

        if num_nodes >= self.k:
            top_vals, top_idx = torch.topk(scores, k=self.k, sorted=True)
        else:
            sorted_vals, sorted_idx = torch.sort(scores, descending=True)
            repeat_factor = (self.k + num_nodes - 1) // num_nodes
            top_vals = sorted_vals.repeat(repeat_factor)[: self.k]
            top_idx = sorted_idx.repeat(repeat_factor)[: self.k]

        gates = torch.tanh(top_vals).unsqueeze(-1)
        return h[top_idx] * gates  # [k, in_channels]


class EvolveGCNHLayer(nn.Module):
    """
    Single EvolveGCN-H layer:
      1. Summarizes input node representations H_t -> Z_t in R^{d_out x d_in}.
      2. Evolves GCN weight matrix W_t^T = GRUCell(Z_t, W_{t-1}^T).
      3. Applies dynamic graph convolution: A_hat_t @ (H_t @ W_t).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.summarizer = TopKSummarizer(in_channels, k=out_channels)
        self.gru_cell = nn.GRUCell(input_size=in_channels, hidden_size=in_channels)

        self.W_init = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W_init)

    def initial_weight(self):
        return self.W_init

    def forward(self, h, adj_norm, w_prev):
        z_t = self.summarizer(h)  # [out_channels, in_channels]
        w_t_T = self.gru_cell(z_t, w_prev.t())  # [out_channels, in_channels]
        w_t = w_t_T.t()  # [in_channels, out_channels]
        support = h @ w_t  # [N_t, out_channels]
        out = torch.sparse.mm(adj_norm, support)
        return out, w_t


class FraudEvolveGCNH(nn.Module):
    """
    2-Layer EvolveGCN-H for Temporal Financial Fraud Detection:
    - Evolves spatial graph convolution filters W_t^{(1)}, W_t^{(2)} across snapshots.
    - Eliminates Stage 4B's additive macro-bias vector broadcast.
    - Supports optional residual skip highways and LayerNorm stabilization (Stage 4A insight).
    """
    def __init__(
        self,
        in_channels=182,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        use_residual=False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate

        self.egcn1 = EvolveGCNHLayer(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.skip1 = nn.Linear(in_channels, hidden_channels) if use_residual else None

        self.egcn2 = EvolveGCNHLayer(hidden_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)
        self.skip2 = nn.Linear(hidden_channels, out_channels) if use_residual else None

        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_weights_state(self):
        return self.egcn1.initial_weight(), self.egcn2.initial_weight()

    def forward_step(self, x, adj_norm, w1_prev, w2_prev):
        h1, w1_next = self.egcn1(x, adj_norm, w1_prev)
        if self.skip1 is not None:
            h1 = h1 + self.skip1(x)
        h1 = self.ln1(h1)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

        h2, w2_next = self.egcn2(h1, adj_norm, w2_prev)
        if self.skip2 is not None:
            h2 = h2 + self.skip2(h1)
        h2 = self.ln2(h2)
        h2 = F.relu(h2)
        h2 = F.dropout(h2, p=self.dropout_rate * 0.7, training=self.training)

        logits = self.head(h2).squeeze(-1)
        return logits, w1_next, w2_next


# =============================================================================
# 3. EVALUATION & THRESHOLD UTILITIES (ALIGNED WITH STAGES 3, 4A, 4B)
# =============================================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Computes full classification and ranking metrics aligned with Stages 3–4B."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    illicit_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    illicit_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    illicit_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    licit_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    licit_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    licit_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    try:
        roc_auc = float(round(roc_auc_score(y_true, y_prob), 4)) if len(np.unique(y_true)) > 1 else None
    except Exception:
        roc_auc = None

    try:
        pr_auc = float(round(average_precision_score(y_true, y_prob), 4)) if (y_true == 1).sum() > 0 else None
    except Exception:
        pr_auc = None

    return {
        "threshold_used": float(round(threshold, 4)),
        "accuracy": float(round(acc, 4)),
        "illicit_precision": float(round(illicit_prec, 4)),
        "illicit_recall": float(round(illicit_rec, 4)),
        "illicit_f1": float(round(illicit_f1, 4)),
        "licit_precision": float(round(licit_prec, 4)),
        "licit_recall": float(round(licit_rec, 4)),
        "licit_f1": float(round(licit_f1, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "micro_f1": float(round(micro_f1, 4)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {
            "true_negatives_licit": int(tn),
            "false_positives_false_alarm": int(fp),
            "false_negatives_missed_fraud": int(fn),
            "true_positives_caught_fraud": int(tp),
        },
        "total_samples": int(len(y_true)),
        "illicit_count": int((y_true == 1).sum()),
        "licit_count": int((y_true == 0).sum()),
    }


def find_best_threshold(y_val, y_val_prob):
    """Finds threshold in [0.10, 0.90] maximizing Validation Illicit F1-Score."""
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (y_val_prob >= t).astype(int)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
    return float(round(best_thresh, 4)), float(round(best_f1, 4))


def find_recall_oriented_threshold(y_val, y_val_prob, min_precision=0.30):
    """Finds validation threshold maximizing Recall subject to Precision >= min_precision."""
    best_thresh = None
    best_recall = -1.0
    for t in np.linspace(0.10, 0.90, 81):
        preds = (y_val_prob >= t).astype(int)
        prec = precision_score(y_val, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_val, preds, pos_label=1, zero_division=0)
        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_thresh = t
    if best_thresh is None:
        best_thresh, _ = find_best_threshold(y_val, y_val_prob)
    return float(round(best_thresh, 4))


# =============================================================================
# 4. SINGLE CONFIGURATION TRAINING & ZERO-LEAKAGE EVALUATION ENGINE
# =============================================================================

def rollout_val_split(model, snapshots_dev, adjs_dev):
    """
    Deterministic validation rollout (Steps 1-34 warm-up -> Steps 35-41 validation).
    Never accesses Test snapshots (42-49).
    """
    model.eval()
    val_probs = []
    with torch.no_grad():
        w1, w2 = model.init_weights_state()
        for t in range(34):
            s = snapshots_dev[t]
            _, w1, w2 = model.forward_step(s.x, adjs_dev[t], w1, w2)

        for t in range(34, 41):
            s = snapshots_dev[t]
            logits, w1, w2 = model.forward_step(s.x, adjs_dev[t], w1, w2)
            probs = torch.sigmoid(logits[s.labeled_mask]).cpu().numpy()
            val_probs.extend(probs)
    return np.array(val_probs)


def rollout_test_split(model, snapshots_dev, adjs_dev):
    """
    Post-selection deterministic test rollout (Steps 1-41 warm-up -> Steps 42-49 test).
    Called strictly AFTER architecture selection and threshold tuning.
    """
    model.eval()
    test_probs = []
    test_step_details = {}

    with torch.no_grad():
        w1, w2 = model.init_weights_state()
        for t in range(41):
            s = snapshots_dev[t]
            _, w1, w2 = model.forward_step(s.x, adjs_dev[t], w1, w2)

        t_inf0 = time.perf_counter()
        for t in range(41, 49):
            s = snapshots_dev[t]
            logits, w1, w2 = model.forward_step(s.x, adjs_dev[t], w1, w2)
            all_probs_step = torch.sigmoid(logits).cpu().numpy()
            lab_probs_step = all_probs_step[s.labeled_mask.cpu().numpy()]
            lab_y_step = s.y[s.labeled_mask].cpu().numpy()
            test_probs.extend(lab_probs_step)
            test_step_details[int(s.timestep)] = {
                "probs": lab_probs_step,
                "y": lab_y_step,
                "labeled_mean_prob": float(round(float(lab_probs_step.mean()), 4)),
                "labeled_median_prob": float(round(float(np.median(lab_probs_step)), 4)),
                "all_nodes_count": int(s.num_nodes),
                "labeled_nodes_count": int(len(lab_y_step)),
                "illicit_count": int((lab_y_step == 1).sum()),
            }
        inf_time = time.perf_counter() - t_inf0

    return np.array(test_probs), test_step_details, inf_time


def train_and_validate_config(
    config_name,
    use_undirected,
    use_residual,
    snapshots_dev,
    adjs_dev,
    y_val_all,
    pos_weight,
    device,
    max_epochs=60,
    lr=0.002,
    patience=15,
    grad_clip=2.0,
    smoke_test=False,
):
    print(f"\n[{config_name}] Undirected={use_undirected} | Residual={use_residual} | MaxEpochs={max_epochs}")
    torch.manual_seed(42)
    np.random.seed(42)

    in_channels = snapshots_dev[0].x.shape[1]
    model = FraudEvolveGCNH(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        use_residual=use_residual,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_val_prauc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    t_train_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        w1, w2 = model.init_weights_state()
        total_train_loss = 0.0
        num_loss_steps = 0

        for t in range(34):
            s = snapshots_dev[t]
            logits_t, w1, w2 = model.forward_step(s.x, adjs_dev[t], w1, w2)
            if s.labeled_mask.sum() > 0:
                loss_t = criterion(logits_t[s.labeled_mask], s.y[s.labeled_mask].float())
                total_train_loss = total_train_loss + loss_t
                num_loss_steps += 1

        avg_train_loss = total_train_loss / max(num_loss_steps, 1)
        avg_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # Validation evaluation (Steps 35-41)
        y_val_prob_ep = rollout_val_split(model, snapshots_dev, adjs_dev)

        try:
            val_prauc = float(average_precision_score(y_val_all, y_val_prob_ep))
        except Exception:
            val_prauc = 0.0

        curr_thresh, curr_val_f1 = find_best_threshold(y_val_all, y_val_prob_ep)
        scheduler.step(val_prauc)

        if val_prauc > best_val_prauc or best_state is None:
            best_val_prauc = val_prauc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
            print(
                f"  Epoch {epoch:02d}/{max_epochs} | Train Loss: {avg_train_loss.item():.4f} | "
                f"Val PR-AUC: {val_prauc:.4f} | Val F1: {curr_val_f1:.4f} (tau={curr_thresh:.2f})"
            )

        if patience_counter >= patience and not smoke_test:
            print(
                f"  Early stopping triggered at epoch {epoch} "
                f"(Best Epoch: {best_epoch}, Best Val PR-AUC: {best_val_prauc:.4f})"
            )
            break

    train_time = time.time() - t_train_start

    # Restore best Validation PR-AUC checkpoint and select thresholds strictly on Validation
    model.load_state_dict(best_state)
    y_val_prob_best = rollout_val_split(model, snapshots_dev, adjs_dev)

    tau_f1, _ = find_best_threshold(y_val_all, y_val_prob_best)
    tau_rec30 = find_recall_oriented_threshold(y_val_all, y_val_prob_best, min_precision=0.30)
    val_metrics = compute_metrics(y_val_all, y_val_prob_best, threshold=tau_f1)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"  >>> [{config_name}] Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
        f"Val F1: {val_metrics['illicit_f1']:.4f} (tau*={tau_f1:.2f}, tau_rec30={tau_rec30:.2f})"
    )

    return {
        "config_name": config_name,
        "use_undirected": bool(use_undirected),
        "use_residual": bool(use_residual),
        "total_parameters": int(param_count),
        "best_epoch": int(best_epoch),
        "training_time_seconds": round(train_time, 3),
        "optimal_threshold_f1": float(tau_f1),
        "threshold_recall_constrained": float(tau_rec30),
        "validation_metrics": val_metrics,
        "best_state": best_state,
    }


def evaluate_config_on_test(cfg_res, snapshots_dev, adjs_dev, y_test_all, device):
    """Evaluates a frozen, already-selected configuration once on Test Steps 42-49."""
    in_channels = snapshots_dev[0].x.shape[1]
    model = FraudEvolveGCNH(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        use_residual=cfg_res["use_residual"],
    ).to(device)
    model.load_state_dict(cfg_res["best_state"])

    y_test_prob_best, step_details, inf_time = rollout_test_split(model, snapshots_dev, adjs_dev)

    tau_f1 = cfg_res["optimal_threshold_f1"]
    tau_rec30 = cfg_res["threshold_recall_constrained"]

    test_metrics_opt = compute_metrics(y_test_all, y_test_prob_best, threshold=tau_f1)
    test_metrics_rec30 = compute_metrics(y_test_all, y_test_prob_best, threshold=tau_rec30)
    test_metrics_def = compute_metrics(y_test_all, y_test_prob_best, threshold=0.5)

    per_snapshot_test = {}
    for step, info in step_details.items():
        m_step = compute_metrics(info["y"], info["probs"], threshold=tau_f1)
        per_snapshot_test[str(step)] = {
            "all_nodes": info["all_nodes_count"],
            "labeled_nodes": info["labeled_nodes_count"],
            "illicit_count": info["illicit_count"],
            "labeled_mean_prob": info["labeled_mean_prob"],
            "labeled_median_prob": info["labeled_median_prob"],
            "illicit_precision": m_step["illicit_precision"],
            "illicit_recall": m_step["illicit_recall"],
            "illicit_f1": m_step["illicit_f1"],
            "pr_auc": m_step["pr_auc"],
            "roc_auc": m_step["roc_auc"],
            "tp": m_step["confusion_matrix"]["true_positives_caught_fraud"],
            "fp": m_step["confusion_matrix"]["false_positives_false_alarm"],
            "fn": m_step["confusion_matrix"]["false_negatives_missed_fraud"],
        }

    cfg_res["inference_time_seconds"] = round(inf_time, 4)
    cfg_res["test_metrics_optimal_threshold"] = test_metrics_opt
    cfg_res["test_metrics_recall_constrained"] = test_metrics_rec30
    cfg_res["test_metrics_default_threshold"] = test_metrics_def
    cfg_res["per_snapshot_test_metrics"] = per_snapshot_test
    cfg_res["test_probs"] = y_test_prob_best

    print(
        f"  >>> [{cfg_res['config_name']}] Test (tau*={tau_f1:.2f}) Prec: {test_metrics_opt['illicit_precision']:.4f}, "
        f"Rec: {test_metrics_opt['illicit_recall']:.4f}, F1: {test_metrics_opt['illicit_f1']:.4f}, "
        f"PR-AUC: {test_metrics_opt['pr_auc']:.4f}, ROC-AUC: {test_metrics_opt['roc_auc']:.4f}"
    )
    return cfg_res


# =============================================================================
# 5. BENCHMARK LOADER & PLOTTING / REPORTING
# =============================================================================

def load_comparison_baselines(reports_dir, results_dir):
    """
    Loads evaluation metrics for HistGradientBoosting, Improved Residual GCN,
    Temporal GCN-LSTM, and Historical GCN-LSTM from verified workspace artifacts.
    Uses results/graphs/model_comparison.json as primary source for saved checkpoint
    evaluations (protecting against overwritten 2-epoch smoke-test Stage 4B JSONs).
    """
    benchmarks = {}

    comp_json = os.path.join(results_dir, "graphs", "model_comparison.json")
    if os.path.exists(comp_json):
        with open(comp_json, "r", encoding="utf-8") as f:
            records = json.load(f)
        by_key = {r["Model_Key"]: r for r in records}
        mapping = {
            "HistGradientBoosting": "HistGradientBoosting",
            "Improved_Residual_GCN": "Improved_Residual_GCN",
            "Temporal_GCN_LSTM": "Temporal_GCN_LSTM",
            "Historical_GCN_LSTM": "Historical_GCN_LSTM",
        }
        for target_name, key in mapping.items():
            if key in by_key:
                r = by_key[key]
                benchmarks[target_name] = {
                    "source": "results/graphs/model_comparison.json",
                    "threshold": r["Threshold"],
                    "illicit_precision": r["Illicit_Precision"],
                    "illicit_recall": r["Illicit_Recall"],
                    "illicit_f1": r["Illicit_F1"],
                    "pr_auc": r["PR_AUC"],
                    "roc_auc": r["ROC_AUC"],
                    "accuracy": r["Accuracy"],
                    "tp": r["True_Positives"],
                    "fp": r["False_Positives"],
                    "fn": r["False_Negatives"],
                }

    # Fallbacks from stage JSONs if model_comparison.json is absent
    s3_path = os.path.join(reports_dir, "stage3_baselines_metrics.json")
    if "HistGradientBoosting" not in benchmarks and os.path.exists(s3_path):
        with open(s3_path, "r", encoding="utf-8") as f:
            gb = json.load(f).get("Gradient_Boosting", {})
            t = gb.get("test_metrics_optimal_threshold", {})
            if t:
                benchmarks["HistGradientBoosting"] = {
                    "source": "reports/stage3_baselines_metrics.json",
                    "threshold": gb.get("optimal_threshold"),
                    "illicit_precision": t["illicit_precision"],
                    "illicit_recall": t["illicit_recall"],
                    "illicit_f1": t["illicit_f1"],
                    "pr_auc": t["pr_auc"],
                    "roc_auc": t["roc_auc"],
                    "tp": t["confusion_matrix"]["true_positives_caught_fraud"],
                    "fp": t["confusion_matrix"]["false_positives_false_alarm"],
                    "fn": t["confusion_matrix"]["false_negatives_missed_fraud"],
                }

    s4a_path = os.path.join(reports_dir, "stage4a_gnn_static_metrics.json")
    if "Improved_Residual_GCN" not in benchmarks and os.path.exists(s4a_path):
        with open(s4a_path, "r", encoding="utf-8") as f:
            gcn = json.load(f).get("GCN_Improved_Residual", {})
            t = gcn.get("test_metrics_optimal_threshold", {})
            if t:
                benchmarks["Improved_Residual_GCN"] = {
                    "source": "reports/stage4a_gnn_static_metrics.json",
                    "threshold": gcn.get("optimal_threshold_f1"),
                    "illicit_precision": t["illicit_precision"],
                    "illicit_recall": t["illicit_recall"],
                    "illicit_f1": t["illicit_f1"],
                    "pr_auc": t["pr_auc"],
                    "roc_auc": t["roc_auc"],
                    "tp": t["confusion_matrix"]["true_positives_caught_fraud"],
                    "fp": t["confusion_matrix"]["false_positives_false_alarm"],
                    "fn": t["confusion_matrix"]["false_negatives_missed_fraud"],
                }

    return benchmarks


def save_stage4c_plots(configs_res, best_key, y_test_all, results_dir):
    """Generates comparative PR, ROC, Confusion Matrix, and Snapshot-42/Temporal plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {
        "EvolveGCN_H_Directed": "#1f77b4",
        "EvolveGCN_H_Directed_Res": "#ff7f0e",
        "EvolveGCN_H_Undirected": "#2ca02c",
        "EvolveGCN_H_Undirected_Res": "#d62728",
    }
    labels_short = {
        "EvolveGCN_H_Directed": "Directed (No Res)",
        "EvolveGCN_H_Directed_Res": "Directed + Residual",
        "EvolveGCN_H_Undirected": "Undirected (No Res)",
        "EvolveGCN_H_Undirected_Res": "Undirected + Residual",
    }

    # 1. Precision-Recall Curves
    ax_pr = axes[0, 0]
    no_skill = float((y_test_all == 1).sum() / len(y_test_all))
    for k, res in configs_res.items():
        prec, rec, _ = precision_recall_curve(y_test_all, res["test_probs"])
        pr_auc = res["test_metrics_optimal_threshold"]["pr_auc"]
        lw = 2.6 if k == best_key else 1.8
        ax_pr.plot(rec, prec, label=f"{labels_short[k]} (PR-AUC={pr_auc:.4f})", color=colors[k], lw=lw)
    ax_pr.axhline(y=no_skill, color="k", linestyle="--", lw=1.0, alpha=0.6, label=f"No-Skill ({no_skill:.4f})")
    ax_pr.set_title("Test Precision-Recall Curves (Steps 42–49)", fontsize=12, fontweight="bold")
    ax_pr.set_xlabel("Illicit Recall")
    ax_pr.set_ylabel("Illicit Precision")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.legend(loc="upper right", fontsize=8.5)

    # 2. ROC Curves
    ax_roc = axes[0, 1]
    for k, res in configs_res.items():
        fpr, tpr, _ = roc_curve(y_test_all, res["test_probs"])
        roc_auc = res["test_metrics_optimal_threshold"]["roc_auc"]
        lw = 2.6 if k == best_key else 1.8
        ax_roc.plot(fpr, tpr, label=f"{labels_short[k]} (AUC={roc_auc:.4f})", color=colors[k], lw=lw)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.6)
    ax_roc.set_title("Test ROC Curves (Steps 42–49)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.legend(loc="lower right", fontsize=8.5)

    # 3. Best Config Confusion Matrix
    ax_cm = axes[1, 0]
    best_res = configs_res[best_key]
    cm = best_res["test_metrics_optimal_threshold"]["confusion_matrix"]
    mat = np.array([
        [cm["true_negatives_licit"], cm["false_positives_false_alarm"]],
        [cm["false_negatives_missed_fraud"], cm["true_positives_caught_fraud"]],
    ])
    ax_cm.imshow(mat, cmap="Blues", interpolation="nearest")
    ax_cm.set_title(
        f"Best Val-Selected Config: {labels_short[best_key]}\n"
        f"(tau*={best_res['optimal_threshold_f1']:.2f} | F1={best_res['test_metrics_optimal_threshold']['illicit_f1']:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred Licit", "Pred Fraud"])
    ax_cm.set_yticklabels(["Actual Licit", "Actual Fraud"])
    for r in range(2):
        for c in range(2):
            txt_color = "white" if mat[r, c] > mat.max() * 0.55 else "black"
            ax_cm.text(c, r, f"{mat[r, c]:,}", ha="center", va="center", color=txt_color, fontweight="bold", fontsize=11)

    # 4. Per-Snapshot Test F1 Across Steps 42-49
    ax_tmp = axes[1, 1]
    steps = sorted(int(s) for s in best_res["per_snapshot_test_metrics"].keys())
    for k, res in configs_res.items():
        f1s = [res["per_snapshot_test_metrics"][str(s)]["illicit_f1"] for s in steps]
        lw = 2.4 if k == best_key else 1.6
        ax_tmp.plot(steps, f1s, marker="o", lw=lw, label=labels_short[k], color=colors[k])
    ax_tmp.set_title("Per-Snapshot Test Illicit F1 (Steps 42–49)", fontsize=12, fontweight="bold")
    ax_tmp.set_xlabel("Test Snapshot Time Step")
    ax_tmp.set_ylabel("Illicit F1-Score")
    ax_tmp.set_xticks(steps)
    ax_tmp.grid(True, alpha=0.3)
    ax_tmp.legend(loc="best", fontsize=8.5)

    plt.suptitle("Stage 4C: EvolveGCN-H Controlled Ablation & Evaluation", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "stage4c_evolvegcn_curves.png")
    plt.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Curves Plot]: {plot_path}")


def generate_stage4c_report(all_configs, best_key, baselines):
    """Generates a strictly data-driven Markdown report from observed test metrics."""
    best = all_configs[best_key]
    best_t = best["test_metrics_optimal_threshold"]
    best_v = best["validation_metrics"]
    best_cm = best_t["confusion_matrix"]

    def fmt(val, pct=False):
        if val is None:
            return "N/A"
        return f"{val*100:.2f}%" if pct else f"{val:.4f}"

    md = f"""# Stage 4C: EvolveGCN-H Controlled Experiments & Benchmark Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Model Family:** EvolveGCN-H (Top-$k$ Node Summarization + Matrix-GRU Weight Evolution + LayerNorm)  
**Evaluation Set:** Unseen Test Snapshots 42–49 (8,841 labeled transactions: 408 Illicit / 8,433 Licit)  

---

## 1. Controlled Ablation Study (Directed vs. Undirected & Residual Connections)

All four configurations were trained chronologically on **Steps 1–34**, tuned for early stopping and decision threshold $\\tau^*$ strictly on **Validation Steps 35–41** (using **Validation PR-AUC** for architecture selection and checkpointing), and evaluated once on **Test Steps 42–49**:

| Configuration | Undirected | Residual Skip | Best Epoch | Val PR-AUC | $\\tau^*$ | Test Prec | Test Rec | Test F1 | Test PR-AUC | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for k, c in all_configs.items():
        vm = c["validation_metrics"]
        tm = c["test_metrics_optimal_threshold"]
        star = " **(Selected)**" if k == best_key else ""
        md += (
            f"| `{k}`{star} | {c['use_undirected']} | {c['use_residual']} | {c['best_epoch']} | "
            f"{fmt(vm['pr_auc'])} | {c['optimal_threshold_f1']:.2f} | "
            f"{fmt(tm['illicit_precision'], True)} | {fmt(tm['illicit_recall'], True)} | "
            f"**{fmt(tm['illicit_f1'])}** | **{fmt(tm['pr_auc'])}** | {fmt(tm['roc_auc'])} |\n"
        )

    md += f"""
### Selected Best Model (by Validation PR-AUC): `{best_key}`
- **Validation PR-AUC:** `{fmt(best_v['pr_auc'])}` | **Validation Illicit F1:** `{fmt(best_v['illicit_f1'])}` at $\\tau^* = {best['optimal_threshold_f1']:.2f}$
- **Trainable Parameters:** `{best['total_parameters']:,}` | **Training Time:** `{best['training_time_seconds']:.2f}s` | **Test Inference Time:** `{best['inference_time_seconds']:.4f}s`
- **Test Confusion Matrix ($\\tau^* = {best['optimal_threshold_f1']:.2f}$):**
  - True Negatives (Licit): **{best_cm['true_negatives_licit']:,}** / 8,433
  - False Positives (False Alarms): **{best_cm['false_positives_false_alarm']:,}**
  - False Negatives (Missed Frauds): **{best_cm['false_negatives_missed_fraud']:,}** / 408
  - True Positives (Caught Frauds): **{best_cm['true_positives_caught_fraud']:,}** / 408
- **Recall-Oriented Operating Point ($\\tau = {best['threshold_recall_constrained']:.2f}$, Val Prec $\\ge 30\\%$):**
  - Test Precision: `{fmt(best['test_metrics_recall_constrained']['illicit_precision'], True)}` | Test Recall: `{fmt(best['test_metrics_recall_constrained']['illicit_recall'], True)}` | Test F1: `{fmt(best['test_metrics_recall_constrained']['illicit_f1'])}`

---

## 2. Benchmark Comparison Against Existing Capstone Models

Comparison on the exact same unseen Test Set (Steps 42–49, `8,841` transactions):

| Model | Paradigm | $\\tau^*$ | Illicit Prec | Illicit Rec | Illicit F1 | PR-AUC | ROC-AUC | TP / 408 | FP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for b_name in ["HistGradientBoosting", "Improved_Residual_GCN", "Temporal_GCN_LSTM", "Historical_GCN_LSTM"]:
        if b_name in baselines:
            b = baselines[b_name]
            md += (
                f"| **{b_name}** | Reference Baseline | {b['threshold']:.2f} | "
                f"{fmt(b['illicit_precision'], True)} | {fmt(b['illicit_recall'], True)} | "
                f"{fmt(b['illicit_f1'])} | {fmt(b['pr_auc'])} | {fmt(b['roc_auc'])} | "
                f"{b['tp']} | {b['fp']:,} |\n"
            )
    md += (
        f"| **`{best_key}` (Stage 4C)** | **EvolveGCN-H** | {best['optimal_threshold_f1']:.2f} | "
        f"{fmt(best_t['illicit_precision'], True)} | {fmt(best_t['illicit_recall'], True)} | "
        f"**{fmt(best_t['illicit_f1'])}** | **{fmt(best_t['pr_auc'])}** | {fmt(best_t['roc_auc'])} | "
        f"{best_cm['true_positives_caught_fraud']} | {best_cm['false_positives_false_alarm']:,} |\n"
    )

    # Objective programmatic deltas (no unsupported claims)
    md += "\n### Objective Test-Set Delta Summary\n"
    for b_name in ["HistGradientBoosting", "Improved_Residual_GCN", "Temporal_GCN_LSTM", "Historical_GCN_LSTM"]:
        if b_name in baselines:
            b = baselines[b_name]
            d_f1 = best_t["illicit_f1"] - b["illicit_f1"]
            d_pr = (best_t["pr_auc"] or 0.0) - b["pr_auc"]
            d_prec = best_t["illicit_precision"] - b["illicit_precision"]
            d_rec = best_t["illicit_recall"] - b["illicit_recall"]
            md += (
                f"- **vs. `{b_name}`:** "
                f"$\\Delta$F1 = `{d_f1:+.4f}`, $\\Delta$PR-AUC = `{d_pr:+.4f}`, "
                f"$\\Delta$Precision = `{d_prec*100:+.2f} pp`, $\\Delta$Recall = `{d_rec*100:+.2f} pp`.\n"
            )

    md += """
---

## 3. Per-Snapshot Test Trajectory & Calibration Check (Steps 42–49)

| Test Step | Labeled Nodes | Illicit | Mean Prob | Median Prob | Precision | Recall | Illicit F1 | PR-AUC | TP | FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for step_str in sorted(best["per_snapshot_test_metrics"].keys(), key=int):
        sm = best["per_snapshot_test_metrics"][step_str]
        md += (
            f"| {step_str} | {sm['labeled_nodes']:,} | {sm['illicit_count']} | "
            f"{sm['labeled_mean_prob']:.4f} | {sm['labeled_median_prob']:.4f} | "
            f"{fmt(sm['illicit_precision'], True)} | {fmt(sm['illicit_recall'], True)} | "
            f"{fmt(sm['illicit_f1'])} | {fmt(sm['pr_auc'])} | {sm['tp']} | {sm['fp']} |\n"
        )

    md += """
---

## 4. Architectural Choices & Leakage Controls
- **Weight Evolution Without Additive Bias:** `EvolveGCNHLayer` evolves $W_t^{(l)}$ directly via `GRUCell` conditioned on `TopKSummarizer(H_t)`, removing the global broadcast vector $s_t$ used in Stage 4B.
- **LayerNorm Stabilization:** Per-node `LayerNorm` replaces `BatchNorm1d`, avoiding cross-node batch-statistics coupling across varying snapshot sizes (`1,089–7,880` nodes).
- **Zero-Leakage Protocol:** Imputer/scaler fitted solely on Steps `1–34`; `100%` intra-snapshot edges; `pos_weight` computed strictly on Steps `1–34`; unlabeled nodes (`y = -1`) excluded from loss/metrics; architecture selection, early stopping, and thresholds chosen strictly on Validation Steps `35–41`.
- **Training Stability:** Gradient norm clipped at `max_norm = 2.0` across 34-step BPTT.
"""
    return md


# =============================================================================
# 6. MAIN PIPELINE ENTRYPOINT
# =============================================================================

def run_stage4c(smoke_test=False):
    print("=" * 85)
    mode_str = " (SMOKE TEST MODE)" if smoke_test else ""
    print(f"STAGE 4C: EVOLVEGCN-H TEMPORAL FRAUD DETECTION & ABLATION STUDY{mode_str}")
    print("=" * 85)
    start_total_time = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "processed")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")

    snapshots_path = os.path.join(processed_dir, "temporal_snapshots.pt")
    if not os.path.exists(snapshots_path):
        raise FileNotFoundError(f"Temporal snapshots not found at {snapshots_path}. Run Stage 2 preprocessing first!")

    print(f"Loading 49 Temporal Snapshots from {snapshots_path}...")
    snapshots = torch.load(snapshots_path, weights_only=False)
    assert len(snapshots) == 49, f"Expected 49 snapshots, got {len(snapshots)}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation Device: {device}")

    # Strict training-only pos_weight (Steps 1-34)
    train_labels = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34)])
    n_licit_tr = int((train_labels == 0).sum())
    n_illicit_tr = int((train_labels == 1).sum())
    pos_weight_val = n_licit_tr / max(n_illicit_tr, 1)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    print(f"Training Class Imbalance (Steps 1-34): {n_illicit_tr:,} Illicit vs {n_licit_tr:,} Licit -> pos_weight = {pos_weight_val:.4f}")

    y_val_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34, 41)])
    y_test_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(41, 49)])
    print(f"Chronological Partitions -> Val (35-41): {len(y_val_all):,} | Test (42-49): {len(y_test_all):,}")

    snapshots_dev = [s.to(device) for s in snapshots]

    # Precompute isolated per-snapshot normalized sparse adjacency matrices
    adjs_directed = [
        build_normalized_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device)
        for s in snapshots
    ]
    adjs_undirected = [
        build_normalized_sparse_adj(s.edge_index, s.num_nodes, use_undirected=True, device=device)
        for s in snapshots
    ]

    # Controlled 2x2 Ablation Matrix: Directed vs. Undirected x No-Residual vs. Residual
    experiment_grid = [
        ("EvolveGCN_H_Directed", False, False),
        ("EvolveGCN_H_Directed_Res", False, True),
        ("EvolveGCN_H_Undirected", True, False),
        ("EvolveGCN_H_Undirected_Res", True, True),
    ]

    max_epochs = 2 if smoke_test else 60
    all_configs = {}

    for cfg_name, use_undir, use_res in experiment_grid:
        adjs = adjs_undirected if use_undir else adjs_directed
        res = train_and_validate_config(
            config_name=cfg_name,
            use_undirected=use_undir,
            use_residual=use_res,
            snapshots_dev=snapshots_dev,
            adjs_dev=adjs,
            y_val_all=y_val_all,
            pos_weight=pos_weight,
            device=device,
            max_epochs=max_epochs,
            lr=0.002,
            patience=15,
            grad_clip=2.0,
            smoke_test=smoke_test,
        )
        all_configs[cfg_name] = res

    # Select winning architecture strictly by Validation PR-AUC (zero test leakage)
    best_key = max(
        all_configs.keys(),
        key=lambda k: (
            all_configs[k]["validation_metrics"]["pr_auc"] or 0.0,
            all_configs[k]["validation_metrics"]["illicit_f1"],
        ),
    )
    best_res = all_configs[best_key]
    print("\n" + "=" * 85)
    print(
        f"SELECTED BEST ARCHITECTURE (Validation PR-AUC): {best_key} "
        f"(Val PR-AUC={best_res['validation_metrics']['pr_auc']:.4f})"
    )
    print("=" * 85)

    # Post-selection Test Evaluation (Test Steps 42-49 touched ONLY after selection)
    print("\n[Post-Selection Evaluation] Evaluating selected model first, then ablation variants on Test Steps 42-49...")
    ordered_keys = [best_key] + [k for k in all_configs.keys() if k != best_key]
    for k in ordered_keys:
        adjs = adjs_undirected if all_configs[k]["use_undirected"] else adjs_directed
        evaluate_config_on_test(all_configs[k], snapshots_dev, adjs, y_test_all, device)

    baselines = load_comparison_baselines(reports_dir, results_dir)

    clean_configs = {
        k: {k2: v2 for k2, v2 in v.items() if k2 not in ("best_state", "test_probs")}
        for k, v in all_configs.items()
    }

    output_payload = {
        "stage": "Stage 4C - EvolveGCN-H",
        "selection_criterion": "validation_pr_auc",
        "selected_best_config": best_key,
        "pos_weight": round(float(pos_weight_val), 4),
        "configs": clean_configs,
        "baselines_comparison": baselines,
    }

    if smoke_test:
        print("\n[Smoke Test] Skipping checkpoint, report, and plot disk writes to protect production artifacts.")
    else:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        ckpt_path = os.path.join(models_dir, "evolvegcn_h.pt")
        torch.save(
            {
                "config_name": best_key,
                "model_state_dict": best_res["best_state"],
                "in_channels": snapshots[0].x.shape[1],
                "hidden_channels": 128,
                "out_channels": 64,
                "use_undirected": best_res["use_undirected"],
                "use_residual": best_res["use_residual"],
                "best_epoch": best_res["best_epoch"],
                "optimal_threshold_f1": best_res["optimal_threshold_f1"],
                "threshold_recall_constrained": best_res["threshold_recall_constrained"],
                "pos_weight": float(pos_weight_val),
                "validation_metrics": best_res["validation_metrics"],
                "test_metrics_optimal_threshold": best_res["test_metrics_optimal_threshold"],
            },
            ckpt_path,
        )
        print(f"  [Saved Best Checkpoint]: {ckpt_path}")

        save_stage4c_plots(all_configs, best_key, y_test_all, results_dir)

        json_path = os.path.join(reports_dir, "stage4c_evolvegcn_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=4)
        print(f"  [Saved Metrics JSON]: {json_path}")

        md_path = os.path.join(reports_dir, "stage4c_evolvegcn_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_stage4c_report(clean_configs, best_key, baselines))
        print(f"  [Saved Markdown Report]: {md_path}")

    total_time = time.time() - start_total_time
    print(f"\nStage 4C completed successfully in {total_time:.2f} seconds!")
    print("=" * 85)
    return output_payload


if __name__ == "__main__":
    is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
    run_stage4c(smoke_test=is_smoke)
