"""
Stage 4C Threshold Recalibration (Validation-Only Grid Expansion: 0.01-0.99)
Loads frozen checkpoint models/evolvegcn_h.pt without retraining or modifying any existing files.
Capstone Project: Temporal Graph Neural Networks with Real-Time Explainable AI
for Online Payment Fraud Ring Detection
"""

import os
import sys
import time
import json
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.train_evolvegcn import (
    FraudEvolveGCNH,
    build_normalized_sparse_adj,
    rollout_val_split,
    rollout_test_split,
    compute_metrics,
)


def find_thresholds_expanded(y_val, y_val_prob, min_precision=0.30):
    """
    Sweeps thresholds in [0.01, 0.99] (step 0.01) strictly on Validation labels/probabilities:
      1. tau_f1: maximizes Validation Illicit F1-score.
      2. tau_rec: maximizes Validation Illicit Recall subject to Precision >= min_precision.
    Returns (tau_f1, best_val_f1, tau_rec, best_val_rec, curve_records).
    """
    grid = np.linspace(0.01, 0.99, 99)
    best_f1_thresh = 0.50
    best_f1 = -1.0

    best_rec_thresh = None
    best_rec = -1.0

    curve_records = []
    for t in grid:
        t_r = float(round(float(t), 2))
        preds = (y_val_prob >= t_r).astype(int)
        prec = float(precision_score(y_val, preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_val, preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_val, preds, pos_label=1, zero_division=0))

        curve_records.append({
            "threshold": t_r,
            "val_precision": round(prec, 4),
            "val_recall": round(rec, 4),
            "val_f1": round(f1, 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_f1_thresh = t_r

        if prec >= min_precision and rec > best_rec:
            best_rec = rec
            best_rec_thresh = t_r

    if best_rec_thresh is None:
        best_rec_thresh = best_f1_thresh
        best_rec = recall_score(y_val, (y_val_prob >= best_f1_thresh).astype(int), pos_label=1, zero_division=0)

    return best_f1_thresh, float(round(best_f1, 4)), best_rec_thresh, float(round(best_rec, 4)), curve_records


def save_recalibrated_plots(
    curve_records,
    orig_tau_f1,
    recal_tau_f1,
    orig_test_opt,
    recal_test_opt,
    orig_per_step,
    recal_per_step,
    results_dir,
):
    """Saves recalibrated threshold sweep, confusion matrices, and per-snapshot F1 comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Validation Threshold Sweep (0.01 - 0.99)
    ax0 = axes[0, 0]
    ts = [r["threshold"] for r in curve_records]
    precs = [r["val_precision"] for r in curve_records]
    recs = [r["val_recall"] for r in curve_records]
    f1s = [r["val_f1"] for r in curve_records]

    ax0.plot(ts, precs, label="Val Precision", color="#1f77b4", lw=2.0)
    ax0.plot(ts, recs, label="Val Recall", color="#d62728", lw=2.0)
    ax0.plot(ts, f1s, label="Val Illicit F1", color="#2ca02c", lw=2.4)
    ax0.axvline(x=orig_tau_f1, color="#555555", linestyle="--", lw=1.6, label=f"Original Grid Cap tau*={orig_tau_f1:.2f}")
    ax0.axvline(x=recal_tau_f1, color="#9467bd", linestyle="-.", lw=2.0, label=f"Recalibrated tau*={recal_tau_f1:.2f}")
    ax0.set_title("Validation Threshold Sweep (Steps 35-41, 0.01-0.99)", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Decision Threshold (tau)")
    ax0.set_ylabel("Validation Metric")
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(-0.02, 1.02)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower left", fontsize=8.5)

    # 2. Original vs Recalibrated Test Metrics Bar Chart
    ax1 = axes[0, 1]
    metric_names = ["Precision", "Recall", "Illicit F1", "Macro F1", "Accuracy"]
    orig_vals = [
        orig_test_opt["illicit_precision"],
        orig_test_opt["illicit_recall"],
        orig_test_opt["illicit_f1"],
        orig_test_opt["macro_f1"],
        orig_test_opt["accuracy"],
    ]
    recal_vals = [
        recal_test_opt["illicit_precision"],
        recal_test_opt["illicit_recall"],
        recal_test_opt["illicit_f1"],
        recal_test_opt["macro_f1"],
        recal_test_opt["accuracy"],
    ]
    x = np.arange(len(metric_names))
    w = 0.36
    b1 = ax1.bar(x - w / 2, orig_vals, w, label=f"Original (tau*={orig_tau_f1:.2f})", color="#7f7f7f", edgecolor="black")
    b2 = ax1.bar(x + w / 2, recal_vals, w, label=f"Recalibrated (tau*={recal_tau_f1:.2f})", color="#2ca02c", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_names, fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Test Set Performance: Original vs. Recalibrated (Steps 42-49)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # 3. Confusion Matrix Comparison (Recalibrated)
    ax2 = axes[1, 0]
    cm = recal_test_opt["confusion_matrix"]
    mat = np.array([
        [cm["true_negatives_licit"], cm["false_positives_false_alarm"]],
        [cm["false_negatives_missed_fraud"], cm["true_positives_caught_fraud"]],
    ])
    ax2.imshow(mat, cmap="Blues", interpolation="nearest")
    ax2.set_title(
        f"Recalibrated Test Confusion Matrix (tau*={recal_tau_f1:.2f})\n"
        f"FP: {orig_test_opt['confusion_matrix']['false_positives_false_alarm']:,} -> {cm['false_positives_false_alarm']:,}",
        fontsize=11.5,
        fontweight="bold",
    )
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Pred Licit", "Pred Fraud"])
    ax2.set_yticklabels(["Actual Licit", "Actual Fraud"])
    for r in range(2):
        for c in range(2):
            txt_col = "white" if mat[r, c] > mat.max() * 0.55 else "black"
            ax2.text(c, r, f"{mat[r, c]:,}", ha="center", va="center", color=txt_col, fontweight="bold", fontsize=11)

    # 4. Per-Snapshot Test F1 Comparison (Steps 42-49)
    ax3 = axes[1, 1]
    steps = sorted(int(s) for s in recal_per_step.keys())
    f1_orig_steps = [orig_per_step[str(s)]["illicit_f1"] for s in steps]
    f1_recal_steps = [recal_per_step[str(s)]["illicit_f1"] for s in steps]
    ax3.plot(steps, f1_orig_steps, marker="s", lw=2.0, color="#7f7f7f", label=f"Original (tau*={orig_tau_f1:.2f})")
    ax3.plot(steps, f1_recal_steps, marker="o", lw=2.4, color="#2ca02c", label=f"Recalibrated (tau*={recal_tau_f1:.2f})")
    ax3.set_title("Per-Snapshot Test Illicit F1 (Steps 42-49)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Test Snapshot Time Step")
    ax3.set_ylabel("Illicit F1-Score")
    ax3.set_xticks(steps)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="best", fontsize=9)

    plt.suptitle("Stage 4C EvolveGCN-H Threshold Recalibration (Validation-Only 0.01-0.99)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = os.path.join(results_dir, "stage4c_evolvegcn_curves_recalibrated.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Recalibrated Plot]: {out_path}")


def generate_recalibrated_report(payload):
    orig = payload["original_stage4c"]
    rec = payload["recalibrated_stage4c"]
    orig_v = orig["validation_metrics"]
    orig_t = orig["test_metrics_optimal_threshold"]
    rec_v = rec["validation_metrics_optimal_threshold"]
    rec_t = rec["test_metrics_optimal_threshold"]
    rec_r30 = rec["test_metrics_recall_constrained"]
    orig_r30 = orig["test_metrics_recall_constrained"]

    ocm = orig_t["confusion_matrix"]
    rcm = rec_t["confusion_matrix"]

    d_prec = rec_t["illicit_precision"] - orig_t["illicit_precision"]
    d_rec = rec_t["illicit_recall"] - orig_t["illicit_recall"]
    d_f1 = rec_t["illicit_f1"] - orig_t["illicit_f1"]
    d_fp = rcm["false_positives_false_alarm"] - ocm["false_positives_false_alarm"]
    d_tp = rcm["true_positives_caught_fraud"] - ocm["true_positives_caught_fraud"]

    md = f"""# Stage 4C Threshold Recalibration Report (Validation-Only Grid `0.01–0.99`)

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** {payload['recalibration_timestamp']}  
**Frozen Checkpoint Evaluated:** [models/evolvegcn_h.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/evolvegcn_h.pt) (`EvolveGCN_H_Directed_Res`, `best_epoch=37`)  
**Protocol Distinction:** This report evaluates the **exact same frozen Stage 4C model** using an expanded **validation-only** threshold grid (`0.01–0.99`, step `0.01`), separated from the original Stage 4C evaluation (`0.10–0.90` grid aligned with Stages 3–4B). Zero retraining and zero test-label tuning were performed.

---

## 1. Validation-Only Threshold Selection Summary (Steps `35–41`, `7,829` Nodes)

| Evaluation Protocol | Search Grid | $\\tau^*$ (F1-Opt) | Val Prec | Val Rec | Val F1 | $\\tau_{{\\text{{rec30}}}}$ (Prec $\\ge 30\\%$) | Val Prec ($\\tau_{{\\text{{rec30}}}}$) | Val Rec ($\\tau_{{\\text{{rec30}}}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage 4C** | `0.10–0.90` | `0.90` | 35.21% | 75.85% | 0.4810 | `0.83` | 30.12% | 80.15% |
| **Recalibrated Stage 4C** | `0.01–0.99` | **`{rec['optimal_threshold_f1']:.2f}`** | **{rec_v['illicit_precision']*100:.2f}%** | **{rec_v['illicit_recall']*100:.2f}%** | **{rec_v['illicit_f1']:.4f}** | **`{rec['threshold_recall_constrained']:.2f}`** | **{rec['validation_metrics_recall_constrained']['illicit_precision']*100:.2f}%** | **{rec['validation_metrics_recall_constrained']['illicit_recall']*100:.2f}%** |

- **Leakage Verification:** `rollout_val_split` was executed strictly on Steps `1–34` (warm-up) and Steps `35–41` (validation); Test Steps `42–49` were evaluated only after both thresholds (`{rec['optimal_threshold_f1']:.2f}` and `{rec['threshold_recall_constrained']:.2f}`) were finalized.

---

## 2. Side-by-Side Test Results: Original vs. Recalibrated (Steps `42–49`, `8,841` Nodes)

### A. F1-Optimal Operating Point ($\\tau^*$)

| Protocol | $\\tau^*$ | Precision | Recall | Illicit F1 | PR-AUC | ROC-AUC | Accuracy | Macro F1 | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage 4C** (`0.10–0.90`) | `0.90` | {orig_t['illicit_precision']*100:.2f}% | {orig_t['illicit_recall']*100:.2f}% | {orig_t['illicit_f1']:.4f} | {orig_t['pr_auc']:.4f} | {orig_t['roc_auc']:.4f} | {orig_t['accuracy']:.4f} | {orig_t['macro_f1']:.4f} | {ocm['true_positives_caught_fraud']} | {ocm['false_positives_false_alarm']:,} | {ocm['true_negatives_licit']:,} | {ocm['false_negatives_missed_fraud']} |
| **Recalibrated Stage 4C** (`0.01–0.99`) | **`{rec['optimal_threshold_f1']:.2f}`** | **{rec_t['illicit_precision']*100:.2f}%** | **{rec_t['illicit_recall']*100:.2f}%** | **{rec_t['illicit_f1']:.4f}** | {rec_t['pr_auc']:.4f} | {rec_t['roc_auc']:.4f} | {rec_t['accuracy']:.4f} | {rec_t['macro_f1']:.4f} | {rcm['true_positives_caught_fraud']} | {rcm['false_positives_false_alarm']:,} | {rcm['true_negatives_licit']:,} | {rcm['false_negatives_missed_fraud']} |
| **Delta (Recalibrated - Original)** | `+{rec['optimal_threshold_f1'] - orig['optimal_threshold_f1']:.2f}` | `{d_prec*100:+.2f} pp` | `{d_rec*100:+.2f} pp` | **`{d_f1:+.4f}`** | `0.0000` | `0.0000` | `{rec_t['accuracy'] - orig_t['accuracy']:+.4f}` | `{rec_t['macro_f1'] - orig_t['macro_f1']:+.4f}` | `{d_tp:+d}` | **`{d_fp:+,d}`** | `{-d_fp:+,d}` | `{-d_tp:+d}` |

### B. Recall-Constrained Operating Point ($\\tau_{{\\text{{rec30}}}}$, Val Prec $\\ge 30\\%$)

| Protocol | $\\tau_{{\\text{{rec30}}}}$ | Precision | Recall | Illicit F1 | PR-AUC | ROC-AUC | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage 4C** (`0.10–0.90`) | `0.83` | {orig_r30['illicit_precision']*100:.2f}% | {orig_r30['illicit_recall']*100:.2f}% | {orig_r30['illicit_f1']:.4f} | {orig_r30['pr_auc']:.4f} | {orig_r30['roc_auc']:.4f} | {orig_r30['confusion_matrix']['true_positives_caught_fraud']} | {orig_r30['confusion_matrix']['false_positives_false_alarm']:,} | {orig_r30['confusion_matrix']['true_negatives_licit']:,} | {orig_r30['confusion_matrix']['false_negatives_missed_fraud']} |
| **Recalibrated Stage 4C** (`0.01–0.99`) | **`{rec['threshold_recall_constrained']:.2f}`** | {rec_r30['illicit_precision']*100:.2f}% | {rec_r30['illicit_recall']*100:.2f}% | {rec_r30['illicit_f1']:.4f} | {rec_r30['pr_auc']:.4f} | {rec_r30['roc_auc']:.4f} | {rec_r30['confusion_matrix']['true_positives_caught_fraud']} | {rec_r30['confusion_matrix']['false_positives_false_alarm']:,} | {rec_r30['confusion_matrix']['true_negatives_licit']:,} | {rec_r30['confusion_matrix']['false_negatives_missed_fraud']} |

*(Note: PR-AUC `{rec_t['pr_auc']:.4f}` and ROC-AUC `{rec_t['roc_auc']:.4f}` are threshold-independent ranking metrics and remain identical across threshold choices.)*

---

## 3. Per-Snapshot Test Comparison (`Steps 42–49`, F1-Optimal Threshold)

| Step | Illicit | Orig FP (`0.90`) | Recal FP (`{rec['optimal_threshold_f1']:.2f}`) | Orig TP (`0.90`) | Recal TP (`{rec['optimal_threshold_f1']:.2f}`) | Orig F1 | Recal F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for step_str in sorted(rec["per_snapshot_test_metrics"].keys(), key=int):
        om = orig["per_snapshot_test_metrics"][step_str]
        rm = rec["per_snapshot_test_metrics"][step_str]
        md += (
            f"| {step_str} | {rm['illicit_count']} | {om['fp']} | **{rm['fp']}** | "
            f"{om['tp']} | {rm['tp']} | {om['illicit_f1']:.4f} | **{rm['illicit_f1']:.4f}** |\n"
        )

    md += f"""
---

## 4. Benchmarking Fairness Note
- The 10 existing baseline models in [results/graphs/model_comparison.csv](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/graphs/model_comparison.csv) (`Logistic Regression`, `Temporal GCN-LSTM`, `Historical GCN-LSTM`, `Improved Residual GAT`, etc.) were tuned on the `0.10–0.90` grid.
- Therefore, **Original Stage 4C** ($\\tau^* = 0.90$) should be used for apples-to-apples comparisons against the `0.10–0.90` grid baselines, while **Recalibrated Stage 4C** ($\\tau^* = {rec['optimal_threshold_f1']:.2f}$) demonstrates the true calibrated operating capacity of the frozen `EvolveGCN_H_Directed_Res` weights without the `0.90` ceiling.
"""
    return md


def run_recalibration():
    print("=" * 85)
    print("STAGE 4C THRESHOLD RECALIBRATION (FROZEN CHECKPOINT, VALIDATION-ONLY 0.01-0.99)")
    print("=" * 85)
    t0 = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(base_dir, "models", "evolvegcn_h.pt")
    snaps_path = os.path.join(base_dir, "processed", "temporal_snapshots.pt")
    orig_json_path = os.path.join(base_dir, "reports", "stage4c_evolvegcn_metrics.json")
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")

    assert os.path.exists(ckpt_path), f"Missing frozen checkpoint: {ckpt_path}"
    assert os.path.exists(snaps_path), f"Missing snapshots: {snaps_path}"
    assert os.path.exists(orig_json_path), f"Missing original Stage 4C metrics: {orig_json_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    snaps = torch.load(snaps_path, weights_only=False)
    with open(orig_json_path, "r", encoding="utf-8") as f:
        orig_json = json.load(f)

    orig_cfg_key = orig_json["selected_best_config"]
    orig_cfg = orig_json["configs"][orig_cfg_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudEvolveGCNH(
        in_channels=ckpt["in_channels"],
        hidden_channels=ckpt["hidden_channels"],
        out_channels=ckpt["out_channels"],
        dropout_rate=0.3,
        use_residual=ckpt["use_residual"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    snaps_dev = [s.to(device) for s in snaps]
    adjs_dev = [
        build_normalized_sparse_adj(s.edge_index, s.num_nodes, use_undirected=ckpt["use_undirected"], device=device)
        for s in snaps
    ]

    # 1. Validation-only rollout & threshold search (ZERO access to Test 42-49)
    y_val_all = np.concatenate([snaps[t].y[snaps[t].labeled_mask].numpy() for t in range(34, 41)])
    val_probs = rollout_val_split(model, snaps_dev, adjs_dev)

    recal_tau_f1, val_best_f1, recal_tau_rec30, val_best_rec30, curve_records = find_thresholds_expanded(
        y_val_all, val_probs, min_precision=0.30
    )

    val_metrics_opt = compute_metrics(y_val_all, val_probs, threshold=recal_tau_f1)
    val_metrics_rec30 = compute_metrics(y_val_all, val_probs, threshold=recal_tau_rec30)

    print(f"Frozen Checkpoint: {ckpt['config_name']} (Best Epoch: {ckpt['best_epoch']})")
    print(f"Validation-Only Selection (Steps 35-41, N={len(y_val_all):,}):")
    print(f"  - Original F1-Opt tau*       = {ckpt['optimal_threshold_f1']:.2f} (Val F1={orig_cfg['validation_metrics']['illicit_f1']:.4f})")
    print(f"  - Recalibrated F1-Opt tau*   = {recal_tau_f1:.2f} (Val F1={val_best_f1:.4f}, Prec={val_metrics_opt['illicit_precision']:.4f}, Rec={val_metrics_opt['illicit_recall']:.4f})")
    print(f"  - Recalibrated tau_rec30     = {recal_tau_rec30:.2f} (Val Prec={val_metrics_rec30['illicit_precision']:.4f}, Rec={val_metrics_rec30['illicit_recall']:.4f})")

    # 2. Post-selection Test rollout (Steps 42-49) ONLY AFTER both thresholds are finalized
    y_test_all = np.concatenate([snaps[t].y[snaps[t].labeled_mask].numpy() for t in range(41, 49)])
    test_probs, step_details, inf_time = rollout_test_split(model, snaps_dev, adjs_dev)

    recal_test_opt = compute_metrics(y_test_all, test_probs, threshold=recal_tau_f1)
    recal_test_rec30 = compute_metrics(y_test_all, test_probs, threshold=recal_tau_rec30)

    per_snapshot_recal = {}
    for step, info in step_details.items():
        m_step = compute_metrics(info["y"], info["probs"], threshold=recal_tau_f1)
        per_snapshot_recal[str(step)] = {
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
            "tn": m_step["confusion_matrix"]["true_negatives_licit"],
        }

    print(f"\nPost-Selection Test Results (Steps 42-49, N={len(y_test_all):,}):")
    print(
        f"  - Original (tau*={ckpt['optimal_threshold_f1']:.2f}): "
        f"Prec={orig_cfg['test_metrics_optimal_threshold']['illicit_precision']:.4f}, "
        f"Rec={orig_cfg['test_metrics_optimal_threshold']['illicit_recall']:.4f}, "
        f"F1={orig_cfg['test_metrics_optimal_threshold']['illicit_f1']:.4f}, "
        f"FP={orig_cfg['test_metrics_optimal_threshold']['confusion_matrix']['false_positives_false_alarm']}"
    )
    print(
        f"  - Recalibrated (tau*={recal_tau_f1:.2f}): "
        f"Prec={recal_test_opt['illicit_precision']:.4f}, "
        f"Rec={recal_test_opt['illicit_recall']:.4f}, "
        f"F1={recal_test_opt['illicit_f1']:.4f}, "
        f"PR-AUC={recal_test_opt['pr_auc']:.4f}, "
        f"ROC-AUC={recal_test_opt['roc_auc']:.4f}, "
        f"TP={recal_test_opt['confusion_matrix']['true_positives_caught_fraud']}, "
        f"FP={recal_test_opt['confusion_matrix']['false_positives_false_alarm']}, "
        f"TN={recal_test_opt['confusion_matrix']['true_negatives_licit']}, "
        f"FN={recal_test_opt['confusion_matrix']['false_negatives_missed_fraud']}"
    )

    payload = {
        "stage": "Stage 4C - EvolveGCN-H Threshold Recalibration",
        "recalibration_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint_evaluated": "models/evolvegcn_h.pt",
        "config_name": ckpt["config_name"],
        "best_epoch": int(ckpt["best_epoch"]),
        "threshold_search_grid": "0.01-0.99 (step 0.01, validation Steps 35-41 only)",
        "test_labels_used_in_threshold_selection": False,
        "original_stage4c": {
            "threshold_search_grid": "0.10-0.90 (step 0.01, validation Steps 35-41 only)",
            "optimal_threshold_f1": orig_cfg["optimal_threshold_f1"],
            "threshold_recall_constrained": orig_cfg["threshold_recall_constrained"],
            "validation_metrics": orig_cfg["validation_metrics"],
            "test_metrics_optimal_threshold": orig_cfg["test_metrics_optimal_threshold"],
            "test_metrics_recall_constrained": orig_cfg["test_metrics_recall_constrained"],
            "per_snapshot_test_metrics": orig_cfg["per_snapshot_test_metrics"],
        },
        "recalibrated_stage4c": {
            "optimal_threshold_f1": recal_tau_f1,
            "threshold_recall_constrained": recal_tau_rec30,
            "inference_time_seconds": round(inf_time, 4),
            "validation_metrics_optimal_threshold": val_metrics_opt,
            "validation_metrics_recall_constrained": val_metrics_rec30,
            "test_metrics_optimal_threshold": recal_test_opt,
            "test_metrics_recall_constrained": recal_test_rec30,
            "per_snapshot_test_metrics": per_snapshot_recal,
            "validation_threshold_curve": curve_records,
        },
    }

    json_out = os.path.join(reports_dir, "stage4c_evolvegcn_metrics_recalibrated.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"\n  [Saved Recalibrated Metrics JSON]: {json_out}")

    md_out = os.path.join(reports_dir, "stage4c_evolvegcn_report_recalibrated.md")
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(generate_recalibrated_report(payload))
    print(f"  [Saved Recalibrated Markdown Report]: {md_out}")

    save_recalibrated_plots(
        curve_records=curve_records,
        orig_tau_f1=orig_cfg["optimal_threshold_f1"],
        recal_tau_f1=recal_tau_f1,
        orig_test_opt=orig_cfg["test_metrics_optimal_threshold"],
        recal_test_opt=recal_test_opt,
        orig_per_step=orig_cfg["per_snapshot_test_metrics"],
        recal_per_step=per_snapshot_recal,
        results_dir=results_dir,
    )

    print(f"\nRecalibration completed successfully in {time.time() - t0:.2f} seconds!")
    print("=" * 85)
    return payload


if __name__ == "__main__":
    run_recalibration()
