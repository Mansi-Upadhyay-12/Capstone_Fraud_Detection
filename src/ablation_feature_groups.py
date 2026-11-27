"""
Stage 4E: Controlled Feature-Group Ablation for GCNAvg (Local-93 vs. Local-93+Domain-17 vs. Full-182)

Isolates the individual contributions of Domain-17 (indices 165..181) and
Aggregate-72 (indices 93..164) under strictly identical architecture, graph
topology, optimizer, weighted-BCE loss, training schedule, and evaluation rules:
  - Preprocessing: Train Steps 1-34 ONLY (StandardScaler + SimpleImputer)
  - Training: Steps 1-34 ONLY
  - Early stopping, config selection, threshold selection: Validation Steps 35-41 ONLY
  - Post-selection evaluation: Test Steps 42-49 ONLY after all selections finalize

Does NOT modify any Stage 1-4D scripts, checkpoints, metrics, or reports.
"""

import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models_nextgen import build_gcn_sparse_adj, SeparatedGraphNet
from src.train_nextgen import compute_metrics_and_calibration, select_thresholds_val_only


SEEDS = [42, 123, 456]


def build_ablation_configs(feature_columns):
    """Defines the 3 controlled feature-subset configurations using exact schema indices."""
    assert len(feature_columns) == 182, f"Expected 182 columns, got {len(feature_columns)}"
    assert all(c.startswith("Local_feature_") for c in feature_columns[:93])
    assert all(c.startswith("Aggregate_feature_") for c in feature_columns[93:165])
    domain_expected = [
        "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
        "num_input_addresses", "num_output_addresses",
        "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
        "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
    ]
    assert feature_columns[165:182] == domain_expected, f"Domain column mismatch: {feature_columns[165:182]}"

    return [
        {
            "config_id": "F1_Local93",
            "label": "Local-93 (1-93)",
            "description": "Node-intrinsic Elliptic local features only (indices 0..92, excludes Aggregate-72 and Domain-17)",
            "feature_indices": list(range(0, 93)),
            "in_channels": 93,
            "feature_names_sample": [feature_columns[0], feature_columns[92]],
        },
        {
            "config_id": "F2_Local93_Domain17",
            "label": "Local-93 + Domain-17 (110)",
            "description": "Local features + 17 BTC/tx-degree domain features (indices 0..92 + 165..181, excludes Aggregate-72)",
            "feature_indices": list(range(0, 93)) + list(range(165, 182)),
            "in_channels": 110,
            "feature_names_sample": [feature_columns[0], feature_columns[92], feature_columns[165], feature_columns[181]],
        },
        {
            "config_id": "F3_Full182",
            "label": "Full-182 (All)",
            "description": "All 182 Elliptic++ features (Local-93 + Aggregate-72 + Domain-17, indices 0..181)",
            "feature_indices": list(range(0, 182)),
            "in_channels": 182,
            "feature_names_sample": [feature_columns[0], feature_columns[93], feature_columns[164], feature_columns[181]],
        },
    ]


def verify_train_only_scaling(snapshots, cfg):
    """Verifies selected feature columns have zero mean / unit std on Train Steps 1-34."""
    idx = cfg["feature_indices"]
    x_tr = torch.cat([s.x[:, idx] for s in snapshots[:34]], dim=0).numpy()
    mean_abs_max = float(np.abs(x_tr.mean(axis=0)).max())
    std_vals = x_tr.std(axis=0)
    std_min = float(std_vals.min())
    std_max = float(std_vals.max())
    assert mean_abs_max < 1e-2, f"Train mean drift too large for {cfg['config_id']}: {mean_abs_max}"
    assert 0.98 < std_min and std_max < 1.02, f"Train std mismatch for {cfg['config_id']}: [{std_min}, {std_max}]"
    return {"train_mean_abs_max": round(mean_abs_max, 6), "train_std_min": round(std_min, 6), "train_std_max": round(std_max, 6)}


def rollout_val_only(model, snapshots_dev, adjs_dev, idx_tensor):
    """Runs deterministic rollout over Steps 1-34 warm-up and Validation Steps 35-41 ONLY."""
    model.eval()
    val_probs = []
    with torch.no_grad():
        state = model.init_state()
        for t in range(34):
            s = snapshots_dev[t]
            _, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), adjs_dev[t], state)
        for t in range(34, 41):
            s = snapshots_dev[t]
            logits, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), adjs_dev[t], state)
            probs = torch.sigmoid(logits).cpu().numpy()
            lm = s.labeled_mask.cpu().numpy()
            val_probs.extend(probs[lm])
    return np.array(val_probs)


def rollout_test_post_selection(model, snapshots_dev, adjs_dev, idx_tensor):
    """Strictly post-selection rollout on Test Steps 42-49 AFTER all selections finalize."""
    model.eval()
    test_probs = []
    step_outputs = {}
    with torch.no_grad():
        state = model.init_state()
        for t in range(41):
            s = snapshots_dev[t]
            _, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), adjs_dev[t], state)
        t0 = time.perf_counter()
        for t in range(41, 49):
            s = snapshots_dev[t]
            logits, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), adjs_dev[t], state)
            probs = torch.sigmoid(logits).cpu().numpy()
            lm = s.labeled_mask.cpu().numpy()
            y_np = s.y.cpu().numpy()
            lab_p = probs[lm]
            lab_y = y_np[lm]
            test_probs.extend(lab_p)
            step_outputs[int(s.timestep)] = {"probs": lab_p, "y": lab_y}
        inf_time = time.perf_counter() - t0
    return np.array(test_probs), step_outputs, inf_time


def train_single_ablation_run(
    cfg,
    seed,
    snapshots_dev,
    adjs_dev,
    y_val_all,
    pos_weight_val,
    device,
    max_epochs=50,
    patience=15,
    smoke_test=False,
):
    """Trains one (feature_config, seed) GCNAvg model strictly on Steps 1-34 and validates on 35-41."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx_tensor = torch.tensor(cfg["feature_indices"], dtype=torch.long, device=device)
    model = SeparatedGraphNet(
        in_channels=cfg["in_channels"],
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        graph_mode="gcn_avg",
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    grad_clip = 2.0

    best_val_prauc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    max_grad_norm_seen = 0.0
    total_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        state = model.init_state()
        total_loss = 0.0
        loss_steps = 0

        for t in range(34):
            s = snapshots_dev[t]
            logits_t, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), adjs_dev[t], state)
            if s.labeled_mask.sum() > 0:
                loss_t = criterion(logits_t[s.labeled_mask], s.y[s.labeled_mask].float())
                total_loss = total_loss + loss_t
                loss_steps += 1

        avg_loss = total_loss / max(loss_steps, 1)
        avg_loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip))
        max_grad_norm_seen = max(max_grad_norm_seen, pre_clip)
        optimizer.step()

        y_val_prob_ep = rollout_val_only(model, snapshots_dev, adjs_dev, idx_tensor)
        val_prauc = float(average_precision_score(y_val_all, y_val_prob_ep))
        scheduler.step(val_prauc)

        if val_prauc > best_val_prauc or best_state is None:
            best_val_prauc = val_prauc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == max_epochs:
            print(
                f"    [{cfg['config_id']} | seed={seed}] Ep {epoch:02d}/{max_epochs} | "
                f"Loss={avg_loss.item():.4f} | Val PR-AUC={val_prauc:.4f} | Grad={pre_clip:.2f}"
            )

        if patience_counter >= patience and not smoke_test:
            print(f"    Early stop at epoch {epoch} (Best Ep: {best_epoch}, Best Val PR-AUC: {best_val_prauc:.4f})")
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    y_val_prob_best = rollout_val_only(model, snapshots_dev, adjs_dev, idx_tensor)

    tau_orig, _, tau_rec30_orig = select_thresholds_val_only(y_val_all, y_val_prob_best, 0.10, 0.90, 81, 0.30)
    tau_rec, _, tau_rec30_rec = select_thresholds_val_only(y_val_all, y_val_prob_best, 0.01, 0.99, 99, 0.30)

    val_metrics_recal = compute_metrics_and_calibration(y_val_all, y_val_prob_best, threshold=tau_rec)
    val_metrics_legacy = compute_metrics_and_calibration(y_val_all, y_val_prob_best, threshold=tau_orig)

    return {
        "config_id": cfg["config_id"],
        "label": cfg["label"],
        "in_channels": cfg["in_channels"],
        "feature_indices": cfg["feature_indices"],
        "seed": int(seed),
        "total_parameters": total_params,
        "best_epoch": int(best_epoch),
        "training_time_seconds": round(train_time, 3),
        "max_pre_clip_grad_norm": round(max_grad_norm_seen, 3),
        "threshold_legacy_01_90": tau_orig,
        "threshold_recal_01_99": tau_rec,
        "threshold_rec30_recal": tau_rec30_rec,
        "validation_metrics_legacy": val_metrics_legacy,
        "validation_metrics_recal": val_metrics_recal,
        "best_state": best_state,
    }


def evaluate_ablation_run_on_test(run_record, snapshots_dev, adjs_dev, y_test_all, device):
    """Post-selection evaluation on Test Steps 42-49 using pre-selected best_state and Val tau*."""
    idx_tensor = torch.tensor(run_record["feature_indices"], dtype=torch.long, device=device)
    model = SeparatedGraphNet(
        in_channels=run_record["in_channels"],
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        graph_mode="gcn_avg",
    ).to(device)
    model.load_state_dict(run_record["best_state"])

    test_probs, step_outputs, inf_time = rollout_test_post_selection(model, snapshots_dev, adjs_dev, idx_tensor)
    test_recal = compute_metrics_and_calibration(y_test_all, test_probs, threshold=run_record["threshold_recal_01_99"])
    test_legacy = compute_metrics_and_calibration(y_test_all, test_probs, threshold=run_record["threshold_legacy_01_90"])

    per_step = {}
    for step_num, s_info in step_outputs.items():
        ms = compute_metrics_and_calibration(s_info["y"], s_info["probs"], threshold=run_record["threshold_recal_01_99"])
        per_step[str(step_num)] = {
            "pr_auc": ms["pr_auc"],
            "roc_auc": ms["roc_auc"],
            "illicit_f1": ms["illicit_f1"],
            "illicit_precision": ms["illicit_precision"],
            "illicit_recall": ms["illicit_recall"],
            "tp": ms["confusion_matrix"]["true_positives_caught_fraud"],
            "fp": ms["confusion_matrix"]["false_positives_false_alarm"],
            "fn": ms["confusion_matrix"]["false_negatives_missed_fraud"],
            "tn": ms["confusion_matrix"]["true_negatives_licit"],
        }

    return {
        "inference_time_seconds": round(inf_time, 4),
        "test_metrics_legacy_01_90": test_legacy,
        "test_metrics_recal_01_99": test_recal,
        "per_snapshot_test_metrics": per_step,
    }


def summarize_seed_metrics(runs, metrics_key):
    """Computes mean +- std (ddof=0) across seeds for ranking, F1, precision, recall, Brier, ECE."""
    def stats(arr):
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
        }

    m_list = [r[metrics_key] for r in runs]
    return {
        "pr_auc": stats([m["pr_auc"] for m in m_list]),
        "roc_auc": stats([m["roc_auc"] for m in m_list]),
        "illicit_f1": stats([m["illicit_f1"] for m in m_list]),
        "illicit_precision": stats([m["illicit_precision"] for m in m_list]),
        "illicit_recall": stats([m["illicit_recall"] for m in m_list]),
        "brier_score": stats([m["brier_score"] for m in m_list]),
        "ece_10bin": stats([m["ece_10bin"] for m in m_list]),
        "best_epoch": stats([r["best_epoch"] for r in runs]),
    }


def save_ablation_plots(config_summaries, results_dir):
    """Generates multi-seed feature-ablation comparison plots (Val & Test mean +- std)."""
    labels = [c["label"] for c in config_summaries]
    x = np.arange(len(labels))
    w = 0.25

    val_pr = [c["validation_summary_recal"]["pr_auc"]["mean"] for c in config_summaries]
    val_pr_err = [c["validation_summary_recal"]["pr_auc"]["std"] for c in config_summaries]
    test_pr = [c["test_summary_recal"]["pr_auc"]["mean"] for c in config_summaries]
    test_pr_err = [c["test_summary_recal"]["pr_auc"]["std"] for c in config_summaries]
    test_roc = [c["test_summary_recal"]["roc_auc"]["mean"] for c in config_summaries]
    test_roc_err = [c["test_summary_recal"]["roc_auc"]["std"] for c in config_summaries]
    test_f1 = [c["test_summary_recal"]["illicit_f1"]["mean"] for c in config_summaries]
    test_f1_err = [c["test_summary_recal"]["illicit_f1"]["std"] for c in config_summaries]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].bar(x - w / 2, val_pr, w, yerr=val_pr_err, capsize=5, label="Val PR-AUC (35-41)", color="#2b6cb0", edgecolor="black")
    axes[0].bar(x + w / 2, test_pr, w, yerr=test_pr_err, capsize=5, label="Test PR-AUC (42-49)", color="#38a169", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("PR-AUC (Mean +/- Std across 3 seeds)")
    axes[0].set_title("Stage 4E Feature Ablation: Ranking Performance (GCNAvg)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x - w / 2, test_f1, w, yerr=test_f1_err, capsize=5, label="Test F1 (tau* from Val)", color="#dd6b20", edgecolor="black")
    axes[1].bar(x + w / 2, test_roc, w, yerr=test_roc_err, capsize=5, label="Test ROC-AUC", color="#805ad5", edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score (Mean +/- Std across 3 seeds)")
    axes[1].set_title("Stage 4E Test F1 & ROC-AUC Across Feature Subsets")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "stage4e_feature_ablation.png")
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"  [Saved Plot]: {plot_path}")


def generate_ablation_report(payload):
    """Generates Stage 4E controlled feature-ablation Markdown report."""
    cfgs = payload["config_summaries"]
    deltas = payload["attribution_deltas"]
    winner = payload["validation_winner"]

    md = f"""# Stage 4E: Controlled Feature-Group Ablation Report (`GCNAvg`)

**Generated On:** {payload['timestamp']}  
**Architecture Held Fixed:** `SeparatedGraphNet(hidden=128, out=64, dropout=0.3, graph_mode="gcn_avg")`, `gcn_directed`  
**Training Protocol:** Train `Steps 1–34` (`Adam lr=0.002, wd=1e-3, grad_clip=2.0`, weighted BCE `pos_weight={payload['train_pos_weight']:.4f}`) $\\to$ Validate/Select `Steps 35–41` $\\to$ Post-Selection Test `Steps 42–49`  
**Seeds:** `{payload['seeds']}` | **Max Epochs / Patience:** `{payload['max_epochs']}` / `{payload['patience']}`  
**Validation-Selected Winner:** `{winner['config_id']}` (Best Seed `{winner['best_seed']}`, Mean Val PR-AUC = `{winner['val_prauc_mean']:.4f} ± {winner['val_prauc_std']:.4f}`)

---

## 1. Validation Performance Across Seeds (`Steps 35–41`, Mean ± Std)

| Config | Features | Params | Best Ep (Mean ± Std) | Val PR-AUC | Val ROC-AUC | Val F1 (`0.01–0.99`) | Val Prec | Val Rec | Val Brier | Val ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for c in cfgs:
        vs = c["validation_summary_recal"]
        md += (
            f"| `{c['config_id']}` | `{c['in_channels']}` | `{c['total_parameters']:,}` | "
            f"{vs['best_epoch']['mean']:.1f} ± {vs['best_epoch']['std']:.1f} | "
            f"**{vs['pr_auc']['mean']:.4f} ± {vs['pr_auc']['std']:.4f}** | "
            f"{vs['roc_auc']['mean']:.4f} ± {vs['roc_auc']['std']:.4f} | "
            f"{vs['illicit_f1']['mean']:.4f} ± {vs['illicit_f1']['std']:.4f} | "
            f"{vs['illicit_precision']['mean']*100:.2f}% ± {vs['illicit_precision']['std']*100:.2f}% | "
            f"{vs['illicit_recall']['mean']*100:.2f}% ± {vs['illicit_recall']['std']*100:.2f}% | "
            f"{vs['brier_score']['mean']:.4f} ± {vs['brier_score']['std']:.4f} | "
            f"{vs['ece_10bin']['mean']:.4f} ± {vs['ece_10bin']['std']:.4f} |\n"
        )

    md += "\n---\n\n## 2. Post-Selection Test Performance Across Seeds (`Steps 42–49`, `8,841` Nodes, Mean ± Std)\n\n"
    md += "| Config | Features | Test PR-AUC | Test ROC-AUC | Test F1 (`0.01–0.99`) | Test Prec | Test Rec | Test Brier | Test ECE |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for c in cfgs:
        ts = c["test_summary_recal"]
        md += (
            f"| `{c['config_id']}` | `{c['in_channels']}` | "
            f"**{ts['pr_auc']['mean']:.4f} ± {ts['pr_auc']['std']:.4f}** | "
            f"{ts['roc_auc']['mean']:.4f} ± {ts['roc_auc']['std']:.4f} | "
            f"**{ts['illicit_f1']['mean']:.4f} ± {ts['illicit_f1']['std']:.4f}** | "
            f"{ts['illicit_precision']['mean']*100:.2f}% ± {ts['illicit_precision']['std']*100:.2f}% | "
            f"{ts['illicit_recall']['mean']*100:.2f}% ± {ts['illicit_recall']['std']*100:.2f}% | "
            f"{ts['brier_score']['mean']:.4f} ± {ts['brier_score']['std']:.4f} | "
            f"{ts['ece_10bin']['mean']:.4f} ± {ts['ece_10bin']['std']:.4f} |\n"
        )

    md += "\n---\n\n## 3. Per-Seed Runs (Parameters, Best Epochs, Validation & Post-Selection Test)\n\n"
    md += "| Config | Seed | Params | Best Ep | Val PR-AUC | Val $\\tau^*$ | Test PR-AUC | Test ROC-AUC | Test F1 | Test Prec | Test Rec | Test Brier | Test ECE | TP / FP / TN / FN |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in payload["individual_runs"]:
        vr = r["validation_metrics_recal"]
        tr = r["test_metrics_recal_01_99"]
        cm = tr["confusion_matrix"]
        md += (
            f"| `{r['config_id']}` | {r['seed']} | `{r['total_parameters']:,}` | {r['best_epoch']} | "
            f"{vr['pr_auc']:.4f} | `{r['threshold_recal_01_99']:.2f}` | {tr['pr_auc']:.4f} | {tr['roc_auc']:.4f} | "
            f"{tr['illicit_f1']:.4f} | {tr['illicit_precision']*100:.2f}% | {tr['illicit_recall']*100:.2f}% | "
            f"{tr['brier_score']:.4f} | {tr['ece_10bin']:.4f} | "
            f"{cm['true_positives_caught_fraud']}/{cm['false_positives_false_alarm']}/{cm['true_negatives_licit']}/{cm['false_negatives_missed_fraud']} |\n"
        )

    md += "\n---\n\n## 4. Isolated Feature-Group Attribution Deltas (Mean Across Seeds)\n\n"
    md += "| Comparison | Isolated Factor | $\\Delta$ Val PR-AUC | $\\Delta$ Test PR-AUC | $\\Delta$ Test ROC-AUC | $\\Delta$ Test F1 | $\\Delta$ Test Brier | $\\Delta$ Test ECE |\n"
    md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for key, d in deltas.items():
        md += (
            f"| `{d['comparison']}` | {d['isolated_factor']} | "
            f"`{d['delta_val_pr_auc']:+.4f}` | `{d['delta_test_pr_auc']:+.4f}` | "
            f"`{d['delta_test_roc_auc']:+.4f}` | `{d['delta_test_f1']:+.4f}` | "
            f"`{d['delta_test_brier']:+.4f}` | `{d['delta_test_ece']:+.4f}` |\n"
        )
    return md


def run_stage4e(smoke_test=False):
    print("=" * 88)
    mode_str = " (SMOKE TEST MODE)" if smoke_test else ""
    print(f"STAGE 4E: CONTROLLED FEATURE-GROUP ABLATION (GCNAvg){mode_str}")
    print("=" * 88)
    t_start = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "processed")
    models_dir = os.path.join(base_dir, "models", "stage4e")
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")

    meta_path = os.path.join(processed_dir, "metadata.json")
    snaps_path = os.path.join(processed_dir, "temporal_snapshots.pt")
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    snapshots = torch.load(snaps_path, weights_only=False)
    assert len(snapshots) == 49

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = build_ablation_configs(metadata["feature_columns"])
    scaling_checks = {}
    for cfg in configs:
        scaling_checks[cfg["config_id"]] = verify_train_only_scaling(snapshots, cfg)
        print(f"[Train-Only Scaler Check PASS] {cfg['config_id']} ({cfg['in_channels']} cols): {scaling_checks[cfg['config_id']]}")

    # Strict training-only pos_weight (Steps 1-34)
    train_labels = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34)])
    n_lic = int((train_labels == 0).sum())
    n_ill = int((train_labels == 1).sum())
    pos_weight_val = n_lic / max(n_ill, 1)
    y_val_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34, 41)])
    print(f"Train (1-34) pos_weight={pos_weight_val:.4f} | Val (35-41) N={len(y_val_all):,} | Test (42-49) SEALED.")

    snapshots_dev = [s.to(device) for s in snapshots]
    adjs_dev = [build_gcn_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device) for s in snapshots]

    max_epochs = 2 if smoke_test else 50
    patience = 15

    # Phase 1: Train & Validate (Steps 1-34 train, 35-41 validation/thresholding only; Test sealed)
    all_runs = []
    runs_by_cfg = {}
    for cfg in configs:
        print(f"\n[Phase 1 Train/Val] {cfg['config_id']} ({cfg['in_channels']} features)")
        cfg_runs = []
        for seed in SEEDS:
            run_rec = train_single_ablation_run(
                cfg=cfg,
                seed=seed,
                snapshots_dev=snapshots_dev,
                adjs_dev=adjs_dev,
                y_val_all=y_val_all,
                pos_weight_val=pos_weight_val,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                smoke_test=smoke_test,
            )
            cfg_runs.append(run_rec)
            all_runs.append(run_rec)
        runs_by_cfg[cfg["config_id"]] = cfg_runs

    # Phase 2: Validation-only configuration and best-seed selection (BEFORE touching Test)
    val_summaries = {}
    for cid, cfg_runs in runs_by_cfg.items():
        val_summaries[cid] = summarize_seed_metrics(cfg_runs, "validation_metrics_recal")

    winner_cfg_id = max(
        runs_by_cfg.keys(),
        key=lambda cid: (val_summaries[cid]["pr_auc"]["mean"], val_summaries[cid]["illicit_f1"]["mean"]),
    )
    winner_best_run = max(
        runs_by_cfg[winner_cfg_id],
        key=lambda r: r["validation_metrics_recal"]["pr_auc"] or 0.0,
    )
    validation_winner = {
        "config_id": winner_cfg_id,
        "best_seed": int(winner_best_run["seed"]),
        "val_prauc_mean": val_summaries[winner_cfg_id]["pr_auc"]["mean"],
        "val_prauc_std": val_summaries[winner_cfg_id]["pr_auc"]["std"],
        "best_seed_val_prauc": winner_best_run["validation_metrics_recal"]["pr_auc"],
    }
    print("\n" + "=" * 88)
    print(
        f"[Validation-Only Selection Finalized] Winner={validation_winner['config_id']} "
        f"(Mean Val PR-AUC={validation_winner['val_prauc_mean']:.4f}, Best Seed={validation_winner['best_seed']})"
    )
    print("=" * 88)

    # Phase 3: Strictly Post-Selection Test Evaluation (Steps 42-49)
    y_test_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(41, 49)])
    for r in all_runs:
        test_res = evaluate_ablation_run_on_test(r, snapshots_dev, adjs_dev, y_test_all, device)
        r.update(test_res)

    config_summaries = []
    summary_map = {}
    for cfg in configs:
        cid = cfg["config_id"]
        cfg_runs = runs_by_cfg[cid]
        val_sum_rec = val_summaries[cid]
        val_sum_leg = summarize_seed_metrics(cfg_runs, "validation_metrics_legacy")
        test_sum_rec = summarize_seed_metrics(cfg_runs, "test_metrics_recal_01_99")
        test_sum_leg = summarize_seed_metrics(cfg_runs, "test_metrics_legacy_01_90")
        best_val_seed_run = max(cfg_runs, key=lambda x: x["validation_metrics_recal"]["pr_auc"] or 0.0)

        entry = {
            "config_id": cid,
            "label": cfg["label"],
            "description": cfg["description"],
            "in_channels": cfg["in_channels"],
            "feature_indices": cfg["feature_indices"],
            "total_parameters": cfg_runs[0]["total_parameters"],
            "seeds": SEEDS,
            "validation_summary_recal": val_sum_rec,
            "validation_summary_legacy": val_sum_leg,
            "test_summary_recal": test_sum_rec,
            "test_summary_legacy": test_sum_leg,
            "best_val_seed": int(best_val_seed_run["seed"]),
            "best_val_seed_test_recal": best_val_seed_run["test_metrics_recal_01_99"],
        }
        config_summaries.append(entry)
        summary_map[cid] = entry
        print(
            f"[Post-Selection Test Summary | {cid}] "
            f"Params={entry['total_parameters']:,} | BestEp={val_sum_rec['best_epoch']['mean']:.1f}±{val_sum_rec['best_epoch']['std']:.1f} | "
            f"Test PR-AUC={test_sum_rec['pr_auc']['mean']:.4f}±{test_sum_rec['pr_auc']['std']:.4f} | "
            f"ROC-AUC={test_sum_rec['roc_auc']['mean']:.4f}±{test_sum_rec['roc_auc']['std']:.4f} | "
            f"F1={test_sum_rec['illicit_f1']['mean']:.4f}±{test_sum_rec['illicit_f1']['std']:.4f}"
        )

    def make_delta(name, rhs_id, lhs_id, factor_desc):
        r_v, l_v = summary_map[rhs_id]["validation_summary_recal"], summary_map[lhs_id]["validation_summary_recal"]
        r_t, l_t = summary_map[rhs_id]["test_summary_recal"], summary_map[lhs_id]["test_summary_recal"]
        return {
            "comparison": f"{rhs_id} - {lhs_id}",
            "isolated_factor": factor_desc,
            "delta_val_pr_auc": round(r_v["pr_auc"]["mean"] - l_v["pr_auc"]["mean"], 4),
            "delta_test_pr_auc": round(r_t["pr_auc"]["mean"] - l_t["pr_auc"]["mean"], 4),
            "delta_test_roc_auc": round(r_t["roc_auc"]["mean"] - l_t["roc_auc"]["mean"], 4),
            "delta_test_f1": round(r_t["illicit_f1"]["mean"] - l_t["illicit_f1"]["mean"], 4),
            "delta_test_precision": round(r_t["illicit_precision"]["mean"] - l_t["illicit_precision"]["mean"], 4),
            "delta_test_recall": round(r_t["illicit_recall"]["mean"] - l_t["illicit_recall"]["mean"], 4),
            "delta_test_brier": round(r_t["brier_score"]["mean"] - l_t["brier_score"]["mean"], 4),
            "delta_test_ece": round(r_t["ece_10bin"]["mean"] - l_t["ece_10bin"]["mean"], 4),
        }

    attribution_deltas = {
        "domain17_effect": make_delta(
            "domain17_effect", "F2_Local93_Domain17", "F1_Local93", "Adding Domain-17 (indices 165..181) to Local-93"
        ),
        "aggregate72_effect": make_delta(
            "aggregate72_effect", "F3_Full182", "F2_Local93_Domain17", "Adding Aggregate-72 (indices 93..164) to Local-93 + Domain-17"
        ),
        "combined_effect": make_delta(
            "combined_effect", "F3_Full182", "F1_Local93", "Adding both Domain-17 and Aggregate-72 to Local-93"
        ),
    }

    clean_runs = [{k: v for k, v in r.items() if k != "best_state"} for r in all_runs]
    payload = {
        "stage": "Stage 4E - Controlled Feature-Group Ablation (GCNAvg)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_test": bool(smoke_test),
        "seeds": SEEDS,
        "max_epochs": max_epochs,
        "patience": patience,
        "train_pos_weight": round(float(pos_weight_val), 4),
        "scaling_checks": scaling_checks,
        "validation_winner": validation_winner,
        "config_summaries": config_summaries,
        "attribution_deltas": attribution_deltas,
        "individual_runs": clean_runs,
    }

    if smoke_test:
        print("\n[Smoke Test] Skipping all checkpoint, JSON, report, and plot disk writes to protect Stage 1-4D outputs.")
    else:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        for r in all_runs:
            ckpt_path = os.path.join(models_dir, f"{r['config_id']}_seed{r['seed']}.pt")
            torch.save({
                "config_id": r["config_id"],
                "in_channels": r["in_channels"],
                "feature_indices": r["feature_indices"],
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "total_parameters": r["total_parameters"],
                "model_state_dict": r["best_state"],
                "threshold_recal_01_99": r["threshold_recal_01_99"],
                "threshold_legacy_01_90": r["threshold_legacy_01_90"],
                "validation_metrics_recal": r["validation_metrics_recal"],
                "test_metrics_recal_01_99": r["test_metrics_recal_01_99"],
            }, ckpt_path)

        winner_ckpt_path = os.path.join(models_dir, "ablation_best_gcn_avg.pt")
        torch.save({
            "role": "stage4e_validation_winner",
            "config_id": winner_best_run["config_id"],
            "in_channels": winner_best_run["in_channels"],
            "feature_indices": winner_best_run["feature_indices"],
            "seed": winner_best_run["seed"],
            "best_epoch": winner_best_run["best_epoch"],
            "total_parameters": winner_best_run["total_parameters"],
            "model_state_dict": winner_best_run["best_state"],
            "threshold_recal_01_99": winner_best_run["threshold_recal_01_99"],
            "validation_metrics_recal": winner_best_run["validation_metrics_recal"],
            "test_metrics_recal_01_99": winner_best_run["test_metrics_recal_01_99"],
        }, winner_ckpt_path)
        print(f"  [Saved Stage 4E Winner Checkpoint]: {winner_ckpt_path}")

        save_ablation_plots(config_summaries, results_dir)

        json_path = os.path.join(reports_dir, "stage4e_feature_ablation_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"  [Saved Metrics JSON]: {json_path}")

        md_path = os.path.join(reports_dir, "stage4e_feature_ablation_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_ablation_report(payload))
        print(f"  [Saved Markdown Report]: {md_path}")

    print(f"\nStage 4E completed in {time.time() - t_start:.2f}s.")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
    run_stage4e(smoke_test=is_smoke)
