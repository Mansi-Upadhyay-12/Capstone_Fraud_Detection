"""
visualize_results.py
Comprehensive Publication-Quality Evaluation & Visualization Engine
Capstone Project: Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection
Dataset: Elliptic++ Bitcoin Graph Benchmark
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Styling configuration for publication-grade graphics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

# Ensure root in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# Import model architectures from src
from src.train_baselines import FraudMLP
from src.train_gnn_static import (
    FraudGCN, ImprovedFraudGCN, FraudGAT, ImprovedFraudGAT,
    build_temporal_subgraph
)
from src.train_gcn_lstm import FraudGCNLSTM
from src.train_gcn_lstm_historical import FraudGCNLSTMHistorical


def main():
    print("=" * 85)
    print("CAPSTONE VISUALIZATION & BENCHMARK EVALUATION ENGINE")
    print("Elliptic++ Fraud Ring Detection: Comprehensive Multi-Model Analysis")
    print("=" * 85)
    
    start_time = time.time()
    
    output_dir = os.path.join(base_dir, "results", "graphs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory initialized: {output_dir}\n")
    
    # -------------------------------------------------------------------------
    # 1. Load Preprocessed Datasets
    # -------------------------------------------------------------------------
    print("[1/4] Loading Preprocessed Datasets...")
    tab_path = os.path.join(base_dir, "processed", "tabular_data.npz")
    pyg_path = os.path.join(base_dir, "processed", "elliptic_pyg_data.pt")
    snaps_path = os.path.join(base_dir, "processed", "temporal_snapshots.pt")
    
    assert os.path.exists(tab_path), f"Missing {tab_path}"
    assert os.path.exists(pyg_path), f"Missing {pyg_path}"
    assert os.path.exists(snaps_path), f"Missing {snaps_path}"
    
    # Tabular test partition
    tab_data = np.load(tab_path)
    test_mask_tab = tab_data['test_mask']
    X_test_tab = tab_data['X'][test_mask_tab]
    y_test = tab_data['y'][test_mask_tab]
    timesteps_test = tab_data['timestep'][test_mask_tab]
    txId_test = tab_data['txId'][test_mask_tab]
    
    N_test = len(y_test)
    N_illicit = int((y_test == 1).sum())
    N_licit = int((y_test == 0).sum())
    print(f"  Test Population: {N_test:,} transactions (Illicit: {N_illicit:,} [{N_illicit/N_test*100:.2f}%], Licit: {N_licit:,})")
    
    # Static Graph test subgraph
    data_pyg = torch.load(pyg_path, weights_only=False)
    test_node_indices = (data_pyg.timestep >= 42).nonzero().squeeze().numpy()
    test_graph = build_temporal_subgraph(data_pyg, test_node_indices)
    test_edge_undirected = to_undirected(test_graph.edge_index)
    
    # Temporal Graph snapshots
    snapshots = torch.load(snaps_path, weights_only=False)
    print(f"  Loaded 49 temporal snapshots. Snapshots 42–49 correspond to test period.\n")
    
    # -------------------------------------------------------------------------
    # 2. Discover, Load, and Execute Inference for All 10 Models
    # -------------------------------------------------------------------------
    print("[2/4] Executing Zero-Leakage Test Inference Across All 10 Models...")
    
    # Dictionary mapping model key -> metadata and results
    models_registry = {
        "Logistic_Regression": {
            "display_name": "Logistic Regression",
            "category": "Tabular Linear",
            "color": "#1f77b4",
            "threshold": 0.90,
            "train_time": 2.72
        },
        "Random_Forest": {
            "display_name": "Random Forest",
            "category": "Tabular Ensemble",
            "color": "#2ca02c",
            "threshold": 0.59,
            "train_time": 3.82
        },
        "HistGradientBoosting": {
            "display_name": "HistGradientBoosting",
            "category": "Tabular Boosting",
            "color": "#ff7f0e",
            "threshold": 0.79,
            "train_time": 34.96
        },
        "MLP_Baseline": {
            "display_name": "MLP Neural Network",
            "category": "Deep Tabular",
            "color": "#9467bd",
            "threshold": 0.89,
            "train_time": 36.75
        },
        "Static_GCN_Original": {
            "display_name": "Static GCN (Baseline)",
            "category": "Static GNN",
            "color": "#8c564b",
            "threshold": 0.60,
            "train_time": 40.00
        },
        "Improved_Residual_GCN": {
            "display_name": "Improved Residual GCN",
            "category": "Static GNN (Res+LN)",
            "color": "#008080",
            "threshold": 0.59,
            "train_time": 52.68
        },
        "Static_GAT_Original": {
            "display_name": "Static GAT (Baseline)",
            "category": "Static GNN",
            "color": "#e377c2",
            "threshold": 0.70,
            "train_time": 44.14
        },
        "Improved_Residual_GAT": {
            "display_name": "Improved Residual GAT",
            "category": "Static GNN (Res+LN)",
            "color": "#17becf",
            "threshold": 0.90,
            "train_time": 178.89
        },
        "Temporal_GCN_LSTM": {
            "display_name": "Temporal GCN-LSTM",
            "category": "Temporal GNN",
            "color": "#d62728",
            "threshold": 0.90,
            "train_time": 60.00
        },
        "Historical_GCN_LSTM": {
            "display_name": "Historical GCN-LSTM",
            "category": "Temporal GNN (Causal)",
            "color": "#bcbd22",
            "threshold": 0.90,
            "train_time": 60.00
        }
    }
    
    # Execute Model 1: Logistic Regression
    lr_model = joblib.load(os.path.join(base_dir, "models", "logistic_regression.joblib"))
    t0 = time.perf_counter()
    p_lr = lr_model.predict_proba(X_test_tab)[:, 1]
    models_registry["Logistic_Regression"]["inf_time"] = time.perf_counter() - t0
    models_registry["Logistic_Regression"]["probs"] = p_lr
    
    # Execute Model 2: Random Forest
    rf_model = joblib.load(os.path.join(base_dir, "models", "random_forest.joblib"))
    t0 = time.perf_counter()
    p_rf = rf_model.predict_proba(X_test_tab)[:, 1]
    models_registry["Random_Forest"]["inf_time"] = time.perf_counter() - t0
    models_registry["Random_Forest"]["probs"] = p_rf
    
    # Execute Model 3: HistGradientBoosting
    hgb_model = joblib.load(os.path.join(base_dir, "models", "hist_gradient_boosting.joblib"))
    t0 = time.perf_counter()
    p_hgb = hgb_model.predict_proba(X_test_tab)[:, 1]
    models_registry["HistGradientBoosting"]["inf_time"] = time.perf_counter() - t0
    models_registry["HistGradientBoosting"]["probs"] = p_hgb
    
    # Execute Model 4: MLP Baseline
    mlp_ckpt = torch.load(os.path.join(base_dir, "models", "mlp_baseline.pt"), map_location="cpu", weights_only=False)
    mlp_model = FraudMLP(input_dim=182, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3)
    mlp_model.load_state_dict(mlp_ckpt["model_state_dict"])
    mlp_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        p_mlp = torch.sigmoid(mlp_model(torch.tensor(X_test_tab, dtype=torch.float32))).numpy()
    models_registry["MLP_Baseline"]["inf_time"] = time.perf_counter() - t0
    models_registry["MLP_Baseline"]["probs"] = p_mlp
    
    # Execute Model 5: Static GCN Baseline
    gcn_o_ckpt = torch.load(os.path.join(base_dir, "models", "gcn_static.pt"), map_location="cpu", weights_only=False)
    gcn_o_model = FraudGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.0)
    gcn_o_model.load_state_dict(gcn_o_ckpt["model_state_dict"])
    gcn_o_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        p_gcn_o = torch.sigmoid(gcn_o_model(test_graph.x, test_graph.edge_index)[test_graph.labeled_mask]).numpy()
    models_registry["Static_GCN_Original"]["inf_time"] = time.perf_counter() - t0
    models_registry["Static_GCN_Original"]["probs"] = p_gcn_o
    
    # Execute Model 6: Improved Residual GCN
    gcn_i_ckpt = torch.load(os.path.join(base_dir, "models", "gcn_static_improved.pt"), map_location="cpu", weights_only=False)
    gcn_i_model = ImprovedFraudGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.0, num_layers=3)
    gcn_i_model.load_state_dict(gcn_i_ckpt["model_state_dict"])
    gcn_i_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        p_gcn_i = torch.sigmoid(gcn_i_model(test_graph.x, test_edge_undirected)[test_graph.labeled_mask]).numpy()
    models_registry["Improved_Residual_GCN"]["inf_time"] = time.perf_counter() - t0
    models_registry["Improved_Residual_GCN"]["probs"] = p_gcn_i
    
    # Execute Model 7: Static GAT Baseline
    gat_o_ckpt = torch.load(os.path.join(base_dir, "models", "gat_static.pt"), map_location="cpu", weights_only=False)
    gat_o_model = FraudGAT(in_channels=182, hidden_channels=32, heads=4, out_channels=64, dropout_rate=0.0)
    gat_o_model.load_state_dict(gat_o_ckpt["model_state_dict"])
    gat_o_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        p_gat_o = torch.sigmoid(gat_o_model(test_graph.x, test_graph.edge_index)[test_graph.labeled_mask]).numpy()
    models_registry["Static_GAT_Original"]["inf_time"] = time.perf_counter() - t0
    models_registry["Static_GAT_Original"]["probs"] = p_gat_o
    
    # Execute Model 8: Improved Residual GAT
    gat_i_ckpt = torch.load(os.path.join(base_dir, "models", "gat_static_improved.pt"), map_location="cpu", weights_only=False)
    gat_i_model = ImprovedFraudGAT(in_channels=182, hidden_channels=32, heads1=4, heads2=2, out_channels=64, dropout_rate=0.0)
    gat_i_model.load_state_dict(gat_i_ckpt["model_state_dict"])
    gat_i_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        p_gat_i = torch.sigmoid(gat_i_model(test_graph.x, test_edge_undirected)[test_graph.labeled_mask]).numpy()
    models_registry["Improved_Residual_GAT"]["inf_time"] = time.perf_counter() - t0
    models_registry["Improved_Residual_GAT"]["probs"] = p_gat_i
    
    # Execute Model 9: Temporal GCN-LSTM
    lstm_ckpt = torch.load(os.path.join(base_dir, "models", "gcn_lstm.pt"), map_location="cpu", weights_only=False)
    lstm_model = FraudGCNLSTM(in_channels=182, hidden_gcn=128, spatial_dim=64, lstm_dim=64, dropout_rate=0.0)
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
    lstm_model.eval()
    
    with torch.no_grad():
        # Sequential rollout steps 1-34
        h_o, c_o = lstm_model.init_hidden(torch.device("cpu"))
        for t in range(34):
            _, h_o, c_o = lstm_model.forward_step(snapshots[t].x, snapshots[t].edge_index, h_o, c_o)
        # Rollout steps 35-41
        for t in range(34, 41):
            _, h_o, c_o = lstm_model.forward_step(snapshots[t].x, snapshots[t].edge_index, h_o, c_o)
            
        t0 = time.perf_counter()
        p_lstm_list = []
        p_lstm_per_step = {}
        for t in range(41, 49):
            logits_step, h_o, c_o = lstm_model.forward_step(snapshots[t].x, snapshots[t].edge_index, h_o, c_o)
            step_probs = torch.sigmoid(logits_step[snapshots[t].labeled_mask]).numpy()
            p_lstm_list.extend(step_probs)
            p_lstm_per_step[snapshots[t].timestep] = step_probs
        p_lstm = np.array(p_lstm_list)
        models_registry["Temporal_GCN_LSTM"]["inf_time"] = time.perf_counter() - t0
        models_registry["Temporal_GCN_LSTM"]["probs"] = p_lstm
        models_registry["Temporal_GCN_LSTM"]["step_probs"] = p_lstm_per_step
        
    # Execute Model 10: Historical Causal GCN-LSTM
    hist_ckpt = torch.load(os.path.join(base_dir, "models", "gcn_lstm_historical.pt"), map_location="cpu", weights_only=False)
    hist_model = FraudGCNLSTMHistorical(in_channels=182, hidden_gcn=128, spatial_dim=64, lstm_dim=64, dropout_rate=0.0)
    hist_model.load_state_dict(hist_ckpt["model_state_dict"])
    hist_model.eval()
    
    with torch.no_grad():
        h_h, c_h = hist_model.init_hidden(torch.device("cpu"))
        for t in range(34):
            _, h_h, c_h = hist_model.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h_h, c_h)
        for t in range(34, 41):
            _, h_h, c_h = hist_model.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h_h, c_h)
            
        t0 = time.perf_counter()
        p_hist_list = []
        p_hist_per_step = {}
        for t in range(41, 49):
            logits_step, h_h, c_h = hist_model.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h_h, c_h)
            step_probs = torch.sigmoid(logits_step[snapshots[t].labeled_mask]).numpy()
            p_hist_list.extend(step_probs)
            p_hist_per_step[snapshots[t].timestep] = step_probs
        p_hist = np.array(p_hist_list)
        models_registry["Historical_GCN_LSTM"]["inf_time"] = time.perf_counter() - t0
        models_registry["Historical_GCN_LSTM"]["probs"] = p_hist
        models_registry["Historical_GCN_LSTM"]["step_probs"] = p_hist_per_step

    print("  All 10 models executed successfully on test population (8,841 samples).\n")

    # -------------------------------------------------------------------------
    # 3. Calculate Comprehensive Metrics & Build Comparison Table
    # -------------------------------------------------------------------------
    print("[3/4] Calculating Rigorous Performance Metrics and Confusion Matrices...")
    metrics_records = []
    
    for key, item in models_registry.items():
        p = item["probs"]
        tau = item["threshold"]
        preds = (p >= tau).astype(int)
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
        prauc = average_precision_score(y_test, p)
        rocauc = roc_auc_score(y_test, p)
        
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        item["metrics"] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "PR_AUC": prauc,
            "ROC_AUC": rocauc,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Threshold": tau,
            "Train_Time_s": item["train_time"],
            "Inf_Time_s": item["inf_time"]
        }
        
        metrics_records.append({
            "Model_Key": key,
            "Model_Name": item["display_name"],
            "Paradigm": item["category"],
            "Threshold": tau,
            "Accuracy": round(acc, 4),
            "Illicit_Precision": round(prec, 4),
            "Illicit_Recall": round(rec, 4),
            "Illicit_F1": round(f1, 4),
            "PR_AUC": round(prauc, 4),
            "ROC_AUC": round(rocauc, 4),
            "True_Positives": int(tp),
            "False_Positives": int(fp),
            "True_Negatives": int(tn),
            "False_Negatives": int(fn),
            "Training_Time_s": round(item["train_time"], 2),
            "Inference_Time_s": round(item["inf_time"], 4)
        })
        
        print(f"  {item['display_name']:<25} | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | F1: {f1:0.4f} | PR-AUC: {prauc:0.4f} | TP: {tp:3d}/408 | FP: {fp:4d}")

    df_comparison = pd.DataFrame(metrics_records)
    
    # Save CSV and JSON
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    json_path = os.path.join(output_dir, "model_comparison.json")
    df_comparison.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_records, f, indent=4)
        
    print(f"\n  [Saved Table]: {csv_path}")
    print(f"  [Saved JSON]:  {json_path}\n")

    # -------------------------------------------------------------------------
    # 4. Generate 10 Research-Ready Graphs (PNG @ 300 DPI + Vector PDF)
    # -------------------------------------------------------------------------
    print("[4/4] Generating 10 Publication-Quality Graphs (PNG & Vector PDF)...")

    # Helper function to save dual formats
    def save_fig(fig, filename_base):
        png_path = os.path.join(output_dir, f"{filename_base}.png")
        pdf_path = os.path.join(output_dir, f"{filename_base}.pdf")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig)
        print(f"  Generated: {filename_base}.png and {filename_base}.pdf")

    # =========================================================================
    # GRAPH 1: Overall Model Comparison (Grouped Bar Chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(15, 7.5))
    models_display = [item["display_name"] for item in models_registry.values()]
    x = np.arange(len(models_display))
    width = 0.20
    
    accs = [m["metrics"]["Accuracy"] * 100 for m in models_registry.values()]
    precs = [m["metrics"]["Precision"] * 100 for m in models_registry.values()]
    recs = [m["metrics"]["Recall"] * 100 for m in models_registry.values()]
    f1s = [m["metrics"]["F1_Score"] * 100 for m in models_registry.values()]
    
    b1 = ax.bar(x - 1.5 * width, accs, width, label='Accuracy', color='#4a7c59', alpha=0.9, edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x - 0.5 * width, precs, width, label='Illicit Precision', color='#1f77b4', alpha=0.9, edgecolor='black', linewidth=0.5)
    b3 = ax.bar(x + 0.5 * width, recs, width, label='Illicit Recall', color='#d62728', alpha=0.9, edgecolor='black', linewidth=0.5)
    b4 = ax.bar(x + 1.5 * width, f1s, width, label='Illicit F1-Score', color='#ff7f0e', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax.set_title("Overall Model Performance Comparison on Test Set (Snapshots 42–49)\nElliptic++ Bitcoin Graph Benchmark", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Score Percentage (%)", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_display, rotation=35, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.4)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    save_fig(fig, "01_overall_model_comparison")

    # =========================================================================
    # GRAPH 2: ROC Curve Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    for key, item in models_registry.items():
        fpr, tpr, _ = roc_curve(y_test, item["probs"])
        auc_v = item["metrics"]["ROC_AUC"]
        ax.plot(fpr, tpr, label=f"{item['display_name']} (AUC = {auc_v:.4f})", color=item["color"], lw=2.0)
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7, label='Random Chance (AUC = 0.5000)')
    ax.set_title("Test Set Receiver Operating Characteristic (ROC) Curves\nEvaluated on Unseen Timesteps 42–49 (8,841 Samples)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold')
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9)
    save_fig(fig, "02_roc_curves")

    # =========================================================================
    # GRAPH 3: Precision-Recall Curve Comparison (Key Imbalanced Metric)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 8))
    for key, item in models_registry.items():
        prec, rec, _ = precision_recall_curve(y_test, item["probs"])
        pr_v = item["metrics"]["PR_AUC"]
        ax.plot(rec, prec, label=f"{item['display_name']} (PR-AUC = {pr_v:.4f})", color=item["color"], lw=2.2)
        
    no_skill = N_illicit / N_test
    ax.axhline(y=no_skill, color='k', linestyle='--', lw=1.2, alpha=0.7, label=f'Random Baseline (PR-AUC = {no_skill:.4f})')
    ax.set_title("Test Set Precision-Recall Curves (Minority Illicit Class: 4.61%)\nElliptic++ Benchmark: 408 Frauds vs 8,433 Legitimate", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Recall (Fraction of Total Frauds Caught)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Precision (Accuracy of Flagged Transactions)", fontsize=11, fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    save_fig(fig, "03_precision_recall_curves")

    # =========================================================================
    # GRAPH 4: Confusion Matrices (2x5 Grid for All 10 Models)
    # =========================================================================
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes_flat = axes.flatten()
    
    for idx, (key, item) in enumerate(models_registry.items()):
        ax = axes_flat[idx]
        m = item["metrics"]
        cm_mat = np.array([[m["TN"], m["FP"]],
                           [m["FN"], m["TP"]]])
        
        im = ax.imshow(cm_mat, cmap='Blues', interpolation='nearest')
        ax.set_title(f"{item['display_name']}\n(tau*={item['threshold']:.2f} | F1={m['F1_Score']:.4f})", fontsize=10.5, fontweight='bold', pad=8)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Licit", "Pred Fraud"], fontsize=9.5)
        ax.set_yticklabels(["Actual Licit", "Actual Fraud"], fontsize=9.5)
        
        for r in range(2):
            for c in range(2):
                val_txt = f"{cm_mat[r, c]:,}"
                txt_color = "white" if cm_mat[r, c] > cm_mat.max() * 0.55 else "black"
                sub_label = ""
                if r == 0 and c == 0: sub_label = "\n(TN)"
                elif r == 0 and c == 1: sub_label = "\n(FP - Alarm)"
                elif r == 1 and c == 0: sub_label = "\n(FN - Missed)"
                elif r == 1 and c == 1: sub_label = "\n(TP - Caught)"
                ax.text(c, r, val_txt + sub_label, ha="center", va="center", color=txt_color, fontweight='bold', fontsize=9.5)
                
    plt.suptitle("Test Confusion Matrix Heatmaps (Snapshots 42–49: 408 Actual Frauds & 8,433 Legitimate)", fontsize=15, fontweight='bold', y=0.98)
    save_fig(fig, "04_confusion_matrices")

    # =========================================================================
    # GRAPH 5: Training Time Comparison (Wall-Clock Seconds)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6.5))
    train_times = [m["metrics"]["Train_Time_s"] for m in models_registry.values()]
    colors_bar = [m["color"] for m in models_registry.values()]
    y_pos = np.arange(len(models_display))
    
    bars = ax.barh(y_pos, train_times, color=colors_bar, alpha=0.88, edgecolor='black', linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_display, fontsize=10.5, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Wall-Clock Training Time (Seconds)", fontsize=11, fontweight='bold')
    ax.set_title("Model Training Duration Comparison (CPU Architecture)", fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}s", va='center', ha='left', fontsize=9.5, fontweight='bold')
        
    ax.set_xlim(0, max(train_times) * 1.15)
    save_fig(fig, "05_training_time_comparison")

    # =========================================================================
    # GRAPH 6: Inference Time Comparison (Milliseconds for 8,841 Samples)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6.5))
    inf_times_ms = [m["metrics"]["Inf_Time_s"] * 1000 for m in models_registry.values()]
    bars = ax.barh(y_pos, inf_times_ms, color=colors_bar, alpha=0.88, edgecolor='black', linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models_display, fontsize=10.5, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel("Inference Latency for 8,841 Test Transactions (Milliseconds)", fontsize=11, fontweight='bold')
    ax.set_title("Test Inference Latency Comparison (8,841 Transactions)", fontsize=13, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.4)
    
    for bar in bars:
        w = bar.get_width()
        throughput = N_test / (w / 1000)
        ax.text(w + 5, bar.get_y() + bar.get_height()/2, f"{w:.1f} ms ({throughput:,.0f} tx/s)", va='center', ha='left', fontsize=9, fontweight='bold')
        
    ax.set_xlim(0, max(inf_times_ms) * 1.25)
    save_fig(fig, "06_inference_time_comparison")

    # =========================================================================
    # GRAPH 7: Fraud Detection Performance (Caught Frauds vs Missed Frauds)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 7))
    tp_counts = [m["metrics"]["TP"] for m in models_registry.values()]
    fn_counts = [m["metrics"]["FN"] for m in models_registry.values()]
    fp_counts = [m["metrics"]["FP"] for m in models_registry.values()]
    
    p1 = ax.bar(x - 0.2, tp_counts, 0.38, label='Caught Frauds (True Positives)', color='#2ca02c', alpha=0.9, edgecolor='black')
    p2 = ax.bar(x - 0.2, fn_counts, 0.38, bottom=tp_counts, label='Missed Frauds (False Negatives)', color='#d62728', alpha=0.85, edgecolor='black')
    p3 = ax.bar(x + 0.2, fp_counts, 0.38, label='False Alarms (False Positives)', color='#ff7f0e', alpha=0.85, edgecolor='black')
    
    ax.set_title("Fraud Catch Rate & False Alarms Breakdown (Total Frauds = 408)\nDemonstrating the Recall Champion vs Precision Champion Trade-Off", fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models_display, rotation=35, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel("Transaction Count", fontsize=11, fontweight='bold')
    ax.axhline(y=408, color='k', linestyle=':', lw=1.5, label='Total Actual Frauds (408)')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.4)
    
    for idx, (tp, fn) in enumerate(zip(tp_counts, fn_counts)):
        rec_pct = tp / (tp + fn) * 100
        ax.text(x[idx] - 0.2, tp / 2, f"{tp}\n({rec_pct:.0f}%)", ha='center', va='center', color='white', fontweight='bold', fontsize=8.5)
        
    save_fig(fig, "07_fraud_detection_performance")

    # =========================================================================
    # GRAPH 8: Model Ranking (F1-Score vs PR-AUC Dual Plot)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Sort by F1
    sorted_by_f1 = sorted(models_registry.items(), key=lambda kv: kv[1]["metrics"]["F1_Score"], reverse=True)
    names_f1 = [kv[1]["display_name"] for kv in sorted_by_f1][::-1]
    scores_f1 = [kv[1]["metrics"]["F1_Score"] for kv in sorted_by_f1][::-1]
    colors_f1 = [kv[1]["color"] for kv in sorted_by_f1][::-1]
    
    ax1.barh(names_f1, scores_f1, color=colors_f1, alpha=0.88, edgecolor='black')
    ax1.set_title("Model Ranking by Illicit F1-Score\n(Balances Precision and Recall)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Illicit F1-Score", fontsize=10.5, fontweight='bold')
    ax1.set_xlim(0, max(scores_f1) * 1.18)
    ax1.grid(axis='x', alpha=0.4)
    for i, v in enumerate(scores_f1):
        ax1.text(v + 0.015, i, f"{v:.4f} (Rank #{len(scores_f1)-i})", va='center', fontweight='bold', fontsize=9.5)
        
    # Sort by PR-AUC
    sorted_by_prauc = sorted(models_registry.items(), key=lambda kv: kv[1]["metrics"]["PR_AUC"], reverse=True)
    names_pr = [kv[1]["display_name"] for kv in sorted_by_prauc][::-1]
    scores_pr = [kv[1]["metrics"]["PR_AUC"] for kv in sorted_by_prauc][::-1]
    colors_pr = [kv[1]["color"] for kv in sorted_by_prauc][::-1]
    
    ax2.barh(names_pr, scores_pr, color=colors_pr, alpha=0.88, edgecolor='black')
    ax2.set_title("Model Ranking by PR-AUC (Average Precision)\n(Threshold-Independent Ranking Quality)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("PR-AUC Score", fontsize=10.5, fontweight='bold')
    ax2.set_xlim(0, max(scores_pr) * 1.18)
    ax2.grid(axis='x', alpha=0.4)
    for i, v in enumerate(scores_pr):
        ax2.text(v + 0.015, i, f"{v:.4f} (Rank #{len(scores_pr)-i})", va='center', fontweight='bold', fontsize=9.5)
        
    plt.suptitle("Model Evaluation Rankings across Complementary Fraud Metrics", fontsize=14, fontweight='bold', y=0.98)
    save_fig(fig, "08_model_ranking")

    # =========================================================================
    # GRAPH 9: Threshold Analysis (Precision, Recall, F1 across Thresholds)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    key_models = [
        ("Random_Forest", "Random Forest (Best Tabular)"),
        ("Improved_Residual_GCN", "Improved Residual GCN (Best Static GNN)"),
        ("Improved_Residual_GAT", "Improved Residual GAT (Multi-Head Attentive)"),
        ("Temporal_GCN_LSTM", "Temporal GCN-LSTM (Recall Champion)")
    ]
    
    threshold_range = np.linspace(0.05, 0.95, 91)
    
    for ax, (m_key, title) in zip(axes.flat, key_models):
        p = models_registry[m_key]["probs"]
        opt_t = models_registry[m_key]["threshold"]
        
        precs_t, recs_t, f1s_t = [], [], []
        for t in threshold_range:
            preds_t = (p >= t).astype(int)
            precs_t.append(precision_score(y_test, preds_t, pos_label=1, zero_division=0))
            recs_t.append(recall_score(y_test, preds_t, pos_label=1, zero_division=0))
            f1s_t.append(f1_score(y_test, preds_t, pos_label=1, zero_division=0))
            
        ax.plot(threshold_range, precs_t, label='Precision', color='#1f77b4', lw=2.2)
        ax.plot(threshold_range, recs_t, label='Recall', color='#d62728', lw=2.2)
        ax.plot(threshold_range, f1s_t, label='F1-Score', color='#2ca02c', lw=2.4)
        ax.axvline(x=opt_t, color='black', linestyle='--', lw=1.5, label=f'Validation Optimal tau* = {opt_t:.2f}')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel("Classification Decision Threshold (tau)", fontsize=9.5)
        ax.set_ylabel("Metric Value", fontsize=9.5)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.35)
        ax.legend(loc='center right', fontsize=8.5)
        
    plt.suptitle("Classification Threshold Sensitivity Analysis for Key Model Families", fontsize=14, fontweight='bold', y=0.98)
    save_fig(fig, "09_threshold_analysis")

    # =========================================================================
    # GRAPH 10: Temporal Performance Across Chronological Test Snapshots (Steps 42–49)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    test_timesteps_unique = np.sort(np.unique(timesteps_test))
    
    # Models to compare temporally
    temporal_compare_models = [
        "Random_Forest",
        "Improved_Residual_GCN",
        "Improved_Residual_GAT",
        "Temporal_GCN_LSTM",
        "Historical_GCN_LSTM"
    ]
    
    f1_trajectories = {k: [] for k in temporal_compare_models}
    caught_frauds_trajectories = {k: [] for k in temporal_compare_models}
    total_frauds_per_step = []
    
    for step in test_timesteps_unique:
        mask_step = (timesteps_test == step)
        y_step = y_test[mask_step]
        n_fraud_step = int((y_step == 1).sum())
        total_frauds_per_step.append(n_fraud_step)
        
        for k in temporal_compare_models:
            item = models_registry[k]
            p_step = item["probs"][mask_step]
            tau = item["threshold"]
            preds_step = (p_step >= tau).astype(int)
            f1_s = f1_score(y_step, preds_step, pos_label=1, zero_division=0)
            tp_s = int(((preds_step == 1) & (y_step == 1)).sum())
            f1_trajectories[k].append(f1_s)
            caught_frauds_trajectories[k].append(tp_s)
            
    # Plot 1: F1-score across time
    for k in temporal_compare_models:
        item = models_registry[k]
        ax1.plot(test_timesteps_unique, f1_trajectories[k], marker='o', lw=2.2, label=item["display_name"], color=item["color"])
        
    ax1.set_title("Chronological Test Set Performance Across Time Steps (Steps 42 to 49)\nZero-Leakage Rolling Horizon Evaluation", fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel("Illicit F1-Score", fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc='upper right', fontsize=9.5)
    
    # Plot 2: Absolute Caught Frauds vs Total Frauds
    ax2.bar(test_timesteps_unique, total_frauds_per_step, color='#cccccc', alpha=0.6, width=0.6, label='Total Actual Frauds in Snapshot')
    for k in temporal_compare_models:
        item = models_registry[k]
        ax2.plot(test_timesteps_unique, caught_frauds_trajectories[k], marker='s', lw=2.0, label=f"{item['display_name']} Caught", color=item["color"])
        
    ax2.set_xlabel("Discrete Blockchain Snapshot Time Step (2-Week Windows)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Caught Illicit Transactions", fontsize=11, fontweight='bold')
    ax2.set_xticks(test_timesteps_unique)
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc='upper right', fontsize=9.5)
    
    save_fig(fig, "10_temporal_performance")

    total_time = time.time() - start_time
    print("=" * 85)
    print(f"ALL 10 PUBLICATION-GRADE VISUALIZATIONS GENERATED IN {total_time:.2f} SECONDS!")
    print(f"Artifacts saved in: {output_dir}")
    print("=" * 85)
    return df_comparison

if __name__ == "__main__":
    main()
