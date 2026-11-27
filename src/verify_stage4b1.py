import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

# Import model architectures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from train_gcn_lstm import FraudGCNLSTM
from train_gcn_lstm_historical import FraudGCNLSTMHistorical

def run_verification_audit():
    print("=" * 80)
    print("STAGE 4B.1 COMPREHENSIVE VERIFICATION AUDIT")
    print("=" * 80)

    # 1. Load snapshots
    snapshots_path = "processed/temporal_snapshots.pt"
    snapshots = torch.load(snapshots_path, weights_only=False)
    print(f"Loaded {len(snapshots)} snapshots.")

    # 2. Load checkpoints
    ckpt_orig = torch.load("models/gcn_lstm.pt", map_location="cpu", weights_only=False)
    ckpt_hist = torch.load("models/gcn_lstm_historical.pt", map_location="cpu", weights_only=False)

    in_channels = snapshots[0].x.shape[1]
    
    # Instantiate models
    model_orig = FraudGCNLSTM(in_channels=in_channels, hidden_gcn=128, spatial_dim=64, lstm_dim=64, dropout_rate=0.3)
    model_orig.load_state_dict(ckpt_orig["model_state_dict"])
    model_orig.eval()

    model_hist = FraudGCNLSTMHistorical(in_channels=in_channels, hidden_gcn=128, spatial_dim=64, lstm_dim=64, dropout_rate=0.3)
    model_hist.load_state_dict(ckpt_hist["model_state_dict"])
    model_hist.eval()

    # Compare weights between the two models
    weight_diffs = []
    for k in model_orig.state_dict():
        w1 = model_orig.state_dict()[k]
        w2 = model_hist.state_dict()[k]
        diff = (w1 - w2).abs().max().item()
        weight_diffs.append((k, diff))
    max_weight_diff = max(d for _, d in weight_diffs)
    print(f"\nModel Weight Difference Check: Max parameter difference across all layers = {max_weight_diff:.6f}")

    # Extract true labels
    val_labels = []
    for t in range(34, 41):
        val_labels.extend(snapshots[t].y[snapshots[t].labeled_mask].numpy())
    y_val = np.array(val_labels)

    test_labels = []
    for t in range(41, 49):
        test_labels.extend(snapshots[t].y[snapshots[t].labeled_mask].numpy())
    y_test = np.array(test_labels)

    print(f"Validation labeled nodes: {len(y_val)} (Illicit: {(y_val==1).sum()}, Licit: {(y_val==0).sum()})")
    print(f"Test labeled nodes: {len(y_test)} (Illicit: {(y_test==1).sum()}, Licit: {(y_test==0).sum()})")

    # 3. Deterministic Forward Pass for Original Model
    with torch.no_grad():
        h, c = model_orig.init_hidden(torch.device("cpu"))
        for t in range(34):
            _, h, c = model_orig.forward_step(snapshots[t].x, snapshots[t].edge_index, h, c)
        
        val_probs_orig = []
        for t in range(34, 41):
            logits, h, c = model_orig.forward_step(snapshots[t].x, snapshots[t].edge_index, h, c)
            val_probs_orig.extend(torch.sigmoid(logits[snapshots[t].labeled_mask]).numpy())
        y_val_prob_orig = np.array(val_probs_orig)

        test_probs_orig = []
        for t in range(41, 49):
            logits, h, c = model_orig.forward_step(snapshots[t].x, snapshots[t].edge_index, h, c)
            test_probs_orig.extend(torch.sigmoid(logits[snapshots[t].labeled_mask]).numpy())
        y_test_prob_orig = np.array(test_probs_orig)

    # 4. Deterministic Forward Pass for Historical Model
    with torch.no_grad():
        h, c = model_hist.init_hidden(torch.device("cpu"))
        for t in range(34):
            _, h, c = model_hist.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h, c)
        
        val_probs_hist = []
        for t in range(34, 41):
            logits, h, c = model_hist.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h, c)
            val_probs_hist.extend(torch.sigmoid(logits[snapshots[t].labeled_mask]).numpy())
        y_val_prob_hist = np.array(val_probs_hist)

        test_probs_hist = []
        for t in range(41, 49):
            logits, h, c = model_hist.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h, c)
            test_probs_hist.extend(torch.sigmoid(logits[snapshots[t].labeled_mask]).numpy())
        y_test_prob_hist = np.array(test_probs_hist)

    # 5. Compare Probability Distributions
    prob_diff = y_test_prob_orig - y_test_prob_hist
    max_prob_diff = np.abs(prob_diff).max()
    mean_prob_diff = np.abs(prob_diff).mean()
    num_diff_samples = (np.abs(prob_diff) > 1e-6).sum()

    print("\n" + "-" * 80)
    print("PREDICTION PROBABILITY COMPARISON ON TEST SET (8,841 samples)")
    print("-" * 80)
    print(f"Max absolute prob difference:  {max_prob_diff:.8f}")
    print(f"Mean absolute prob difference: {mean_prob_diff:.8f}")
    print(f"Samples with diff > 1e-6:      {num_diff_samples} / {len(y_test)} ({num_diff_samples/len(y_test)*100:.2f}%)")
    print(f"Original test probs: min={y_test_prob_orig.min():.6f}, max={y_test_prob_orig.max():.6f}, mean={y_test_prob_orig.mean():.6f}")
    print(f"Historical test probs: min={y_test_prob_hist.min():.6f}, max={y_test_prob_hist.max():.6f}, mean={y_test_prob_hist.mean():.6f}")

    # 6. Unrounded Metric Calculations
    prauc_orig_raw = average_precision_score(y_test, y_test_prob_orig)
    prauc_hist_raw = average_precision_score(y_test, y_test_prob_hist)
    roc_orig_raw = roc_auc_score(y_test, y_test_prob_orig)
    roc_hist_raw = roc_auc_score(y_test, y_test_prob_hist)

    print("\n" + "-" * 80)
    print("UNROUNDED PR-AUC & ROC-AUC VERIFICATION")
    print("-" * 80)
    print(f"Original PR-AUC (Raw float64):   {prauc_orig_raw:.10f} -> Rounded (4 dec): {prauc_orig_raw:.4f}")
    print(f"Historical PR-AUC (Raw float64): {prauc_hist_raw:.10f} -> Rounded (4 dec): {prauc_hist_raw:.4f}")
    print(f"Raw PR-AUC difference:           {abs(prauc_orig_raw - prauc_hist_raw):.10f}")
    print(f"Original ROC-AUC (Raw float64):   {roc_orig_raw:.10f} -> Rounded (4 dec): {roc_orig_raw:.4f}")
    print(f"Historical ROC-AUC (Raw float64): {roc_hist_raw:.10f} -> Rounded (4 dec): {roc_hist_raw:.4f}")
    print(f"Raw ROC-AUC difference:           {abs(roc_orig_raw - roc_hist_raw):.10f}")

    # 7. Confusion Matrices and Classification Metrics at tau=0.90
    cm_orig = confusion_matrix(y_test, (y_test_prob_orig >= 0.90).astype(int), labels=[0, 1])
    cm_hist = confusion_matrix(y_test, (y_test_prob_hist >= 0.90).astype(int), labels=[0, 1])

    prec_orig = precision_score(y_test, (y_test_prob_orig >= 0.90).astype(int), pos_label=1)
    rec_orig = recall_score(y_test, (y_test_prob_orig >= 0.90).astype(int), pos_label=1)
    f1_orig = f1_score(y_test, (y_test_prob_orig >= 0.90).astype(int), pos_label=1)

    prec_hist = precision_score(y_test, (y_test_prob_hist >= 0.90).astype(int), pos_label=1)
    rec_hist = recall_score(y_test, (y_test_prob_hist >= 0.90).astype(int), pos_label=1)
    f1_hist = f1_score(y_test, (y_test_prob_hist >= 0.90).astype(int), pos_label=1)

    print("\n" + "-" * 80)
    print("CLASSIFICATION METRICS AT TAU = 0.90")
    print("-" * 80)
    print("Original GCN-LSTM:")
    print(f"  CM: TN={cm_orig[0,0]}, FP={cm_orig[0,1]}, FN={cm_orig[1,0]}, TP={cm_orig[1,1]}")
    print(f"  Precision: {prec_orig:.6f} ({prec_orig:.4f}), Recall: {rec_orig:.6f} ({rec_orig:.4f}), F1: {f1_orig:.6f} ({f1_orig:.4f})")
    print("Historical-Only GCN-LSTM:")
    print(f"  CM: TN={cm_hist[0,0]}, FP={cm_hist[0,1]}, FN={cm_hist[1,0]}, TP={cm_hist[1,1]}")
    print(f"  Precision: {prec_hist:.6f} ({prec_hist:.4f}), Recall: {rec_hist:.6f} ({rec_hist:.4f}), F1: {f1_hist:.6f} ({f1_hist:.4f})")

    # 8. Snapshot 42 In-Depth Audit
    s42 = snapshots[41] # index 41 is Snapshot 42
    total_nodes_42 = s42.x.shape[0]
    labeled_mask_42 = s42.labeled_mask
    num_labeled_42 = labeled_mask_42.sum().item()
    y_42 = s42.y[labeled_mask_42].numpy()
    num_illicit_42 = (y_42 == 1).sum()
    num_licit_42 = (y_42 == 0).sum()
    num_unlabeled_42 = (s42.y == -1).sum().item()

    # Node level probabilities in Snapshot 42 for both models
    with torch.no_grad():
        # Orig forward to step 42
        h_o, c_o = model_orig.init_hidden(torch.device("cpu"))
        for t in range(41):
            _, h_o, c_o = model_orig.forward_step(snapshots[t].x, snapshots[t].edge_index, h_o, c_o)
        logits_o_42, _, _ = model_orig.forward_step(s42.x, s42.edge_index, h_o, c_o)
        probs_o_42_all = torch.sigmoid(logits_o_42).numpy()
        probs_o_42_lab = torch.sigmoid(logits_o_42[labeled_mask_42]).numpy()

        # Hist forward to step 42
        h_h, c_h = model_hist.init_hidden(torch.device("cpu"))
        for t in range(41):
            _, h_h, c_h = model_hist.forward_step_historical(snapshots[t].x, snapshots[t].edge_index, h_h, c_h)
        logits_h_42, _, _ = model_hist.forward_step_historical(s42.x, s42.edge_index, h_h, c_h)
        probs_h_42_all = torch.sigmoid(logits_h_42).numpy()
        probs_h_42_lab = torch.sigmoid(logits_h_42[labeled_mask_42]).numpy()

    print("\n" + "-" * 80)
    print("SNAPSHOT 42 POPULATION & PROBABILITY AUDIT")
    print("-" * 80)
    print(f"Total Nodes in Snapshot 42:          {total_nodes_42:,}")
    print(f"Unlabeled Nodes (Class 3 / y=-1):    {num_unlabeled_42:,}")
    print(f"Labeled Nodes (Classes 1 & 2):       {num_labeled_42:,}")
    print(f"  - Illicit Nodes (Class 1 / y=1):   {num_illicit_42:,} ({num_illicit_42/num_labeled_42*100:.2f}% of labeled)")
    print(f"  - Licit Nodes (Class 2 / y=0):     {num_licit_42:,} ({num_licit_42/num_labeled_42*100:.2f}% of labeled)")
    print("\nOriginal GCN-LSTM in Snapshot 42:")
    print(f"  - ALL NODES ({total_nodes_42:,}):  Mean Prob = {probs_o_42_all.mean():.4f}, Median = {np.median(probs_o_42_all):.4f}, Count >= 0.90 = {(probs_o_42_all >= 0.90).sum():,} ({(probs_o_42_all >= 0.90).mean()*100:.2f}%)")
    print(f"  - LABELED NODES ({num_labeled_42:,}): Mean Prob = {probs_o_42_lab.mean():.4f}, Median = {np.median(probs_o_42_lab):.4f}, Count >= 0.90 = {(probs_o_42_lab >= 0.90).sum():,} ({(probs_o_42_lab >= 0.90).mean()*100:.2f}%)")
    print("\nHistorical GCN-LSTM in Snapshot 42:")
    print(f"  - ALL NODES ({total_nodes_42:,}):  Mean Prob = {probs_h_42_all.mean():.4f}, Median = {np.median(probs_h_42_all):.4f}, Count >= 0.90 = {(probs_h_42_all >= 0.90).sum():,} ({(probs_h_42_all >= 0.90).mean()*100:.2f}%)")
    print(f"  - LABELED NODES ({num_labeled_42:,}): Mean Prob = {probs_h_42_lab.mean():.4f}, Median = {np.median(probs_h_42_lab):.4f}, Count >= 0.90 = {(probs_h_42_lab >= 0.90).sum():,} ({(probs_h_42_lab >= 0.90).mean()*100:.2f}%)")

    # 9. Verify metrics JSON files match exactly
    with open("reports/stage4b_gcn_lstm_metrics.json", "r") as f:
        s4b_json = json.load(f)["GCN_LSTM"]
    with open("reports/stage4b1_gcn_lstm_historical_metrics.json", "r") as f:
        s4b1_json = json.load(f)["GCN_LSTM_Historical"]

    print("\n" + "-" * 80)
    print("VERIFICATION AGAINST SAVED METRICS JSON FILES")
    print("-" * 80)
    print("Saved Stage 4B JSON Test Metrics:")
    print("  Optimal Threshold:", s4b_json["optimal_threshold"])
    print("  Confusion Matrix:", s4b_json["test_metrics_optimal_threshold"]["confusion_matrix"])
    print("  Illicit Prec/Rec/F1:", s4b_json["test_metrics_optimal_threshold"]["illicit_precision"], s4b_json["test_metrics_optimal_threshold"]["illicit_recall"], s4b_json["test_metrics_optimal_threshold"]["illicit_f1"])
    print("  PR-AUC / ROC-AUC:", s4b_json["test_metrics_optimal_threshold"]["pr_auc"], s4b_json["test_metrics_optimal_threshold"]["roc_auc"])

    print("\nSaved Stage 4B.1 JSON Test Metrics:")
    print("  Optimal Threshold:", s4b1_json["optimal_threshold"])
    print("  Confusion Matrix:", s4b1_json["test_metrics_optimal_threshold"]["confusion_matrix"])
    print("  Illicit Prec/Rec/F1:", s4b1_json["test_metrics_optimal_threshold"]["illicit_precision"], s4b1_json["test_metrics_optimal_threshold"]["illicit_recall"], s4b1_json["test_metrics_optimal_threshold"]["illicit_f1"])
    print("  PR-AUC / ROC-AUC:", s4b1_json["test_metrics_optimal_threshold"]["pr_auc"], s4b1_json["test_metrics_optimal_threshold"]["roc_auc"])

if __name__ == "__main__":
    run_verification_audit()
