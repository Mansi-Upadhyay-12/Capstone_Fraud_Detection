# Stage 4C: EvolveGCN-H Controlled Experiments & Benchmark Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** 2026-11-26 10:45:21  
**Model Family:** EvolveGCN-H (Top-$k$ Node Summarization + Matrix-GRU Weight Evolution + LayerNorm)  
**Evaluation Set:** Unseen Test Snapshots 42–49 (8,841 labeled transactions: 408 Illicit / 8,433 Licit)  

---

## 1. Controlled Ablation Study (Directed vs. Undirected & Residual Connections)

All four configurations were trained chronologically on **Steps 1–34**, tuned for early stopping and decision threshold $\tau^*$ strictly on **Validation Steps 35–41** (using **Validation PR-AUC** for architecture selection and checkpointing), and evaluated once on **Test Steps 42–49**:

| Configuration | Undirected | Residual Skip | Best Epoch | Val PR-AUC | $\tau^*$ | Test Prec | Test Rec | Test F1 | Test PR-AUC | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `EvolveGCN_H_Directed` | False | False | 7 | 0.1313 | 0.55 | 5.20% | 29.17% | **0.0883** | **0.0606** | 0.5358 |
| `EvolveGCN_H_Directed_Res` **(Selected)** | False | True | 37 | 0.6387 | 0.90 | 10.63% | 68.63% | **0.1841** | **0.3107** | 0.7963 |
| `EvolveGCN_H_Undirected` | True | False | 2 | 0.1518 | 0.53 | 6.88% | 61.27% | **0.1238** | **0.0560** | 0.6062 |
| `EvolveGCN_H_Undirected_Res` | True | True | 52 | 0.3140 | 0.79 | 1.79% | 0.49% | **0.0077** | **0.0893** | 0.6843 |

### Selected Best Model (by Validation PR-AUC): `EvolveGCN_H_Directed_Res`
- **Validation PR-AUC:** `0.6387` | **Validation Illicit F1:** `0.4810` at $\tau^* = 0.90$
- **Trainable Parameters:** `364,883` | **Training Time:** `212.51s` | **Test Inference Time:** `0.1698s`
- **Test Confusion Matrix ($\tau^* = 0.90$):**
  - True Negatives (Licit): **6,080** / 8,433
  - False Positives (False Alarms): **2,353**
  - False Negatives (Missed Frauds): **128** / 408
  - True Positives (Caught Frauds): **280** / 408
- **Recall-Oriented Operating Point ($\tau = 0.83$, Val Prec $\ge 30\%$):**
  - Test Precision: `10.33%` | Test Recall: `74.26%` | Test F1: `0.1814`

---

## 2. Benchmark Comparison Against Existing Capstone Models

Comparison on the exact same unseen Test Set (Steps 42–49, `8,841` transactions):

| Model | Paradigm | $\tau^*$ | Illicit Prec | Illicit Rec | Illicit F1 | PR-AUC | ROC-AUC | TP / 408 | FP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | Reference Baseline | 0.79 | 88.53% | 47.30% | 0.6166 | 0.5646 | 0.8848 | 193 | 25 |
| **Improved_Residual_GCN** | Reference Baseline | 0.59 | 27.66% | 38.24% | 0.3210 | 0.2761 | 0.7546 | 156 | 408 |
| **Temporal_GCN_LSTM** | Reference Baseline | 0.90 | 9.10% | 72.55% | 0.1617 | 0.2003 | 0.7805 | 296 | 2,956 |
| **Historical_GCN_LSTM** | Reference Baseline | 0.90 | 9.06% | 72.79% | 0.1611 | 0.2003 | 0.7805 | 297 | 2,982 |
| **`EvolveGCN_H_Directed_Res` (Stage 4C)** | **EvolveGCN-H** | 0.90 | 10.63% | 68.63% | **0.1841** | **0.3107** | 0.7963 | 280 | 2,353 |

### Objective Test-Set Delta Summary
- **vs. `HistGradientBoosting`:** $\Delta$F1 = `-0.4325`, $\Delta$PR-AUC = `-0.2539`, $\Delta$Precision = `-77.90 pp`, $\Delta$Recall = `+21.33 pp`.
- **vs. `Improved_Residual_GCN`:** $\Delta$F1 = `-0.1369`, $\Delta$PR-AUC = `+0.0346`, $\Delta$Precision = `-17.03 pp`, $\Delta$Recall = `+30.39 pp`.
- **vs. `Temporal_GCN_LSTM`:** $\Delta$F1 = `+0.0224`, $\Delta$PR-AUC = `+0.1104`, $\Delta$Precision = `+1.53 pp`, $\Delta$Recall = `-3.92 pp`.
- **vs. `Historical_GCN_LSTM`:** $\Delta$F1 = `+0.0230`, $\Delta$PR-AUC = `+0.1104`, $\Delta$Precision = `+1.57 pp`, $\Delta$Recall = `-4.16 pp`.

---

## 3. Per-Snapshot Test Trajectory & Calibration Check (Steps 42–49)

| Test Step | Labeled Nodes | Illicit | Mean Prob | Median Prob | Precision | Recall | Illicit F1 | PR-AUC | TP | FP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 2,154 | 239 | 0.4363 | 0.1514 | 31.51% | 89.12% | 0.4656 | 0.7882 | 213 | 463 |
| 43 | 1,370 | 24 | 0.5092 | 0.4702 | 3.77% | 75.00% | 0.0717 | 0.0892 | 18 | 460 |
| 44 | 1,591 | 24 | 0.6302 | 0.9361 | 1.51% | 54.17% | 0.0293 | 0.0143 | 13 | 850 |
| 45 | 1,221 | 5 | 0.3248 | 0.0701 | 0.85% | 40.00% | 0.0167 | 0.0122 | 2 | 233 |
| 46 | 712 | 2 | 0.2273 | 0.0594 | 0.00% | 0.00% | 0.0000 | 0.0129 | 0 | 89 |
| 47 | 846 | 22 | 0.2701 | 0.0570 | 5.93% | 36.36% | 0.1019 | 0.0658 | 8 | 127 |
| 48 | 471 | 36 | 0.2440 | 0.0501 | 27.69% | 50.00% | 0.3564 | 0.2403 | 18 | 47 |
| 49 | 476 | 56 | 0.2921 | 0.0609 | 8.70% | 14.29% | 0.1081 | 0.1318 | 8 | 84 |

---

## 4. Architectural Choices & Leakage Controls
- **Weight Evolution Without Additive Bias:** `EvolveGCNHLayer` evolves $W_t^{(l)}$ directly via `GRUCell` conditioned on `TopKSummarizer(H_t)`, removing the global broadcast vector $s_t$ used in Stage 4B.
- **LayerNorm Stabilization:** Per-node `LayerNorm` replaces `BatchNorm1d`, avoiding cross-node batch-statistics coupling across varying snapshot sizes (`1,089–7,880` nodes).
- **Zero-Leakage Protocol:** Imputer/scaler fitted solely on Steps `1–34`; `100%` intra-snapshot edges; `pos_weight` computed strictly on Steps `1–34`; unlabeled nodes (`y = -1`) excluded from loss/metrics; architecture selection, early stopping, and thresholds chosen strictly on Validation Steps `35–41`.
- **Training Stability:** Gradient norm clipped at `max_norm = 2.0` across 34-step BPTT.
