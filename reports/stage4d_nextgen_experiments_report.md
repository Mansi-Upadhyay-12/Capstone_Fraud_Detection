# Stage 4D: Next-Generation Graph Fraud Detection Experimental Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** 2026-11-26 12:17:23  
**Selection Protocol:** Train Steps `1–34` $\to$ Validate/Select Steps `35–41` (PR-AUC) $\to$ Post-Selection Test Steps `42–49`  
**Overall Champion (Across Both Groups, by Multi-Seed Val PR-AUC):** `E6_Local93_GCNAvg` (`graph`, Mean Val PR-AUC = `0.9040`)  
**Best Graph-Based Champion:** `E6_Local93_GCNAvg` (Best Seed `42`)  
**Best No-Graph Control:** `E6_Local93_NoGraph` (Best Seed `456`)  

---

## 1. Screening Results Across Controlled Experiments (Validation Steps `35–41`)

| Exp ID | Type | Group | Best Ep | Val PR-AUC | Val F1 (`0.10–0.90`) | Val F1 (`0.01–0.99`) | Val Prec | Val Rec | Brier | ECE | Unl $\ge 0.90$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E6_Local93_GCNAvg` | Graph | `Exp6_Graph_Value_Test` | 40 | **0.9051** | 0.8569 | 0.8654 | 94.24% | 80.00% | 0.0657 | 0.1274 | 10.4% |
| `E6_Local93_NoGraph` | No-Graph | `Exp6_Graph_Value_Test` | 40 | **0.8927** | 0.8602 | 0.8602 | 89.31% | 82.96% | 0.0634 | 0.1263 | 8.6% |
| `E6_Local93_SAGEConcat` | Graph | `Exp6_Graph_Value_Test` | 25 | **0.8120** | 0.7247 | 0.7481 | 84.51% | 67.11% | 0.0723 | 0.1077 | 15.2% |
| `E5_Loss_ModeratedBCE_pw2.76` | Graph | `Exp5_Loss_Comparison` | 15 | **0.7412** | 0.6812 | 0.6812 | 71.69% | 64.89% | 0.0652 | 0.1296 | 0.0% |
| `E6_Full182_NoGraph` | No-Graph | `Exp6_Graph_Value_Test` | 18 | **0.7196** | 0.6484 | 0.6848 | 88.50% | 55.85% | 0.0900 | 0.1283 | 11.0% |
| `E5_Loss_Focal_a0.50_g2.0` | Graph | `Exp5_Loss_Comparison` | 25 | **0.7015** | 0.6616 | 0.6616 | 76.61% | 58.22% | 0.0922 | 0.2095 | 0.0% |
| `E6_Full182_GCNAvg` | Graph | `Exp6_Graph_Value_Test` | 4 | **0.6967** | 0.6593 | 0.6593 | 77.03% | 57.63% | 0.2115 | 0.3977 | 0.0% |
| `E4_CausalTemporalAttnGCN` | Graph | `Exp4_Temporal_Attention` | 4 | **0.6902** | 0.6320 | 0.6320 | 63.30% | 63.11% | 0.1521 | 0.2828 | 3.0% |
| `E1_Stab_EvolveGCNH_DropEdge` | Graph | `Exp1_Stabilize_EvolveGCNH` | 10 | **0.6853** | 0.6480 | 0.6480 | 76.05% | 56.44% | 0.1819 | 0.3483 | 0.0% |
| `E3_Stab_EvolveGCNO` | Graph | `Exp3_EvolveGCNO` | 33 | **0.6705** | 0.5800 | 0.5800 | 95.58% | 41.63% | 0.0957 | 0.1484 | 0.0% |
| `E1_Stab_EvolveGCNH_DualLN_Bound` | Graph | `Exp1_Stabilize_EvolveGCNH` | 27 | **0.6469** | 0.5352 | 0.5986 | 73.75% | 50.37% | 0.1160 | 0.1701 | 23.6% |
| `E5_Loss_Focal_a0.75_g2.0` | Graph | `Exp5_Loss_Comparison` | 4 | **0.6357** | 0.5912 | 0.5912 | 57.58% | 60.74% | 0.1725 | 0.3178 | 0.0% |
| `E4_CausalAttnEvolveGCN` | Graph | `Exp4_Temporal_Attention` | 8 | **0.5664** | 0.5185 | 0.5185 | 44.99% | 61.19% | 0.1741 | 0.3200 | 0.0% |
| `E6_Full182_SAGEConcat` | Graph | `Exp6_Graph_Value_Test` | 15 | **0.5440** | 0.5349 | 0.5349 | 53.06% | 53.93% | 0.1213 | 0.2224 | 0.0% |

---

## 2. Multi-Seed Shortlist Evaluation (Validation Steps `35–41`)

| Shortlisted Exp ID | Category | Seeds | Val PR-AUC (Mean $\pm$ Std) | Val F1 `0.01–0.99` (Mean $\pm$ Std) | Brier Mean | ECE Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `E6_Local93_GCNAvg` | Graph | `[42, 123, 456]` | **0.9040 $\pm$ 0.0008** | 0.8633 $\pm$ 0.0031 | 0.0673 | 0.1272 |
| `E6_Local93_NoGraph` | No-Graph Control | `[42, 123, 456]` | **0.8946 $\pm$ 0.0038** | 0.8633 $\pm$ 0.0040 | 0.0612 | 0.1223 |
| `E6_Local93_SAGEConcat` | Graph | `[42, 123, 456]` | **0.8399 $\pm$ 0.0219** | 0.7739 $\pm$ 0.0226 | 0.0686 | 0.1083 |
| `E6_Full182_NoGraph` | No-Graph Control | `[42, 123, 456]` | **0.7207 $\pm$ 0.0145** | 0.6723 $\pm$ 0.0127 | 0.1271 | 0.2037 |
| `E5_Loss_ModeratedBCE_pw2.76` | Graph | `[42, 123, 456]` | **0.7091 $\pm$ 0.0345** | 0.6514 $\pm$ 0.0284 | 0.0629 | 0.1001 |

### Per-Seed Validation Breakdown

| Exp ID | Seed | Best Ep | Val PR-AUC | Val F1 (`0.01–0.99`) | Val Prec | Val Rec | Brier | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E6_Local93_GCNAvg` | 42 | 40 | 0.9051 | 0.8654 | 94.24% | 80.00% | 0.0657 | 0.1274 |
| `E6_Local93_GCNAvg` | 123 | 50 | 0.9034 | 0.8590 | 94.17% | 78.96% | 0.0595 | 0.1100 |
| `E6_Local93_GCNAvg` | 456 | 40 | 0.9035 | 0.8656 | 94.72% | 79.70% | 0.0767 | 0.1442 |
| `E6_Local93_SAGEConcat` | 42 | 25 | 0.8120 | 0.7481 | 84.51% | 67.11% | 0.0723 | 0.1077 |
| `E6_Local93_SAGEConcat` | 123 | 28 | 0.8422 | 0.7705 | 79.94% | 74.37% | 0.0611 | 0.0984 |
| `E6_Local93_SAGEConcat` | 456 | 30 | 0.8654 | 0.8032 | 88.14% | 73.78% | 0.0723 | 0.1189 |
| `E5_Loss_ModeratedBCE_pw2.76` | 42 | 15 | 0.7412 | 0.6812 | 71.69% | 64.89% | 0.0652 | 0.1296 |
| `E5_Loss_ModeratedBCE_pw2.76` | 123 | 16 | 0.7250 | 0.6597 | 66.32% | 65.63% | 0.0644 | 0.0870 |
| `E5_Loss_ModeratedBCE_pw2.76` | 456 | 14 | 0.6612 | 0.6132 | 58.86% | 64.00% | 0.0590 | 0.0836 |
| `E6_Local93_NoGraph` | 42 | 40 | 0.8927 | 0.8602 | 89.31% | 82.96% | 0.0634 | 0.1263 |
| `E6_Local93_NoGraph` | 123 | 40 | 0.8911 | 0.8608 | 92.96% | 80.15% | 0.0633 | 0.1235 |
| `E6_Local93_NoGraph` | 456 | 46 | 0.8999 | 0.8690 | 94.28% | 80.59% | 0.0570 | 0.1170 |
| `E6_Full182_NoGraph` | 42 | 18 | 0.7196 | 0.6848 | 88.50% | 55.85% | 0.0900 | 0.1283 |
| `E6_Full182_NoGraph` | 123 | 18 | 0.7035 | 0.6773 | 83.99% | 56.74% | 0.1127 | 0.1430 |
| `E6_Full182_NoGraph` | 456 | 7 | 0.7390 | 0.6548 | 81.64% | 54.67% | 0.1787 | 0.3397 |

---

## 3. Post-Selection Test Evaluation: Graph Champion vs. No-Graph Control (Steps `42–49`, `8,841` Nodes)

| Champion Role | Exp ID | Grid | $\tau^*$ | Precision | Recall | Illicit F1 | PR-AUC | ROC-AUC | Brier | ECE | TP | FP | TN | FN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graph Champion** | `E6_Local93_GCNAvg` | `0.10–0.90` | `0.90` | 75.20% | 46.08% | **0.5714** | **0.5268** | 0.8188 | 0.0740 | 0.1145 | 188 | 62 | 8,371 | 220 |
| **Graph Champion** | `E6_Local93_GCNAvg` | `0.01–0.99` | `0.93` | 85.39% | 45.83% | **0.5965** | **0.5268** | 0.8188 | 0.0740 | 0.1145 | 187 | 32 | 8,401 | 221 |
| **No-Graph Control** | `E6_Local93_NoGraph` | `0.10–0.90` | `0.90` | 81.82% | 44.12% | **0.5732** | **0.4591** | 0.7980 | 0.0685 | 0.1124 | 180 | 40 | 8,393 | 228 |
| **No-Graph Control** | `E6_Local93_NoGraph` | `0.01–0.99` | `0.93` | 88.06% | 43.38% | **0.5813** | **0.4591** | 0.7980 | 0.0685 | 0.1124 | 177 | 24 | 8,409 | 231 |

### Quantified Graph Message Passing Impact (`Graph Champion` minus `No-Graph Control`, `0.01–0.99` Grid)
- **$\Delta$ Test PR-AUC:** `+0.0677` (`Graph higher`)
- **$\Delta$ Test ROC-AUC:** `+0.0208`
- **$\Delta$ Test Illicit F1:** `+0.0152` (`Graph higher`)
- **$\Delta$ Precision / Recall:** `-2.67 pp` / `+2.45 pp`
- **$\Delta$ Calibration (Brier / ECE):** `+0.0055` / `+0.0021`
- **$\Delta$ Confusion Counts (TP / FP / TN / FN):** `+10` / `+8` / `-8` / `-10`
