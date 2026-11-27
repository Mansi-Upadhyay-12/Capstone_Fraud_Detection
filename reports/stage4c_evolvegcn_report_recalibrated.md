# Stage 4C Threshold Recalibration Report (Validation-Only Grid `0.01–0.99`)

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** 2026-11-26 10:56:41  
**Frozen Checkpoint Evaluated:** [models/evolvegcn_h.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/evolvegcn_h.pt) (`EvolveGCN_H_Directed_Res`, `best_epoch=37`)  
**Protocol Distinction:** This report evaluates the **exact same frozen Stage 4C model** using an expanded **validation-only** threshold grid (`0.01–0.99`, step `0.01`), separated from the original Stage 4C evaluation (`0.10–0.90` grid aligned with Stages 3–4B). Zero retraining and zero test-label tuning were performed.

---

## 1. Validation-Only Threshold Selection Summary (Steps `35–41`, `7,829` Nodes)

| Evaluation Protocol | Search Grid | $\tau^*$ (F1-Opt) | Val Prec | Val Rec | Val F1 | $\tau_{\text{rec30}}$ (Prec $\ge 30\%$) | Val Prec ($\tau_{\text{rec30}}$) | Val Rec ($\tau_{\text{rec30}}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage 4C** | `0.10–0.90` | `0.90` | 35.21% | 75.85% | 0.4810 | `0.83` | 30.12% | 80.15% |
| **Recalibrated Stage 4C** | `0.01–0.99` | **`0.97`** | **60.62%** | **58.37%** | **0.5947** | **`0.83`** | **30.12%** | **80.15%** |

- **Leakage Verification:** `rollout_val_split` was executed strictly on Steps `1–34` (warm-up) and Steps `35–41` (validation); Test Steps `42–49` were evaluated only after both thresholds (`0.97` and `0.83`) were finalized.

---

## 2. Side-by-Side Test Results: Original vs. Recalibrated (Steps `42–49`, `8,841` Nodes)

### A. F1-Optimal Operating Point ($\tau^*$)

| Protocol | $\tau^*$ | Precision | Recall | Illicit F1 | PR-AUC | ROC-AUC | Accuracy | Macro F1 | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage 4C** (`0.10–0.90`) | `0.90` | 10.63% | 68.63% | 0.1841 | 0.3107 | 0.7963 | 0.7194 | 0.5073 | 280 | 2,353 | 6,080 | 128 |
| **Recalibrated Stage 4C** (`0.01–0.99`) | **`0.97`** | **14.11%** | **55.64%** | **0.2251** | 0.3107 | 0.7963 | 0.8232 | 0.5627 | 227 | 1,382 | 7,051 | 181 |
| **Delta (Recalibrated - Original)** | `+0.07` | `+3.48 pp` | `-12.99 pp` | **`+0.0410`** | `0.0000` | `0.0000` | `+0.1038` | `+0.0554` | `-53` | **`-971`** | `+971` | `+53` |

### B. Recall-Constrained Operating Point ($\tau_{\text{rec30}}$, Val Prec $\ge 30\%$)

| Protocol | $\tau_{\text{rec30}}$ | Precision | Recall | Illicit F1 | PR-AUC | ROC-AUC | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Stage 4C** (`0.10–0.90`) | `0.83` | 10.33% | 74.26% | 0.1814 | 0.3107 | 0.7963 | 303 | 2,629 | 5,804 | 105 |
| **Recalibrated Stage 4C** (`0.01–0.99`) | **`0.83`** | 10.33% | 74.26% | 0.1814 | 0.3107 | 0.7963 | 303 | 2,629 | 5,804 | 105 |

*(Note: PR-AUC `0.3107` and ROC-AUC `0.7963` are threshold-independent ranking metrics and remain identical across threshold choices.)*

---

## 3. Per-Snapshot Test Comparison (`Steps 42–49`, F1-Optimal Threshold)

| Step | Illicit | Orig FP (`0.90`) | Recal FP (`0.97`) | Orig TP (`0.90`) | Recal TP (`0.97`) | Orig F1 | Recal F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 42 | 239 | 463 | **214** | 213 | 194 | 0.4656 | **0.5997** |
| 43 | 24 | 460 | **256** | 18 | 11 | 0.0717 | **0.0756** |
| 44 | 24 | 850 | **640** | 13 | 8 | 0.0293 | **0.0238** |
| 45 | 5 | 233 | **68** | 2 | 1 | 0.0167 | **0.0270** |
| 46 | 2 | 89 | **46** | 0 | 0 | 0.0000 | **0.0000** |
| 47 | 22 | 127 | **69** | 8 | 5 | 0.1019 | **0.1042** |
| 48 | 36 | 47 | **25** | 18 | 5 | 0.3564 | **0.1515** |
| 49 | 56 | 84 | **64** | 8 | 3 | 0.1081 | **0.0488** |

---

## 4. Benchmarking Fairness Note
- The 10 existing baseline models in [results/graphs/model_comparison.csv](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/graphs/model_comparison.csv) (`Logistic Regression`, `Temporal GCN-LSTM`, `Historical GCN-LSTM`, `Improved Residual GAT`, etc.) were tuned on the `0.10–0.90` grid.
- Therefore, **Original Stage 4C** ($\tau^* = 0.90$) should be used for apples-to-apples comparisons against the `0.10–0.90` grid baselines, while **Recalibrated Stage 4C** ($\tau^* = 0.97$) demonstrates the true calibrated operating capacity of the frozen `EvolveGCN_H_Directed_Res` weights without the `0.90` ceiling.
