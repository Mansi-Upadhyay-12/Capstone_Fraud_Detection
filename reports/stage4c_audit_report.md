# Stage 4C Comprehensive Audit Report: EvolveGCN-H Verification & Forensic Diagnostics

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Audit Date:** 2026-11-26  
**Artifacts Audited:**
- [src/train_evolvegcn.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_evolvegcn.py)
- [train_evolvegcn.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/train_evolvegcn.py)
- [models/evolvegcn_h.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/evolvegcn_h.pt)
- [reports/stage4c_evolvegcn_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_metrics.json)
- [reports/stage4c_evolvegcn_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_report.md)
- [processed/temporal_snapshots.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/processed/temporal_snapshots.pt)

---

## 1. Executive Summary

An independent, deterministic, read-only audit of Stage 4C (`EvolveGCN-H`) was executed by inspecting [src/train_evolvegcn.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_evolvegcn.py), loading the saved checkpoint [models/evolvegcn_h.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/evolvegcn_h.pt), re-running deterministic chronological rollouts across all 49 snapshots, and verifying every Validation, Test, confusion-matrix, and per-snapshot metric against [reports/stage4c_evolvegcn_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_metrics.json) and [reports/stage4c_evolvegcn_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_report.md).

**Overall Audit Verdict:** **PASS** (with 4 documented operational/calibration caveats for future refinement; zero data leakage and 100% metric reproducibility confirmed).

---

## 2. Detailed Verification Findings

### Check 1: Validation-Only Architecture Selection & Post-Selection Test Evaluation — **PASS**
- **Code Path Verified ([src/train_evolvegcn.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_evolvegcn.py)):**
  - `train_and_validate_config` (lines 320–428) trains strictly on snapshot indices `0..33` (`Steps 1–34`) and calls `rollout_val_split` (lines 268–285), which iterates only over `0..33` (warm-up) and `34..40` (`Steps 35–41`, Validation). Test snapshots `41..48` (`Steps 42–49`) are never accessed.
  - `run_stage4c` selects `best_key` strictly via `validation_metrics["pr_auc"]` (lines 868–875), picking `EvolveGCN_H_Directed_Res` (`Val PR-AUC = 0.6387` vs `0.3140`, `0.1518`, and `0.1313`).
  - `evaluate_config_on_test` (lines 431–482) is invoked only **after** `best_key` is selected (`lines 884–887`), evaluating `EvolveGCN_H_Directed_Res` first, followed by the remaining 3 ablation variants for post-selection comparison.

### Check 2: Validation-Only Operating Thresholds ($\tau^*$ and $\tau_{\text{rec30}}$) — **PASS (with Grid-Ceiling Caveat)**
- Both thresholds are computed inside `train_and_validate_config` strictly from `y_val_all` and `y_val_prob_best` (`Steps 35–41`, `7,829` labeled nodes):
  - **Optimal F1 Threshold ($\tau^*$):** Recomputed as **`0.90`**, yielding Validation Illicit F1 = **`0.4810`** (`Prec = 0.3521`, `Rec = 0.7585`).
  - **Recall-Constrained Threshold ($\tau_{\text{rec30}}$, $\text{Val Prec} \ge 0.30$):** Recomputed as **`0.83`**, yielding Validation Precision = **`0.3012`**, Recall = **`0.8015`**, F1 = **`0.4379`**.
- **Important Grid-Ceiling Finding:** Because `find_best_threshold` sweeps `np.linspace(0.10, 0.90, 81)` (matching Stages 3, 4A, 4B, and 4B.1), $\tau^*$ is capped at `0.90`. Evaluating thresholds above `0.90` on Validation reveals that Validation F1 peaks at **`0.97`** (`Val F1 = 0.5947`, `Val Prec = 0.6062`, `Val Rec = 0.5837`), where Test F1 would be `0.2251` (`Prec = 0.1411`, `Rec = 0.5564`, `FP = 1,382`).

### Check 3 & Check 5: Checkpoint Reproduction & Metric/Report Consistency — **PASS**
- Loading [models/evolvegcn_h.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/evolvegcn_h.pt) (`config_name = EvolveGCN_H_Directed_Res`, `use_undirected = False`, `use_residual = True`, `best_epoch = 37`) and running deterministic `rollout_val_split` and `rollout_test_split` reproduced every metric in [reports/stage4c_evolvegcn_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_metrics.json) and [reports/stage4c_evolvegcn_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_report.md) with **zero differences (`diffs = {}`)**:
  - **Validation (`Steps 35–41`, $\tau^*=0.90$):** `PR-AUC = 0.6387` (`0.638722`), `ROC-AUC = 0.8849` (`0.884865`), `F1 = 0.4810`, `TP = 512`, `FP = 942`, `FN = 163`, `TN = 6,212` — **Exact Match**.
  - **Test Optimal ($\tau^*=0.90$):** `PR-AUC = 0.3107`, `ROC-AUC = 0.7963`, `Prec = 0.1063`, `Rec = 0.6863`, `F1 = 0.1841`, `TP = 280`, `FP = 2,353`, `FN = 128`, `TN = 6,080` — **Exact Match**.
  - **Test Recall-Constrained ($\tau_{\text{rec30}}=0.83$):** `Prec = 0.1033`, `Rec = 0.7426`, `F1 = 0.1814`, `TP = 303`, `FP = 2,629`, `FN = 105`, `TN = 5,804` — **Exact Match**.
  - **Test Default ($\tau=0.50$):** `Prec = 0.0927`, `Rec = 0.7941`, `F1 = 0.1660`, `TP = 324`, `FP = 3,172`, `FN = 84`, `TN = 5,261` — **Exact Match**.
  - **All 8 Per-Snapshot Test Entries (`42–49`):** `labeled_mean_prob`, `labeled_median_prob`, `illicit_precision`, `illicit_recall`, `illicit_f1`, `pr_auc`, `roc_auc`, `tp`, `fp`, `fn` — **8/8 Exact Match**.

### Check 4: Investigation of Step 44 High False Positives & Step 46 Missed Frauds — **PASS (Root Causes Identified)**

#### A. Why Did Step 44 Produce `850` False Positives (`54.24%` of Licit Nodes)?
Step 44 contains `4,975` total nodes (`1,591` labeled: `24` illicit, `1,567` licit). Of the `1,567` labeled licit nodes, `850` scored $p \ge 0.90$, raising labeled median probability to `0.9361` (and all-node median probability to `0.9668`):
1. **Convolutional Weight Dominance Over Residual Skip:** Across Steps `41–44`, Layer 1 evolved weight norm `||W1||` peaked at `93.90` (Step 43) and `75.89` (Step 44), while `||W2||` grew monotonically (`8.35 -> 13.69 -> 16.55 -> 17.80`). As a result, Layer 1 convolution output norm (`h1_conv_norm = 98.63`) was **~9.5× larger** than the residual skip highway (`h1_skip_norm = 10.41`), reducing the stabilizing effect of the residual branch.
2. **Feature-Norm Regime Shift on Step 44 Labeled Licit Nodes:** In Step 44, labeled licit transactions experienced a sharp drop in average feature norm (`|x|_mean = 14.66`, vs `19.33` in Step 42, `22.42` in Step 43, `23.60` in Step 45, and `23.14` in Step 46) and lower mean in-degree (`1.64` vs `2.31–2.55` in Steps 45–46), shifting them closer to the unlabeled background cluster (`|x|_mean = 11.88`, where `70.2%` of unlabeled nodes scored $\ge 0.90$).
3. **Top-$k$ Summarizer Composition in Step 44:** Layer 1's `TopK-128` selection for `W_44` contained `0` illicit nodes (`70` licit, `58` unlabeled), shifting the evolved filter toward background high-score activation (`AllMeanP = 0.7324`, `AllMedP = 0.9668`). In Step 45, when labeled licit `|x|_mean` rebounded to `23.60`, labeled median probability dropped back to `0.0701`.

#### B. Why Were `0 / 2` Frauds Caught in Step 46 (`TP = 0`, `FN = 2`)?
Step 46 (`3,519` total nodes, `712` labeled) contains **only 2 ground-truth illicit transactions**, and both received elevated probabilities below the strict $\tau^* = 0.90$ cutoff:
1. **`txId = 75078587` (`local_idx = 1279`):** Predicted **`prob = 0.6835`** (`logit = +0.7700`). It is a high-in-degree sink node (`in_deg = 45, out_deg = 0`) whose 45 incoming neighbors consist of **`8` known licit nodes (`y=0`) and `37` unlabeled nodes (`y=-1`)** with zero illicit predecessors. Directed `GCN` aggregation diluted its fraud signal with 45 non-illicit incoming nodes.
2. **`txId = 196122645` (`local_idx = 3392`):** Predicted **`prob = 0.8379`** (`logit = +1.6430`), `in_deg = 0, out_deg = 1` (outgoing to an unlabeled node). It was missed solely because $\tau^* = 0.90$ exceeded `0.8379`—it **is caught** under the validation recall-oriented threshold $\tau_{\text{rec30}} = 0.83$, and **both** Step 46 frauds (`2/2`, `100%` recall) are caught at the default threshold $\tau = 0.50$.

### Check 6: Ablation Conclusions Verification — **PASS**
All ablation conclusions in [reports/stage4c_evolvegcn_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_report.md) are directly supported by the logged metrics in [reports/stage4c_evolvegcn_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_evolvegcn_metrics.json):
- **Residual Skip Connections:** Adding residual skips boosted Validation PR-AUC from `0.1313` $\to$ `0.6387` (directed) and `0.1518` $\to$ `0.3140` (undirected), and Test PR-AUC from `0.0606` $\to$ `0.3107` (directed) and `0.0560` $\to$ `0.0893` (undirected).
- **Directed vs. Undirected Topology:** Directed edges (`EvolveGCN_H_Directed_Res`) substantially outperformed undirected edges (`EvolveGCN_H_Undirected_Res`) on both Validation PR-AUC (`0.6387` vs `0.3140`) and Test PR-AUC (`0.3107` vs `0.0893`). Furthermore, `EvolveGCN_H_Undirected_Res` suffered severe threshold miscalibration between Validation (`tau*=0.79`, Val Recall `37.19%`) and Test (Test Recall `0.49%`, `TP = 2/408`).

### Check 7: Data Leakage, Temporal Leakage & Evaluation Inconsistencies — **PASS (with 4 Documented Caveats)**
- **Verified Zero Leakage:**
  - Features imputed & scaled strictly on Steps `1–34`.
  - `100%` intra-snapshot edges (`0` cross-timestep edges).
  - Class-imbalance `pos_weight = 7.6349` computed strictly on Steps `1–34`.
  - All `157,205` unlabeled nodes (`y = -1`) excluded from loss and evaluation metrics.
  - Architecture selection, early stopping, and thresholds computed strictly on Validation Steps `35–41`.
- **Documented Caveats / Evaluation Inconsistencies:**
  1. **Retrospective Batch-Snapshot Mode:** Within each 2-week snapshot `t`, `TopKSummarizer(H_t)` and Elliptic++'s 72 precomputed 1-hop aggregate features incorporate all nodes in that 2-week window (including unlabeled nodes, though zero labels `y` are used). Like Stages 4A/4B/4B.1, this is retrospective 2-week batch auditing rather than per-second streaming.
  2. **Threshold Sweep Upper Bound (`0.90`):** `find_best_threshold` stops at `0.90` (matching Stages 3–4B), leaving higher-precision validation thresholds (`0.91–0.97`) unexplored.
  3. **Unlabeled Background Drift & Weight Norm Growth:** Unbounded `||W1||` growth (`75.89–93.90`) combined with `pos_weight=7.6349` causes `40.9%–70.2%` of unlabeled nodes in Steps `42–46` to exceed `0.90`, making labeled licit nodes vulnerable when their feature norms dip (Step 44).
  4. **Post-Selection Ablation Test Runs & Unintegrated Scripts:** `run_stage4c` evaluates the selected model first on Test, then evaluates the 3 non-selected variants on Test for the ablation table (zero leakage since selection already finished, though 4 configs total touched Test post-selection). Separately, [visualize_results.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/visualize_results.py) and [smoke_test_all.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/smoke_test_all.py) have not yet been updated to include Stage 4C, and [reports/stage4b_gcn_lstm_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4b_gcn_lstm_metrics.json) / [reports/stage4b1_gcn_lstm_historical_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4b1_gcn_lstm_historical_metrics.json) still contain 2-epoch smoke-test values (Stage 4C correctly avoided them by reading [results/graphs/model_comparison.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/graphs/model_comparison.json)).

---

## 3. PASS / FAIL Audit Checklist

| # | Audit Criterion | Status | Notes |
| :---: | :--- | :---: | :--- |
| 1 | Architecture selected using Validation PR-AUC only; Test `42–49` evaluated post-selection | **PASS** | `EvolveGCN_H_Directed_Res` selected on `Val PR-AUC = 0.6387`; `evaluate_config_on_test` runs strictly afterward. |
| 2 | Both thresholds ($\tau^* = 0.90$, $\tau_{\text{rec30}} = 0.83$) selected on Validation only | **PASS** | Computed strictly from `y_val_all` (`Steps 35–41`); note `[0.10, 0.90]` grid ceiling. |
| 3 | Reported test metrics, confusion counts, and per-snapshot results match saved JSON/MD | **PASS** | `0` discrepancies across all overall, threshold-variant, and 8 per-snapshot metrics. |
| 4 | Root causes identified for Step 44 (`850` FP) and Step 46 (`0/2` caught) | **PASS** | Step 44 feature-norm shift + `h1_conv` dominance; Step 46 `2` frauds (`0.6835`, `0.8379`) below `0.90` cutoff. |
| 5 | Saved checkpoint [models/evolvegcn_h.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/evolvegcn_h.pt) reproduces reported metrics | **PASS** | Exact bit-level metric match from deterministic rollout. |
| 6 | Ablation conclusions accurately supported by results | **PASS** | Directed + Residual wins on Val PR-AUC (`0.6387`) and Test PR-AUC (`0.3107`). |
| 7 | Data leakage, temporal leakage, and evaluation inconsistencies audited | **PASS (Caveats Noted)** | Zero split/label/feature leakage; 4 calibration/operational caveats documented above. |

---

## 4. Recommended Future Corrections (For Next Iteration, Not Applied Here)
1. **Expand Threshold Search / Calibrate Probabilities:** Allow `find_best_threshold` to sweep `[0.10, 0.99]` (or calibrate logits via temperature scaling on Validation) so $\tau^*$ is not artificially clipped at `0.90` under `pos_weight=7.6349`.
2. **Stabilize Weight Norms Across Snapshots:** Evaluate weight-norm regularization / `LayerNorm` on $W_t$ or residual scaling in `FraudEvolveGCNH` to prevent `||W1||` drift (`75–94`) from overwhelming the skip connection during regime shifts (Step 44).
3. **Sync Benchmark Pipeline:** Once Stage 4C refinements (if any) are finalized, add `EvolveGCN_H_Directed_Res` to [visualize_results.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/visualize_results.py) and [smoke_test_all.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/smoke_test_all.py), and regenerate the overwritten Stage 4B/4B.1 JSON reports.
