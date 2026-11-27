# Stage 4B.1 Focused Verification Audit Report: Original GCN-LSTM vs. Historical-Only GCN-LSTM

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Audit Executed:** 2026-10-09  
**Artifacts Audited:**
- `src/train_gcn_lstm.py` & `models/gcn_lstm.pt` (Stage 4B)
- `src/train_gcn_lstm_historical.py` & `models/gcn_lstm_historical.pt` (Stage 4B.1)
- `reports/stage4b_gcn_lstm_metrics.json` & `reports/stage4b_gcn_lstm_report.md`
- `reports/stage4b1_gcn_lstm_historical_metrics.json` & `reports/stage4b1_gcn_lstm_historical_report.md`
- `processed/temporal_snapshots.pt` (49 snapshots, 203,769 total nodes, 8,841 test labeled nodes)

---

## 1. Executive Summary & Verification Conclusion

An exhaustive, deterministic audit was performed by loading the saved model checkpoints, executing independent forward passes across all 49 snapshots, and recalculating continuous test prediction probabilities and all derived metrics from raw arrays.

### Key Audit Findings:
1. **Distinct Trained Models:** The checkpoints contain distinct weights across all layers ($\max |\mathbf{W}_{\text{orig}} - \mathbf{W}_{\text{hist}}| = 0.458588$).
2. **Distinct Continuous Predictions:** Over **99.41% of test samples** (8,789 / 8,841) have different predicted continuous probabilities between the two models ($\max |\Delta p| = 0.072774$).
3. **Explanation of Identical 4-Decimal PR-AUC and ROC-AUC:**
   - Original PR-AUC: `0.2002827657` $\to$ Rounds to `0.2003`
   - Historical PR-AUC: `0.2002571145` $\to$ Rounds to `0.2003` (Raw Difference: `0.0000256512`)
   - Original ROC-AUC: `0.7805406166` $\to$ Rounds to `0.7805`
   - Historical ROC-AUC: `0.7804683921` $\to$ Rounds to `0.7805` (Raw Difference: `0.0000722244`)
   - **Conclusion:** The values are **not duplicated or copied**. They are distinct float64 numbers that round to the exact same 4-decimal values.
4. **Threshold Selection Purity:** The decision threshold $\tau^* = 0.90$ was determined strictly on the 7,829 validation nodes (snapshots 35–41) without any exposure to test labels.
5. **Causal Temporal Flow Verified:** The historical model strictly uses $\mathbf{s}_{t-1}$ to compute node logits for snapshot $t$, updating the LSTM memory $\mathbf{s}_t$ only after snapshot $t$ predictions are finalized.
6. **Snapshot 42 Node Population Clarified:** The audit clearly resolves the population basis: total nodes in snapshot 42 = 7,140, labeled nodes = 2,154. The probability statistics were calculated over the labeled population.

---

## 2. Direct Recalculation & Side-by-Side Comparison (Test Set: Steps 42–49)

*Population: 8,841 labeled nodes (408 Illicit / 8,433 Licit)*

| Metric / Attribute | Original GCN-LSTM (Stage 4B) | Historical-Only GCN-LSTM (Stage 4B.1) | Mathematical Difference ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Temporal Conditioning State** | $\mathbf{s}_t$ (Contemporaneous) | $\mathbf{s}_{t-1}$ (Strict Historical) | Causal Separation |
| **Optimal Validation Threshold ($\tau^*$)** | `0.90` (Val F1 = 0.4115) | `0.90` (Val F1 = 0.4082) | Exact Match |
| **Illicit True Positives (TP)** | **296** / 408 | **297** / 408 | +1 Caught Fraud |
| **Illicit False Negatives (FN)** | **112** / 408 | **111** / 408 | -1 Missed Fraud |
| **Licit True Negatives (TN)** | **5,477** / 8,433 | **5,451** / 8,433 | -26 Licit Correct |
| **Licit False Positives (FP)** | **2,956** | **2,982** | +26 False Alarms |
| **Illicit Recall (Sensitivity)** | **0.725490** (72.55%) | **0.727941** (72.79%) | +0.24% |
| **Illicit Precision** | **0.091021** (0.0910) | **0.090576** (0.0906) | -0.04% |
| **Illicit F1-Score** | **0.161749** (0.1617) | **0.161107** (0.1611) | -0.06% |
| **PR-AUC (Unrounded float64)** | **0.2002827657** | **0.2002571145** | -0.0000256512 |
| **PR-AUC (Reported 4 dec)** | **0.2003** | **0.2003** | Identical at 4 decimals |
| **ROC-AUC (Unrounded float64)** | **0.7805406166** | **0.7804683921** | -0.0000722244 |
| **ROC-AUC (Reported 4 dec)** | **0.7805** | **0.7805** | Identical at 4 decimals |
| **Macro F1-Score** | **0.4715** | **0.4700** | -0.15% |

---

## 3. Investigation of Identical PR-AUC & ROC-AUC (Root Cause)

### 3.1. Verification that Arrays and Metrics are Independent
- **Check 1: Checkpoint Weights:** Verified that $\mathbf{W}_{\text{orig}} \neq \mathbf{W}_{\text{hist}}$ across all layers.
- **Check 2: Prediction Differences:** Calculated $\Delta \mathbf{p} = \hat{\mathbf{y}}_{\text{orig}} - \hat{\mathbf{y}}_{\text{hist}}$ on the 8,841 test nodes:
  - Samples with $\Delta p \neq 0$: **8,789 / 8,841 (99.41%)**
  - $\max |\Delta p| = 0.072774$
  - $\text{mean} |\Delta p| = 0.005325$
  - Mean probability: Original = `0.566841`, Historical = `0.568995`
- **Check 3: Unrounded Area Calculations:**
  $$\text{PR-AUC}_{\text{orig}} = 0.2002827657 \quad \text{vs.} \quad \text{PR-AUC}_{\text{hist}} = 0.2002571145 \quad (\Delta = 2.56 \times 10^{-5})$$
  $$\text{ROC-AUC}_{\text{orig}} = 0.7805406166 \quad \text{vs.} \quad \text{ROC-AUC}_{\text{hist}} = 0.7804683921 \quad (\Delta = 7.22 \times 10^{-5})$$

### 3.2. Why are the Curves so Close?
Because the temporal state evolution $\mathbf{s}_{t-1} \to \mathbf{s}_t$ in the LSTM is driven by the global macro state of the network, which evolves smoothly across 2-week intervals. Shifting from $\mathbf{s}_t$ to $\mathbf{s}_{t-1}$ creates a 1-step lag in the macro context vector, but the underlying spatial GCN embeddings $\mathbf{Z}_t^{\text{spatial}}$ (which supply the transaction-specific discriminating signal) remain identical in architecture. Consequently, the global ranking of probabilities is preserved with only microscopic perturbations, causing the 4-decimal rounded PR-AUC and ROC-AUC to match.

---

## 4. Verification of Temporal Causality & Implementation Purity

### 4.1. Code Verification in `src/train_gcn_lstm_historical.py`
In [FraudGCNLSTMHistorical.forward_step_historical](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_gcn_lstm_historical.py#L67-L97):
```python
# Step 1: Spatial Message Passing using only current snapshot graph (X_t, E_t)
z_spatial = self.conv2(F.relu(self.bn1(self.conv1(x, edge_index))), edge_index)

# Step 2: Prediction conditioned STRICTLY on prior historical state h_prev (s_{t-1})
num_nodes = z_spatial.shape[0]
h_prev_broadcast = h_prev.expand(num_nodes, -1)
z_combined = torch.cat([z_spatial, h_prev_broadcast], dim=-1)
logits_t = self.fusion(z_combined).squeeze(-1)

# Step 3: Graph Readout of snapshot t (unsupervised mean + max pooling)
g_t = torch.cat([z_spatial.mean(dim=0, keepdim=True), z_spatial.max(dim=0, keepdim=True)[0]], dim=-1)

# Step 4: Update LSTM memory for FUTURE snapshots (s_{t-1} -> s_t)
h_next, c_next = self.lstm_cell(g_t, (h_prev, c_prev))
return logits_t, h_next, c_next
```
- Node predictions $\hat{\mathbf{y}}_t$ have **zero dependence on $\mathbf{g}_t$ or $\mathbf{s}_t$**.
- Temporal memory update occurs **strictly after** logits are computed.
- No label information ($y$) is ever fed into the readout $\mathbf{g}_t$ or LSTM state transitions.

---

## 5. Snapshot 42 Node Population & Probability Discrepancy Resolution

To ensure complete clarity across all documentation, the population breakdown and probability distributions for Snapshot 42 were audited across both full-graph and labeled-subset scopes:

### 5.1. Exact Node Counts in Snapshot 42 (Index 41, First Test Snapshot)
- **Total Graph Nodes:** **7,140**
- **Unlabeled Background Nodes ($y = -1$, Class 3):** **4,986** (69.83% of snapshot)
- **Supervised Labeled Nodes ($y \in \{0, 1\}$):** **2,154** (30.17% of snapshot)
  - Illicit / Fraudulent ($y = 1$, Class 1): **239** (11.10% of labeled)
  - Licit / Legitimate ($y = 0$, Class 2): **1,915** (88.90% of labeled)

### 5.2. Distribution Breakdown Across Populations

| Population Evaluated | Node Count | Model | Mean Predicted Prob | Median Predicted Prob | Count with $\hat{p} \ge 0.90$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Snapshot 42 Nodes** | **7,140** | Original GCN-LSTM | 0.7795 | 0.9476 | 4,391 (61.50%) |
| **All Snapshot 42 Nodes** | **7,140** | Historical GCN-LSTM | 0.7798 | 0.9487 | 4,399 (61.61%) |
| **Labeled Snapshot 42 Nodes** | **2,154** | Original GCN-LSTM | 0.6874 | 0.9032 | 1,088 (50.51%) |
| **Labeled Snapshot 42 Nodes** | **2,154** | Historical GCN-LSTM | 0.6885 | 0.9048 | 1,097 (50.93%) |

**Resolution:** The earlier Stage 4B audit report listed `Nodes: 7,140` and `Labeled: 2,154` side-by-side, but the probability columns (Mean `0.6874`, Median `0.9032`, Count $\ge 0.90$: `1,088`) were computed strictly over the **2,154 labeled nodes**. This is now explicitly clarified in the documentation.

---

## 6. Verification of Chronological Splits & Academic Boundary

1. **Chronological Splits:**
   - Training: Steps 1–34 (34 snapshots, 29,894 labeled nodes: 3,462 Illicit / 26,432 Licit).
   - Validation: Steps 35–41 (7 snapshots, 7,829 labeled nodes: 675 Illicit / 7,154 Licit).
   - Test: Steps 42–49 (8 snapshots, 8,841 labeled nodes: 408 Illicit / 8,433 Licit).
2. **Preprocessing Purity:**
   - Missing value imputation (`SimpleImputer(strategy='median')`) and feature standardization (`StandardScaler`) were fitted **exclusively on Training steps 1–34**.
3. **Operational Mode Clarification:**
   - The spatial features in Elliptic++ (specifically the 72 1-hop aggregate statistics) summarize 2-week intervals.
   - Therefore, while the historical temporal recurrence $\mathbf{s}_{t-1}$ enforces strict snapshot-to-snapshot causality, the model operates in **retrospective batch-snapshot forensic audit mode** rather than real-time per-second stream scoring.

---

## 7. Audit Sign-Off & Recommendation

- **Audit Status:** **PASSED WITH COMPLETE REPRODUCIBILITY**.
- All metrics reported in Stage 4B and Stage 4B.1 have been independently validated against raw continuous prediction arrays.
- **Architectural Conclusion:** Both GCN-LSTM variants demonstrate that an additive macro-temporal context vector $\mathbf{s}$ boosts fraud recall to $\approx 72.7\%$ but suffers from high false positives because the GCN spatial weights remain static across time.
- **Next Logical Step:** Proceed to **Stage 4C: EvolveGCN (Evolving Graph Convolutional Networks)**, where an RNN directly evolves the graph convolution weight matrix $\mathbf{W}^{(t)} = \text{RNN}(\mathbf{W}^{(t-1)})$, enabling dynamic structural adaptability without additive node bias.
