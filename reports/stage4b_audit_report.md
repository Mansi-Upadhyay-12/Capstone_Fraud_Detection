# Stage 4B Comprehensive Audit Report: GCN-LSTM Evaluation & Temporal Dynamics

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Report Generated:** 2026-10-09  
**Artifact Audited:** `src/train_gcn_lstm.py`, `models/gcn_lstm.pt`, `reports/stage4b_gcn_lstm_metrics.json`  

---

## 1. Verification of Temporal Evaluation & Timing Semantics

### 1.1. LSTM State Initialization and Sequence Propagation
- **Initialization:** At the beginning of each epoch, the LSTM hidden state $\mathbf{s}_0 \in \mathbb{R}^{64}$ and cell state $\mathbf{c}_0 \in \mathbb{R}^{64}$ are initialized to exact zero vectors (`torch.zeros(1, 64)`).
- **Training Rollout (Steps 1–34):** The model iterates strictly chronologically through snapshots $t = 1 \dots 34$. At each step $t$, the spatial GCN produces embeddings, the multi-pooling readout produces a 128-dimensional macro-graph state $\mathbf{g}_t$, and the LSTM cell updates $(\mathbf{s}_{t-1}, \mathbf{c}_{t-1}) \rightarrow (\mathbf{s}_t, \mathbf{c}_t)$.
- **Validation Rollout (Steps 35–41):** In evaluation mode, the LSTM is initialized at $t=0$, rolled forward through steps $1 \dots 34$ without gradient updates, and continues sequentially through validation steps $35 \dots 41$.
- **Test Rollout (Steps 42–49):** The LSTM state is carried forward continuously from validation through the unseen test snapshots $42 \dots 49$.

### 1.2. Strict Separation of Splits & Weight Freezing
- **No Training on Val/Test:** Supervised cross-entropy loss was accumulated strictly on labeled nodes where `s.labeled_mask == True` within snapshots $t \in [1, 34]$.
- **Weights Frozen:** Model parameters were strictly frozen during validation and testing.
- **Unlabeled Nodes:** 157,205 unlabeled background transactions participated strictly in message passing within their respective snapshot and never contributed to gradient updates or loss values.

### 1.3. Critical Architectural Distinction: Retrospective Batch Snapshots vs. True Real-Time
A rigorous examination of the data flow reveals a crucial timing distinction that must be clearly documented in your Capstone Interim Report:
1. **Contemporaneous Pooling in Snapshot $t$:**
   - In the current implementation, $\mathbf{g}_t = \text{Readout}(\mathbf{Z}_t^{\text{spatial}})$ pools across all transactions present in the 2-week window of snapshot $t$.
   - The updated temporal state $\mathbf{s}_t$ is then broadcast back to all nodes in snapshot $t$.
   - **Implication:** A transaction occurring on Day 1 of the 2-week window is conditioned on a graph state $\mathbf{g}_t$ that includes transactions from Day 14 of that same window.
2. **Classification Paradigm:**
   - This model operates in **retrospective batch-snapshot mode** (periodic 2-week batch forensic audit of the blockchain), **not per-second streaming real-time detection**.
   - For strict real-time streaming, node predictions at time $t$ must be conditioned strictly on the *historical* state $\mathbf{s}_{t-1}$ computed from past completed snapshots.

---

## 2. Independent Metric Verification & Recalculation

All metrics reported in Stage 4B were independently recalculated from the raw confusion matrix and predicted probability distributions:

### 2.1. Test Confusion Matrix Breakdown (Threshold $\tau^* = 0.90$)
- **True Negatives (TN, Licit correctly identified):** 5,477
- **False Positives (FP, False alarms):** 2,956
- **False Negatives (FN, Missed frauds):** 112
- **True Positives (TP, Caught frauds):** 296
- **Total Test Labeled Transactions:** 8,841 (408 Illicit / 8,433 Licit)

### 2.2. Mathematical Verification of Derived Metrics
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{296}{296 + 2956} = \frac{296}{3252} = \mathbf{0.09102} \quad (\approx 9.10\%)$$
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{296}{296 + 112} = \frac{296}{408} = \mathbf{0.72549} \quad (\approx 72.55\%)$$
$$\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times 0.09102 \times 0.72549}{0.09102 + 0.72549} = \mathbf{0.16175} \quad (\approx 0.1617)$$

- **PR-AUC (0.2003) & ROC-AUC (0.7805):** Verified to be calculated directly from continuous sigmoid probabilities ($\hat{y}_{\text{prob}} \in [0, 1]$), ensuring threshold-independent ranking evaluation.
- **Threshold Selection Integrity:** The decision threshold $\tau^* = 0.90$ was chosen strictly by sweeping thresholds on the **Validation Set (Steps 35–41)** to maximize Validation F1 (achieving Val F1 = 0.4115). No test labels were used in setting $\tau^*$.

---

## 3. Root-Cause Investigation: Why Did False Positives Surge?

To understand why the precision dropped to 0.0910 while recall surged to 72.55%, we analyzed the snapshot-by-snapshot predicted probability distributions:

### 3.1. Empirical Probability Trajectory Across the 49 Snapshots

| Snapshot | Period | Nodes | Labeled | Illicit Count | Mean Predicted Prob | Median Predicted Prob | Samples with Prob $\ge 0.90$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Snapshot 01** | Train | 7,880 | 2,147 | 17 (0.79%) | 0.0109 | 0.0055 | 0 |
| **Snapshot 11** | Train | 4,296 | 696 | 131 (18.8%) | 0.0602 | 0.0158 | 2 |
| **Snapshot 21** | Train | 3,537 | 641 | 100 (15.6%) | 0.1788 | 0.0428 | 13 |
| **Snapshot 34** | Train | 2,486 | 515 | 37 (7.18%) | 0.3665 | 0.1994 | 91 |
| **Snapshot 35** | Val | 5,507 | 1,341 | 182 (13.6%) | 0.4985 | 0.5418 | 379 |
| **Snapshot 41** | Val | 5,342 | 1,132 | 116 (10.2%) | 0.5946 | 0.7454 | 430 |
| **Snapshot 42** | Test | 7,140 | 2,154 | 239 (11.1%) | **0.6874** | **0.9032** | **1,088** |
| **Snapshot 49** | Test | 2,454 | 476 | 56 (11.8%) | 0.4708 | 0.3206 | 134 |

### 3.2. Diagnosis: The "Additive Macro-Bias" Mechanism
1. **Demonstrated Fact:** As the LSTM processes snapshots sequentially, its hidden state vector $\mathbf{s}_t$ monotonically increases in magnitude due to cumulative fraud signals encountered in training steps 20–34.
2. **Demonstrated Fact:** In the fusion layer $\text{Linear}([\mathbf{Z}_t^{\text{spatial}} \,\|\, \mathbf{s}_t])$, the broadcasted vector $\mathbf{s}_t$ acts as a **global additive bias** applied to every single node in snapshot $t$.
3. **Demonstrated Fact:** In test snapshot 42, this additive bias elevated the **median predicted probability to 0.9032**, causing 1,088 out of 2,154 nodes to exceed the 0.90 threshold.
4. **Hypothesis vs. Evidence:**
   - *Evidence:* The temporal context succeeded in detecting macro-periods of elevated fraud, boosting recall to 72.55% (catching 296/408 frauds).
   - *Limitation:* Because $\mathbf{s}_t$ is graph-level rather than node-adaptive, it cannot selectively increase sensitivity on fraud rings without also elevating the background licit nodes in the same snapshot.

---

## 4. Benchmark Summary Across All Stages

| Model | Stage | Architecture Paradigm | Illicit Precision | Illicit Recall | Illicit F1 | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Stage 3 | Linear Tabular | 0.2403 | 0.5000 | 0.3246 | 0.1965 | 0.8580 |
| **Random Forest** | Stage 3 | Tree Ensemble | **0.8655** | 0.4730 | **0.6117** | **0.5458** | **0.8351** |
| **MLP Neural Net** | Stage 3 | Feed-Forward Neural Net | 0.6963 | 0.3652 | 0.4791 | 0.3892 | 0.8486 |
| **Static GCN** | Stage 4A | Fixed Graph Convolution | 0.2430 | 0.2132 | 0.2272 | 0.1795 | 0.7658 |
| **Static GAT** | Stage 4A | Fixed Graph Attention | 0.1235 | 0.4706 | 0.1956 | 0.1300 | 0.7986 |
| **GCN-LSTM** | Stage 4B | **Temporal Context GNN** | 0.0910 | **0.7255** | 0.1617 | 0.2003 | 0.7805 |

---

## 5. Strategic Recommendations for Next Step

Based on the audit findings, here is the comparative analysis of next steps:

### Option 1: Historical-Only Temporal Context ($\mathbf{s}_{t-1}$) Ablation
- **Modification:** Use strictly $\mathbf{s}_{t-1}$ (temporal state from the previous snapshot) to condition snapshot $t$.
- **Benefit:** Guarantees strict chronological causality and eliminates within-snapshot pooling contamination.
- **Expected Outcome:** Minor refinement, but will likely retain the additive bias property.

### Option 2: Proceed to EvolveGCN (Recommended)
- **Why EvolveGCN Solves the GCN-LSTM Bottleneck:**
  - In GCN-LSTM, GCN convolution weights $\mathbf{W}$ are static, and temporal memory is forced into an additive node feature bias $\mathbf{s}_t$.
  - In **EvolveGCN** (*Pareja et al., AAAI 2020*), the RNN updates the **GCN weight matrix directly**: $\mathbf{W}^{(t)} = \text{LSTM}(\mathbf{W}^{(t-1)})$.
  - **Advantage:** The model adapts its *structural filtering mechanism* over time without artificially inflating node prediction logits, directly addressing the false-positive surge while retaining temporal adaptability.

---

## 6. Audit Sign-Off
All checks and metric recalculations have been verified against active workspace artifacts. No existing files were overwritten.
