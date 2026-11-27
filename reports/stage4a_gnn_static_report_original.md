# Stage 4A: Static Graph Neural Networks (GCN & GAT) Evaluation Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** 2026-10-10 22:32:11  
**Evaluation Set:** Test Snapshots 42–49 (8,841 labeled transactions: 408 Illicit / 8,433 Licit)  

---

## 1. Executive Summary & Benchmark Comparison

In Stage 4A, we implemented and evaluated two fundamental static Graph Neural Network (GNN) architectures:
1. **Graph Convolutional Network (GCN):** Aggregates local neighborhood features using standardized symmetric graph convolution.
2. **Graph Attention Network (GAT):** Uses multi-head self-attention mechanisms to dynamically learn which neighboring transactions are most indicative of money laundering.

Both models were trained on **Timesteps 1–34** (29,894 labeled nodes + 106,371 unlabeled graph bridge nodes), validated on **Timesteps 35–41** (7,829 labeled nodes), and evaluated on unseen future **Timesteps 42–49** (8,841 labeled nodes).

### Final Benchmark Table on Test Snapshots (Steps 42–49):

| Model Architecture | Model Category | Illicit Precision | Illicit Recall | Illicit F1-Score | PR-AUC (Avg Prec) | ROC-AUC | Macro F1 | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Stage 3 Baseline)** | Tabular Ensemble | **0.8655** | 0.4730 | **0.6117** | **0.5458** | **0.8351** | **0.7987** | 2.22s |
| **Graph Convolutional Network (GCN)** | Static Graph GNN | 0.2430 | 0.2132 | 0.2272 | 0.1795 | 0.7658 | 0.5961 | 164.436s |
| **Graph Attention Network (GAT)** | Attentive Graph GNN | 0.1235 | **0.4706** | 0.1956 | 0.1300 | 0.7986 | 0.5476 | 117.969s |

---

## 2. Detailed Performance & Confusion Matrices on Test Set

### 2.1. Graph Convolutional Network (GCN)
- **Architecture:** `182 -> GCNConv(128) -> BatchNorm -> ReLU -> Dropout(0.3) -> GCNConv(64) -> BatchNorm -> ReLU -> Linear(1)`
- **Validation-Selected Optimal Threshold:** `0.80` (Validation Illicit F1: `0.4003`)
- **Test Confusion Matrix (out of 408 Frauds & 8,433 Legitimate):**
  - True Negatives (Licit correctly identified): **8,162** / 8,433
  - False Positives (False alarms): **271**
  - False Negatives (Missed fraud): **321** / 408
  - True Positives (Caught fraud): **87** / 408 (**Recall: 21.32%**)

### 2.2. Graph Attention Network (GAT)
- **Architecture:** `182 -> GATConv(32x4 heads = 128) -> BatchNorm -> ELU -> Dropout(0.3) -> GATConv(64) -> BatchNorm -> ELU -> Linear(1)`
- **Validation-Selected Optimal Threshold:** `0.77` (Validation Illicit F1: `0.3648`)
- **Test Confusion Matrix (out of 408 Frauds & 8,433 Legitimate):**
  - True Negatives (Licit correctly identified): **7,070** / 8,433
  - False Positives (False alarms): **1,363**
  - False Negatives (Missed fraud): **216** / 408
  - True Positives (Caught fraud): **192** / 408 (**Recall: 47.06%**)

---

## 3. Deep Research Analysis: Why Static GNNs Suffer Under Temporal Drift

1. **The Static GNN Assumption Failure:**
   - Static GCN and GAT assume that graph convolution weights remain stationary over time.
   - However, in Bitcoin transaction graphs, illicit entities constantly adapt their behavior to avoid detection (e.g. changing transaction sizes, splitting funds through different intermediary graph structures).
   - Consequently, fixed-weight static GNNs trained on time steps 1–34 overfit to historical graph topology and experience performance drop when tested on steps 42–49.
2. **Why Random Forest Held Up Better on Static Features:**
   - Random Forest splits on isolated tabular features (BTC amounts, fees) which retain predictive value regardless of network topology changes.
   - Static GNNs intertwine node features directly with local neighborhood connectivity. When the graph connectivity changes, static GNN representations degrade.
3. **The Core Motivation for Stage 4B (Temporal GNNs):**
   - This exact empirical finding mirrors the foundational findings of **Weber et al. (2019)** and **Pareja et al. (EvolveGCN, 2020)**.
   - To overcome temporal concept drift, we must move from **Static GNNs** to **Temporal Graph Neural Networks (EvolveGCN / GCN-LSTM)**, where recurrent units dynamically evolve the GCN weight matrices from snapshot to snapshot.

---

## 4. Academic Integrity & Temporal Leakage Documentation

- **Graph Topology Isolation:** 100% of the 234,355 edges in the dataset connect transactions within the same 2-week time step. There are **zero cross-timestep edges**, strictly preventing future graph connections from leaking into earlier snapshots.
- **Supervision Purity:** The 157,205 unlabeled transactions (`y = -1`) participated strictly in message passing and were entirely excluded from loss and metric calculations.
- **Dataset Property Note for MTE Report:** The 72 aggregate features in Elliptic++ summarize 1-hop statistics within each 2-week snapshot window. Within any given 2-week block, transactions occurring later in that block could contribute to aggregate features of earlier transactions in the same block.

---

## 5. Next Step
Proceed to **Stage 4B: Temporal Graph Neural Network Implementation (EvolveGCN / GCN-LSTM)** to dynamically track evolving fraud rings across temporal snapshots and achieve state-of-the-art fraud detection performance.
