# 🛡️ Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic and industrial-grade research framework evaluating **Tabular ML**, **Static Graph Neural Networks (GCN, GAT)**, and **Temporal Graph Neural Networks (GCN-LSTM, EvolveGCN)** for detecting money laundering, illicit Bitcoin transactions, and coordinated fraud rings over dynamic transaction graphs.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset: Elliptic++ Bitcoin Graph](#-dataset-elliptic-bitcoin-graph)
3. [Key Benchmark Results](#-key-benchmark-results)
4. [Project Architecture & Stages](#-project-architecture--stages)
5. [Repository Structure](#-repository-structure)
6. [Quick Start & Setup](#-quick-start--setup)
7. [Running Experiments](#-running-experiments)
8. [Next Steps & Roadmap](#-next-steps--roadmap)

---

## 🚀 Project Overview

Financial transaction networks present three critical challenges for traditional fraud detection systems:
1. **Dynamic Topology:** Fraud rings actively evolve their laundering patterns, creating subgraphs that change across time.
2. **Ephemeral Transactions:** Transactions exist only at the discrete time step when confirmed on-chain.
3. **Severe Class Imbalance & Concept Drift:** Frauds constitute less than 2% of network activity, and transaction distributions drift drastically across market cycles.

This capstone project rigorously builds, audits, and compares machine learning paradigms under a **strict zero-data-leakage chronological evaluation protocol**:
- **Tabular Machine Learning:** Logistic Regression, MLP Neural Network, Random Forest.
- **Static Graph Neural Networks:** 2-layer Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT).
- **Temporal Graph Neural Networks:** GCN-LSTM (Macro-temporal context) and Historical-Only GCN-LSTM Ablation.
- **Future Phases:** EvolveGCN (Dynamic Graph Convolutions), Subgraph Explainability (XAI with GNNExplainer), and an Interactive Web Dashboard.

---

## 📊 Dataset: Elliptic++ Bitcoin Graph

The project utilizes the **Elliptic++** benchmark, an enriched graph dataset of Bitcoin transactions:
- **Total Nodes (Transactions):** `203,769`
- **Total Directed Edges (Payment Flows):** `234,355` (100% intra-timestep, 0 cross-timestep edges)
- **Features per Node:** `182` features (93 local transaction features + 72 1-hop aggregate neighborhood features + 17 2-hop aggregate features)
- **Time Steps:** `49` discrete snapshots (approx. 2-week intervals from April 2017 to March 2019)
- **Class Labels:**
  - **Illicit (Fraud / Laundering):** `4,545` nodes (`Class 1`, $y = 1$)
  - **Licit (Legitimate):** `42,019` nodes (`Class 2`, $y = 0$)
  - **Unlabeled Background:** `157,205` nodes (`Class 3`, $y = -1$) — *used strictly for topological message passing, never in supervised loss or evaluation.*

### Chronological Data Split (Zero-Leakage)
- **Training Snapshots (Steps 1–34):** 29,894 labeled transactions (3,462 Illicit / 26,432 Licit)
- **Validation Snapshots (Steps 35–41):** 7,829 labeled transactions (675 Illicit / 7,154 Licit) — *Used exclusively for early stopping & threshold selection ($\tau^*$).*
- **Test Snapshots (Steps 42–49):** 8,841 labeled transactions (408 Illicit / 8,433 Licit) — *Evaluated strictly once per model.*

---

## 🏆 Key Benchmark Results

All models were evaluated on the **Unseen Test Set (Snapshots 42–49)** using decision thresholds ($\tau^*$) selected strictly on the Validation Set:

| Model Architecture | Paradigm | Illicit Precision | Illicit Recall | Illicit F1-Score | PR-AUC (Avg Prec) | ROC-AUC | Key Insight |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | Tabular Ensemble | **86.55%** | 47.30% | **0.6117** | **0.5458** | **0.8351** | Strongest precision; relies heavily on aggregated tabular features. |
| **MLP Neural Net** | Tabular Deep Learning | 69.63% | 36.52% | 0.4791 | 0.3892 | 0.8486 | Moderate performance without graph topology. |
| **Logistic Regression** | Linear Tabular | 24.03% | 50.00% | 0.3246 | 0.1965 | 0.8580 | Fast baseline; high false alarm rate. |
| **Static GCN** | Static Graph GNN | 24.30% | 21.32% | 0.2272 | 0.1795 | 0.7658 | Degrades under temporal concept drift on future snapshots. |
| **Static GAT** | Attentive Graph GNN | 12.35% | 47.06% | 0.1956 | 0.1300 | 0.7986 | Attention captures more fraud chains, but lowers precision. |
| **GCN-LSTM (Original)** | Contemporaneous $s_t$ | 9.10% | 72.55% | 0.1617 | 0.2003 | 0.7805 | Catches 296/408 frauds; high recall via macro-state context. |
| **GCN-LSTM (Historical)** | **Historical $s_{t-1}$** | 9.06% | **72.79%** | 0.1611 | 0.2003 | 0.7805 | **Highest recall (297/408 frauds caught)**; enforces strict causal temporal separation. |

### Summary of Experimental Insights:
1. **The Recall Champion:** The **Temporal GCN-LSTM** family caught **~73% of all illicit transactions**, far surpassing static GNNs (21%) and Random Forest (47%).
2. **The Precision Bottleneck:** GCN-LSTM's false-positive rate is caused by keeping GCN weights static while using an additive macro-temporal vector. This empirically proves the need for **EvolveGCN** (where the graph convolution weights themselves evolve over time).

---

## 🔬 Project Architecture & Stages

```
┌────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Dataset Inspection & Structural Validation                    │
│ - Verified 203k nodes, 234k intra-snapshot edges, 49 discrete steps    │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼─────────────────────────────────────┐
│ Stage 2: Preprocessing & Temporal Graph Construction                   │
│ - PyG Data Snapshot construction, Median Imputation & Standard Scaling │
│ - Strict Zero-Leakage: Scaler fitted exclusively on Steps 1-34         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Stage 3: Tabular │      │ Stage 4A: Static │      │ Stage 4B / 4B.1: │
│ Baselines        │      │ Graph GNNs       │      │ Temporal GNNs    │
│ - Logistic Reg.  │      │ - Static GCN     │      │ - GCN-LSTM       │
│ - Random Forest  │      │ - Static GAT     │      │ - Hist. Ablation │
│ - MLP Baseline   │      │                  │      │                  │
└──────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                             │
                                   ┌─────────────────────────┴─────────┐
                                   ▼                                   ▼
                        ┌──────────────────────┐            ┌──────────────────────┐
                        │ Stage 4C (Upcoming): │            │ Stage 5 & 6:         │
                        │ EvolveGCN            │            │ XAI & Dashboard Demo │
                        │ (Weight Evolution)   │            │ (GNNExplainer + UI)  │
                        └──────────────────────┘            └──────────────────────┘
```

---

## 📁 Repository Structure

```
Capstone_Fraud_Detection/
├── .gitignore                      # Excludes large binaries (>100MB) & cache
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Python environment dependencies
│
├── inspect_dataset.py              # Stage 1 Dataset validation script
├── preprocess_graph.py             # Stage 2 Graph preprocessing pipeline
│
├── train_baselines.py              # Root launcher for Tabular Baselines (Stage 3)
├── train_gnn_static.py             # Root launcher for Static GNNs (Stage 4A)
├── train_gcn_lstm.py               # Root launcher for GCN-LSTM (Stage 4B)
├── train_gcn_lstm_historical.py    # Root launcher for Historical GCN-LSTM (Stage 4B.1)
│
├── src/                            # Core implementation source code
│   ├── train_baselines.py          # Tabular models (LR, RF, MLP)
│   ├── train_gnn_static.py         # PyG Static GCN & GAT architectures
│   ├── train_gcn_lstm.py           # Temporal GCN-LSTM architecture
│   ├── train_gcn_lstm_historical.py# Causal Historical-Only GCN-LSTM
│   └── verify_stage4b1.py          # Independent verification audit script
│
├── models/                         # Saved trained model checkpoints
│   ├── random_forest.joblib        # Best Tabular Model (F1: 0.6117)
│   ├── mlp_baseline.pt             # PyTorch MLP Weights
│   ├── logistic_regression.joblib  # Scikit-learn Logistic Regression
│   ├── gcn_static.pt               # PyG Static GCN Weights
│   ├── gat_static.pt               # PyG Static GAT Weights
│   ├── gcn_lstm.pt                 # Temporal GCN-LSTM Weights
│   └── gcn_lstm_historical.pt      # Historical-Only GCN-LSTM Weights
│
└── reports/                        # Comprehensive Markdown & JSON audit reports
    ├── dataset_inspection_report.md
    ├── stage2_preprocessing_report.md
    ├── stage3_baselines_report.md
    ├── stage3_baselines_metrics.json
    ├── stage4a_gnn_static_report.md
    ├── stage4a_gnn_static_metrics.json
    ├── stage4b_gcn_lstm_report.md
    ├── stage4b_gcn_lstm_metrics.json
    ├── stage4b_audit_report.md
    ├── stage4b1_gcn_lstm_historical_report.md
    ├── stage4b1_gcn_lstm_historical_metrics.json
    └── stage4b1_verification_audit.md
```

---

## ⚡ Quick Start & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mansi-Upadhyay-12/Capstone_Fraud_Detection.git
cd Capstone_Fraud_Detection
```

### 2. Set Up Python Virtual Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 Running Experiments

All experiment pipelines are modular and reproducible with a single command:

### 1. Run Data Preprocessing
```bash
python preprocess_graph.py
```

### 2. Run Tabular Baselines (Random Forest, MLP, Logistic Regression)
```bash
python train_baselines.py
```

### 3. Run Static Graph Neural Networks (GCN & GAT)
```bash
python train_gnn_static.py
```

### 4. Run Temporal GCN-LSTM
```bash
python train_gcn_lstm.py
```

### 5. Run Historical-Only Temporal Context Ablation
```bash
python train_gcn_lstm_historical.py
```

### 6. Run Independent Verification Audit
```bash
python src/verify_stage4b1.py
```

---

## 🔮 Next Steps & Roadmap

- [x] **Stage 1:** Dataset structural verification & class distribution analysis.
- [x] **Stage 2:** Zero-leakage temporal graph construction (49 PyG snapshots).
- [x] **Stage 3:** Tabular ML baselines (Random Forest F1: 0.6117).
- [x] **Stage 4A:** Static GNNs (GCN & GAT) & temporal drift identification.
- [x] **Stage 4B & 4B.1:** Temporal GCN-LSTM & Historical Ablation (Fraud recall: 72.79%).
- [ ] **Stage 4C:** **EvolveGCN** (Evolving Graph Convolutional Neural Networks).
- [ ] **Stage 5:** **Explainable AI (XAI)** with GNNExplainer for fraud ring subgraphs.
- [ ] **Stage 6:** **Interactive Web Dashboard** for real-time visualization.

---

## 👤 Author & Acknowledgements

- **Author:** Mansi Upadhyay ([@Mansi-Upadhyay-12](https://github.com/Mansi-Upadhyay-12))
- **Project:** Final Year B.Tech Computer Science & Engineering Capstone Project
- **Dataset Reference:** *Elliptic++: An Enriched Bitcoin Transaction Dataset for Fraud Detection* (Youssef et al., 2023).
