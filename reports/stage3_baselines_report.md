# Stage 3: Tabular Baseline Models Performance Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** 2026-10-13 11:52:08  
**Evaluation Set:** Test Snapshots 42–49 (8,841 labeled transactions: 408 Illicit / 8,433 Licit)  

---

## 1. Executive Summary & Benchmark Comparison

This stage benchmarks four classical non-graph machine learning models on the preprocessed 182-dimensional Elliptic++ feature set:
1. **Logistic Regression (Linear)**
2. **Random Forest (Tree Ensemble)**
3. **Gradient Boosting (HistGradientBoostingClassifier)**
4. **Multi-Layer Perceptron (Deep Feed-Forward Network)**

All models were trained strictly on **Timesteps 1–34** (29,894 labeled nodes), validated on **Timesteps 35–41** (7,829 labeled nodes) for hyperparameter/threshold selection, and evaluated on unseen future **Timesteps 42–49** (8,841 labeled nodes).

### Final Test Performance (Timesteps 42–49):

| Classifier Model | Illicit Precision | Illicit Recall | Illicit F1-Score | PR-AUC (Avg Prec) | ROC-AUC | Macro F1 | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression (Linear)** | 0.2383 | 0.5000 | **0.3228** | 0.1940 | 0.8574 | 0.6353 | 2.252s |
| **Random Forest (Ensemble)** | **0.8655** | 0.4730 | **0.6117** | **0.5458** | 0.8351 | **0.7987** | 2.826s |
| **Gradient Boosting (HistGB)** | 0.8853 | 0.4730 | **0.6166** | 0.5646 | **0.8848** | 0.8013 | 31.222s |
| **MLP Neural Network (Deep)** | 0.7143 | 0.4289 | **0.5360** | 0.4742 | 0.8360 | 0.7591 | 79.096s |

---

## 2. Detailed Performance Breakdown by Model

### 2.1. Logistic Regression (Baseline Linear Classifier)
- **Architecture & Setup:** L2-regularized logistic regression with `class_weight='balanced'`.
- **Validation Tuning:** Optimal decision threshold selected on validation set = `0.90` (Validation Illicit F1: `0.4962`).
- **Confusion Matrix on Test Set:**
  - True Negatives (Licit identified correctly): **7,781** / 8,433
  - False Positives (False alarms): **652**
  - False Negatives (Missed frauds): **204** / 408
  - True Positives (Caught frauds): **204** / 408

### 2.2. Random Forest (Non-Linear Tree Ensemble)
- **Architecture & Setup:** 100 decision trees, max depth 16, balanced sample weighting.
- **Validation Tuning:** Optimal decision threshold selected on validation set = `0.59` (Validation Illicit F1: `0.9116`).
- **Confusion Matrix on Test Set:**
  - True Negatives (Licit identified correctly): **8,403** / 8,433
  - False Positives (False alarms): **30**
  - False Negatives (Missed frauds): **215** / 408
  - True Positives (Caught frauds): **193** / 408

### 2.3. Gradient Boosting (Histogram-Based Tree Boosting)
- **Architecture & Setup:** HistGradientBoostingClassifier with `class_weight='balanced'`, 100 max iterations, max depth 8, learning rate 0.1.
- **Validation Tuning:** Optimal decision threshold selected on validation set = `0.79` (Validation Illicit F1: `0.9113`).
- **Confusion Matrix on Test Set:**
  - True Negatives (Licit identified correctly): **8,408** / 8,433
  - False Positives (False alarms): **25**
  - False Negatives (Missed frauds): **215** / 408
  - True Positives (Caught frauds): **193** / 408

### 2.4. Multi-Layer Perceptron (Deep Neural Network)
- **Architecture & Setup:** 3-layer MLP (`182 -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3) -> Linear(64) -> BatchNorm -> ReLU -> Linear(1)`) with `BCEWithLogitsLoss(pos_weight=7.6349)`.
- **Validation Tuning:** Best model checkpoint saved at Epoch 4 with optimal threshold = `0.89` (Validation Illicit F1: `0.7405`).
- **Confusion Matrix on Test Set:**
  - True Negatives (Licit identified correctly): **8,363** / 8,433
  - False Positives (False alarms): **70**
  - False Negatives (Missed frauds): **233** / 408
  - True Positives (Caught frauds): **175** / 408

---

## 3. Key Observations & Limitations of Tabular Baselines

1. **The Core Limitation of Non-Graph Models:**
   - Tabular classifiers treat every transaction as an **independent, isolated data point (IID assumption)**.
   - They fail to capture multi-hop structural money-laundering patterns (e.g., peeling chains, fan-out/fan-in layering, and ring topologies) where illicit funds are deliberately fragmented across intermediate accounts.
2. **Impact of 77.15% Unlabeled Data:**
   - Classical tabular models must completely discard the 157,205 unlabeled background transactions during training.
   - In contrast, Graph Neural Networks (Stage 4) can propagate structural embeddings through unlabeled intermediary nodes to uncover hidden fraud rings.
3. **Temporal Concept Drift:**
   - The illicit ratio drops from 11.58% in training to 4.61% in test timesteps. Tree ensembles like Random Forest and Gradient Boosting generalize better to shifting distributions than unregularized networks, establishing a strong tabular benchmark for Stage 4 GNN comparison.

---

## 4. Next Step
Proceed to **Stage 4: Graph Neural Network & Temporal GNN Implementation** (evaluating GCN, GAT, and Temporal GCN-LSTM on the 49 temporal snapshots to capture structural topological patterns).
