# Elliptic++ Dataset Inspection & Integrity Validation Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Report Generated:** 2026-10-09 16:20:56  
**Dataset Analyzed:** Elliptic++ Bitcoin Transaction Graph  

---

## 1. Executive Summary & Key Metrics

| Metric | Measured Value | Notes / Implications |
| :--- | :--- | :--- |
| **Total Transactions (Nodes)** | **203,769** | Exact 1-to-1 match across feature & class files |
| **Total Transaction Flows (Edges)** | **234,355** | Directed transaction input $\rightarrow$ output graph |
| **Total Features per Node** | **182** | 93 Local + 72 Aggregate + 17 Elliptic++ Domain Features |
| **Discrete Time Steps** | **49** | Time steps 1 to 49 (each step $\approx$ 2 weeks interval) |
| **Labeled Transactions** | **46,564 (22.85%)** | Illicit + Licit ground truth labels |
| **Unlabeled Transactions (Class 3)** | **157,205 (77.15%)** | Unlabeled nodes in graph structure |
| **Illicit Transactions (Fraud / Class 1)** | **4,545 (9.76% of labeled)** | Minority fraud class (Extreme class imbalance) |
| **Licit Transactions (Legitimate / Class 2)** | **42,019 (90.24% of labeled)** | Majority legitimate class |
| **Missing Values in 165 Core Features** | **0** | Local (93) and Aggregate (72) features are 100% complete |
| **Missing Values in 17 Domain Features** | **16,405 (965 rows, 0.474%)** | Affects 965 licit/unlabeled nodes; easily imputed |
| **Node ID Cross-File Alignment** | **100% Perfect Match** | Row-by-row matching between features & classes |
| **Dangling Edges** | **0** | All 234,355 edges connect existing nodes |
| **Cross-Time-Step Edges** | **0 (100% within time step)** | Graph partitions cleanly into 49 temporal snapshots |

---

## 2. Dataset Files & Schema Breakdown

### 2.1. Node Classes (`txs_classes.txt`)
- **Rows:** 203,769
- **Columns:** ['txId', 'class']
- **Class Encoding:**
  - `1` = **Illicit / Fraudulent** (4,545 transactions)
  - `2` = **Licit / Legitimate** (42,019 transactions)
  - `3` = **Unknown / Unlabeled** (157,205 transactions)
- **Data Quality:** 0 missing values, 0 duplicate transaction IDs.

### 2.2. Graph Topology / Edges (`txs_edgelist.txt`)
- **Rows (Directed Edges):** 234,355
- **Columns:** ['txId1', 'txId2'] (`txId1` = source transaction, `txId2` = destination transaction)
- **Topology Analysis:**
  - Source nodes (`txId1`): 166,345 unique
  - Destination nodes (`txId2`): 148,447 unique
  - Total unique nodes participating in edges: 203,769 (100.00% of all nodes)
  - Isolated nodes (0 in-degree and 0 out-degree): 0 (0.0%)
  - Self-loops: 0
  - Duplicate edges: 0

### 2.3. Transaction Features (`txs_features.txt`)
- **Rows:** 203,769
- **Columns:** 184 total
- **Feature Structure:**
  1. `txId`: Unique transaction identifier
  2. `Time step`: Integer step from 1 to 49
  3. `Local_feature_1` to `Local_feature_93` (93 local transaction attributes: standardized fees, transaction size, script properties)
  4. `Aggregate_feature_1` to `Aggregate_feature_72` (72 aggregated 1-hop neighborhood statistics)
  5. **17 Elliptic++ Domain/Graph Features (Interpretable domain metrics):**
     - Degree metrics: `in_txs_degree`, `out_txs_degree`
     - Volume & Fee: `total_BTC`, `fees`, `size`
     - Address metrics: `num_input_addresses`, `num_output_addresses`
     - Input Bitcoin flow stats: `in_BTC_min`, `in_BTC_max`, `in_BTC_mean`, `in_BTC_median`, `in_BTC_total`
     - Output Bitcoin flow stats: `out_BTC_min`, `out_BTC_max`, `out_BTC_mean`, `out_BTC_median`, `out_BTC_total`

---

## 3. Critical Research Observations & Data Quality Findings

### 3.1. Temporal Snapshot Architecture
Every single edge in `txs_edgelist.txt` connects nodes belonging to the **exact same time step** (234,355 edges, 100%).  
This means the dataset is naturally structured as **49 distinct static subgraphs (temporal graph snapshots)**. For our capstone:
- We can train temporal graph models using rolling-window or sequential snapshot mechanisms (e.g., EvolveGCN, GCN-LSTM, or TGN across time steps 1..34 for training, 35..49 for testing as standard in benchmark literature).

### 3.2. Extreme Class Imbalance & Evaluation Strategy
Among the labeled transactions, only **9.76%** (4,545 / 46,564) are illicit (fraud).  
Standard accuracy is completely misleading (a dummy model predicting all licit achieves >90% accuracy).  
**Required Evaluation Metrics:**
- **Minority Class Precision, Recall, and F1-Score (Illicit class)**
- **Precision-Recall AUC (PR-AUC)**
- **ROC-AUC**

### 3.3. Semi-Supervised / Large Unlabeled Proportion
157,205 transactions (77.15%) are unlabeled (Class 3).  
Graph Neural Networks are uniquely suited here because message passing propagates information through unlabeled bridge nodes to connect fraud patterns.

### 3.4. Missing Values in Domain Features
Exactly 965 transactions (~0.47%) have NaN in the 17 added domain features. 0 illicit nodes are affected (519 licit, 446 unknown). In Stage 2 preprocessing, these will be handled cleanly via median/zero imputation before feeding into baseline or GNN models.

### 3.5. Distinction Between Crypto Graphs vs. Bank/UPI
Elliptic++ contains Bitcoin blockchain transaction-to-transaction flow graphs with UTXO (Unspent Transaction Output) structure. In the project report and MTE presentation, this distinction must be clearly documented:
- Nodes represent transactions (not individual bank user accounts).
- Edges represent Bitcoin flows between transactions.
- Domain features include Bitcoin values, miner fees, and address counts.

---

## 4. Next Step
Proceed to **Stage 2: Preprocessing, Imputation, and PyTorch Geometric / Graph Construction** (creating PyG `Data` / `TemporalData` objects with clean temporal splits).
