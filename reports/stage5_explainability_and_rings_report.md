# Stage 5: Explainable AI (XAI) & Fraud Ring Community Analysis Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** 2026-10-10 23:41:12  
**Evaluation Set:** Test Snapshots 42–49 (8,841 labeled transactions: 408 Illicit / 8,433 Licit)  

---

## 1. Executive Summary

This stage addresses the critical regulatory, forensic, and topological requirements of the capstone project:
1. **Explainable AI (XAI):** Uncovering *why* specific Bitcoin transactions are classified as fraudulent, analyzing feature importances across 182 dimensions and performing subgraph attribution on Graph Neural Networks.
2. **Fraud Ring Community Detection:** Treating the detection of coordinated multi-node fraud syndicates as an explicit network-level task, distinct from single-transaction node classification.

---

## 2. Global Feature Importance (Tabular & Ensemble XAI)

Analysis of tree ensemble models reveals the relative importance of feature families in identifying illicit money transfers:

| Feature Family | Total Dimensions | Importance Share | Key Intuition |
| :--- | :--- | :--- | :--- |
| **Local Features** | 93 | **70.7%** | Direct transaction metrics (transaction fees, input/output counts, BTC amounts). |
| **Aggregate Features** | 72 | **17.4%** | 1-hop neighborhood statistics (neighbor fees, neighbor transaction counts). |
| **Domain Features** | 17 | **11.9%** | 2-hop aggregations and wallet-level patterns. |

### Top Predictive Features for Illicit Detection:
- **Rank 1**: `Local_feature_53` (Importance: 0.065192)
- **Rank 2**: `Local_feature_55` (Importance: 0.055026)
- **Rank 3**: `Local_feature_41` (Importance: 0.042674)
- **Rank 4**: `size` (Importance: 0.040300)
- **Rank 5**: `Local_feature_49` (Importance: 0.037281)

*(Plot saved to: `results/feature_importance.png`)*

---

## 3. GNN Local Subgraph Explanation & Attribution

For an identified illicit transaction evaluated by the Graph Neural Network:
- **Target Node Index:** `167212`
- **Predicted Fraud Probability:** `57.84%`
- **Extracted Computation Subgraph:** `14` nodes, `13` transfer edges.

### Subgraph Attribution Insights:
- The GNN leverages local topological connectivity to corroborate illicit status. Even when individual transaction amounts appear benign, the structural flow connecting the transaction to suspicious neighbor clusters strongly amplifies the model's fraud logit.
- Gradient saliency reveals that neighbor transaction frequency and fee variance drive the greatest shift in node representations.

*(Visualization saved to: `results/gnn_subgraph_explanation.png`)*

---

## 4. Fraud-Ring Detection vs. Single-Node Classification

### 4.1. Formal Distinction
- **Single-Transaction Classification:** Computes $P(y_i = 1) \ge \tau^*$ for isolated nodes. Evaluates precision, recall, and F1 at the individual transaction level.
- **Fraud-Ring Detection:** Detects subgraphs $C \subseteq V$ where $|C| \ge 2$, with internal edge transfers linking coordinated illicit transactions and intermediate money mules.

### 4.2. Quantitative Ring Detection Results (Test Period 42–49)
- **Total Ground-Truth Illicit Transactions in Test:** `408`
- **Isolated Singleton Frauds:** `298`
- **Detected Coordinated Fraud Rings (Size $\ge 2$):** **`21` rings**

### 4.3. Laundering Topology Breakdown:
- **Peeling Chains (Sequential Hopping):** `1` rings
- **Fan-Out Dispersion (Layering):** `2` rings
- **Fan-In Aggregation (Consolidation):** `3` rings
- **Complex Mesh / Cyclic Syndicates:** `15` rings

*(Topology distribution chart saved to: `results/ring_topology_distribution.png`)*

### 4.4. Paradigm Comparison for Ring Detection:
1. **Tabular Models (Random Forest):** Treats transactions independently. Misses 53.8% of multi-node fraud rings because it completely fractures connected laundering chains and cannot leverage unlabeled bridging nodes.
2. **Static GNN (GCN/GAT):** Incorporates local edge topology but suffers from severe concept drift over time (illicit recall = 21.3%).
3. **Temporal GNN (GCN-LSTM):** Recovers **73.1% of all multi-node laundering rings**, maintaining topological continuity and capturing evolving temporal ring structures.

---

## 5. Artifacts & Outputs
- **JSON Metrics:** [reports/stage5_explainability_and_rings_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage5_explainability_and_rings_metrics.json)
- **Feature Importance Plot:** [results/feature_importance.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/feature_importance.png)
- **GNN Subgraph Explanation:** [results/gnn_subgraph_explanation.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/gnn_subgraph_explanation.png)
- **Ring Topology Distribution:** [results/ring_topology_distribution.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/ring_topology_distribution.png)
