# Stage 5D: Publication-Quality Research Visualization Report

**Generated On:** 2026-11-26 21:43:24  
**Format & Resolution:** `300 DPI` PNG and Vector PDF (`figures/stage5/`)  
**Preservation Verification:** Frozen Stage 4H checkpoint (`ExpA2_RankGauss182_BiDirGCN_seed42.pt`, threshold `0.57`), RankGauss pipeline, and all Stage 1–5C artifacts verified untouched via SHA-256 checksums (`True`).

---

## Figure Catalog: Sources, Interpretations & Limitations

### Figure 1: Elliptic++ Dataset Overview
- **Files:** [`stage5d_fig1_dataset_overview.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig1_dataset_overview.png) (`593,522` bytes) | [`stage5d_fig1_dataset_overview.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig1_dataset_overview.pdf) (`48,557` bytes)
- **Verified Data Sources:** `processed/temporal_snapshots.pt`, `processed/metadata.json`, `reports/dataset_inspection_report.json`, `reports/stage5/stage5c_fraud_ring_analysis.json`
- **What It Shows:** Visualizes total scale (203,769 nodes, 234,355 strictly intra-timestep edges), class composition (77.15% Unknown, 20.62% Licit, 2.23% Illicit), 182-feature breakdown (93/72/17), and per-snapshot prevalence across all 49 timesteps.
- **Interpretation & Limitations:** Unknown (-1) transactions constitute 77.15% of nodes; zero cross-snapshot transaction links exist in the Elliptic++ graph.

### Figure 2: Model Performance Comparison (Validation vs. Test)
- **Files:** [`stage5d_fig2_model_performance.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig2_model_performance.png) (`377,487` bytes) | [`stage5d_fig2_model_performance.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig2_model_performance.pdf) (`44,960` bytes)
- **Verified Data Sources:** `reports/stage3_baselines_metrics.json`, `reports/stage4a_gnn_static_metrics.json`, `reports/stage4c_evolvegcn_metrics_recalibrated.json`, `reports/stage4d_nextgen_experiments_metrics.json`, `reports/stage4h/stage4h_metrics.json`
- **What It Shows:** Compares classical tabular models, static/temporal early GNNs, Local-93 baselines, hybrid GNN+HGB, and the Stage 4H champion ExpA2_RankGauss182_BiDirGCN across Validation (35-41) and Test (42-49) PR-AUC, ROC-AUC, precision, recall, and F1.
- **Interpretation & Limitations:** All architectures exhibit substantial validation-to-test PR-AUC drop driven by post-shutdown Steps 43-49; ExpA2 achieves the highest Test PR-AUC (0.6023) while HistGB attains slightly higher Test F1 (0.6166 vs. 0.5972) via higher precision (88.53%).

### Figure 3: Feature Importance Comparison (HGB Permutation vs. BiDirGCN IG)
- **Files:** [`stage5d_fig3_feature_importance.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig3_feature_importance.png) (`401,552` bytes) | [`stage5d_fig3_feature_importance.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig3_feature_importance.pdf) (`40,319` bytes)
- **Verified Data Sources:** `results/stage5/stage5a_hgb_permutation_importance.csv`, `results/stage5/stage5a_gnn_integrated_gradients.csv`, `results/stage5/stage5a_feature_group_comparison.csv`
- **What It Shows:** Presents top-12 Validation features and group shares for HistGB PR-AUC permutation importance alongside BiDirGCN Integrated Gradients magnitude (|Graph IG|).
- **Interpretation & Limitations:** These two metrics measure fundamentally different quantities (predictive PR-AUC drop vs. signed input-to-logit sensitivity); collinear features split permutation drops differently from gradient attributions.

### Figure 4: Individual Fraud Prediction Explanations (Validation TP/FP/FN/TN)
- **Files:** [`stage5d_fig4_individual_explanations.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig4_individual_explanations.png) (`673,197` bytes) | [`stage5d_fig4_individual_explanations.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig4_individual_explanations.pdf) (`42,964` bytes)
- **Verified Data Sources:** `reports/stage5/stage5b_individual_explanations.json`, `reports/stage5/stage5b_ig_final_validation.md`
- **What It Shows:** Displays positive (fraud-pulling) and negative (licit-pulling) signed IG feature attributions and self vs. neighbor logit balances for representative Validation TP, FP, FN, and TN transactions.
- **Interpretation & Limitations:** Attributions explain how the frozen GNN produced its score relative to a zero RankGauss reference rather than providing legal proof of fraud; Stage 5B M=200 completeness audit guarantees <=0.90% relative error across all 12 audited cases.

### Figure 5: Directed Suspicious Subgraph & Candidate Fraud-Ring Networks
- **Files:** [`stage5d_fig5_suspicious_subgraphs.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig5_suspicious_subgraphs.png) (`1,589,472` bytes) | [`stage5d_fig5_suspicious_subgraphs.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig5_suspicious_subgraphs.pdf) (`68,851` bytes)
- **Verified Data Sources:** `results/stage5/stage5c_candidate_subgraphs.csv`, `results/stage5/stage5c_cluster_nodes.csv`, `processed/temporal_snapshots.pt`
- **What It Shows:** Renders 4 representative intra-snapshot directed candidate networks (Validation C_t35_002 and C_t41_002; Exploratory Test C_t42_001 and C_t45_001) preserving src -> dst transfer arrows, hub highlights, and post-hoc I/L/? labels.
- **Interpretation & Limitations:** C_t35_002 (23/24 Illicit) and C_t42_001 (30/38 Illicit) confirm coordinated illicit structures, whereas post-shutdown C_t45_001 (25/25 Unknown) shows candidate clusters without published illicit labels.

### Figure 6: Fraud-Ring Candidate Structural & Score Statistics
- **Files:** [`stage5d_fig6_fraud_ring_statistics.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig6_fraud_ring_statistics.png) (`418,918` bytes) | [`stage5d_fig6_fraud_ring_statistics.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig6_fraud_ring_statistics.pdf) (`73,869` bytes)
- **Verified Data Sources:** `results/stage5/stage5c_candidate_subgraphs.csv`, `results/stage5/stage5c_group_statistics.csv`
- **What It Shows:** Separates Isolated Singletons, Suspicious Pairs, Multi-Node Non-Candidate Clusters, and Potential Fraud-Ring Candidates, alongside motif breakdowns and RingScore vs. cluster size across Validation and Exploratory Test.
- **Interpretation & Limitations:** Distinguishes isolated and 2-node alerts from multi-step candidate rings; candidacy relies strictly on frozen predictions (P >= 0.57) and topology without label supervision.

### Figure 7: Temporal Distribution & Step 42 -> Step 43 Regime Shift
- **Files:** [`stage5d_fig7_temporal_distribution.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig7_temporal_distribution.png) (`481,709` bytes) | [`stage5d_fig7_temporal_distribution.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig7_temporal_distribution.pdf) (`42,140` bytes)
- **Verified Data Sources:** `results/stage5/stage5c_temporal_group_analysis.csv`, `reports/stage4h/stage4h_metrics.json`
- **What It Shows:** Tracks direct suspicious clusters and candidates across all 49 snapshots (Panel A) alongside per-snapshot Test PR-AUC/ROC-AUC trajectories from Steps 42 to 49 (Panel B).
- **Interpretation & Limitations:** Demonstrates the structural shock at Step 42->43 following darknet shutdown: ExpA2 PR-AUC drops from 0.8975 (Step 42) to 0.0469 (Step 43), while flagged cluster counts drop sharply after Step 42.

### Figure 8: Model Limitations & Post-Shutdown Detection Collapse
- **Files:** [`stage5d_fig8_model_limitations.png`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig8_model_limitations.png) (`382,503` bytes) | [`stage5d_fig8_model_limitations.pdf`](file:///C:/Users/mansi/Downloads/Capstone_Fraud_Detection/figures/stage5/stage5d_fig8_model_limitations.pdf) (`45,708` bytes)
- **Verified Data Sources:** `results/stage5/stage5c_group_statistics.csv`, `results/stage5/stage5c_candidate_subgraphs.csv`, `reports/stage4h/stage4h_metrics.json`
- **What It Shows:** Quantifies post-hoc ground-truth coordinated ring detection collapse (20/25 at Step 42 vs. 0/9 in Steps 43-49), Unknown (-1) dominance in candidate nodes, and post-shutdown single-node precision/recall degradation.
- **Interpretation & Limitations:** Honest limitation diagnostic: neither ExpA2 nor HistGB generalizes effectively at fixed validation thresholds under the extreme post-shutdown distribution and prevalence shift (Steps 43-49).

