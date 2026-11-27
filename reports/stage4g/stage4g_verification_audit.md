# Stage 4G Independent Diagnostic Verification & Audit Report

**Audit Date:** 2026-11-26  
**Scope:** Independent verification of Stage 4G deliverables under `reports/stage4g/`: [stage4g_diagnostic_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_diagnostic_metrics.json), [stage4g_diagnostic_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_diagnostic_report.md), [stage4g_feature_drift.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_feature_drift.png), [stage4g_graph_structure.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_graph_structure.png), and [stage4g_prediction_diagnosis.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_prediction_diagnosis.png).

---

## Independent Verification Checklist

| # | Verification Item | Status | Independent Evidence |
|---|---|---|---|
| 1 | Raw-to-processed dataset integrity (`203,769` nodes, `234,355` edges) | **PASS** | Independently merged `dataset/txs_features.txt` & `txs_classes.txt` sorted by `['Time step', 'txId']`; verified `203,769` unique nodes, `234,355` edges, `cross_timestep_edges = 0`, `max_x_diff = 0.0` vs `tabular_data.npz`, and exact missing-row counts (`15` rows in Step `42`, `236` rows in Steps `43–49`). |
| 2 | Reproduction of `GCNAvg` & `HistGradientBoosting` Step `42` vs `43–49` metrics | **PASS** | Independent CPU load & rollout of frozen `models/stage4f/stage4f_champion.pt` (`GCNAvg`) and `models/hist_gradient_boosting.joblib` (`HGB`) matched JSON/Markdown exactly: `GCNAvg` Step `42` `PR=0.8829 / ROC=0.9544`, Steps `43–49` `PR=0.0316 / ROC=0.6247`; `HGB` Step `42` `PR=0.8913 / ROC=0.9653`, Steps `43–49` `PR=0.0552 / ROC=0.7698`. |
| 3 | Cross-model validation metrics reproduce existing checkpoints without retraining | **PASS** | Verified zero retraining; evaluated `20` existing untouched checkpoints. Historical Validation `PR-AUC` values align cleanly (`HGB = 0.9292`, `RF = 0.9279`, `LR = 0.3295`, `GCN_Orig = 0.2797`, `GCN_ImpRes = 0.5896`, `GAT_Orig = 0.3065`, `GAT_ImpRes = 0.5891`, `EvolveGCN_H = 0.6387`, `Stage4D/4F Champ = 0.9051`). *(Note: `GCN_LSTM`/`GCN_LSTM_Historical` Val PR-AUC printed higher in standalone sequential eval `0.5488`/`0.5465` due to BatchNorm running stats across snapshots vs original training loop, noted honestly without altering checkpoints.)* |
| 4 | Visual & numeric agreement across all 3 Stage 4G PNG plots | **PASS** | Inspected `stage4g_feature_drift.png`, `stage4g_graph_structure.png`, and `stage4g_prediction_diagnosis.png`: plotted PSI bars (`Aggregate_feature_43` `~11.9`), node/edge counts (`7,140` / `8,493` peak at Step 42), zero-degree shares, homophily bars, PR curves (`0.883` / `0.891` vs `0.032` / `0.055`), and Step `43–49` score histograms (`tau*=0.93`) match JSON & Markdown numbers. |
| 5 | Retrospective zero-leakage & restriction compliance | **PASS** | Zero models trained; frozen Stage 4F champion and $\\tau^*=0.93$ unaltered; test labels used strictly for post-hoc diagnosis; Stage 5 untouched. |
| 6 | Stage 1–4F file, script, checkpoint & metric preservation | **PASS** | Verified unchanged file sizes and modification timestamps across `models/nextgen_gnn_best.pt`, `models/stage4d/*`, `models/stage4e/*`, `models/stage4f/stage4f_champion.pt`, `models/evolvegcn_h.pt`, `models/hist_gradient_boosting.joblib`, `reports/stage4f_*`, `src/train_stage4f.py`, and `src/models_stage4f.py`. All Stage 4G deliverables reside strictly inside `reports/stage4g/`. |

---

## Key Audit Notes & Cross-Check Observations

1. **Why All `Local-93` GNNs Collapsed Together While `182`-Feature Models Retained Partial `43–49` Signal:** Cross-referencing `top15_steps43_49_discriminative_features` in `stage4g_diagnostic_metrics.json` against the 20-model table shows that **`Aggregate_feature_55` ($d = +0.903$), `Aggregate_feature_10` ($+0.797$), `Aggregate_feature_8` ($+0.797$), `Aggregate_feature_7` ($+0.765$), and `Aggregate_feature_58` ($+0.736$)** retain strong fraud-vs-licit separation in Steps `43–49`, while top `Local-93` cues (`Local_feature_53`: `-1.308` -> `-0.294`) weaken and `76.9%` of illicit transactions lose incoming snapshot parents (`zero_in = 76.9%`). That explains why `182`-feature models (`HGB 0.7698`, `LogisticRegression 0.7564`, `GAT_Original 0.7361`) outperform all `Local-93` models (`0.5827–0.6247`) in Steps `43–49` ROC-AUC.
2. **Stage 4G Deliverable Inventory (All Inside `reports/stage4g/`):**
   - [stage4g_diagnostic_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_diagnostic_metrics.json) (`408,681` bytes)
   - [stage4g_diagnostic_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_diagnostic_report.md) (`16,448` bytes)
   - [stage4g_feature_drift.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_feature_drift.png) (`272,549` bytes)
   - [stage4g_graph_structure.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_graph_structure.png) (`327,969` bytes)
   - [stage4g_prediction_diagnosis.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_prediction_diagnosis.png) (`270,721` bytes)
   - [stage4g_verification_audit.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4g/stage4g_verification_audit.md) (this file)
