# Stage 4D Independent Audit Report

**Audit Date:** 2026-11-26  
**Scope:** Independent verification of Stage 4D next-generation GNN experiments ([src/models_nextgen.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/models_nextgen.py), [src/train_nextgen.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py)), saved champions ([nextgen_gnn_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/nextgen_gnn_best.pt), [nograph_control_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4d/nograph_control_best.pt), and all 24 per-run checkpoints under `models/stage4d/`), metrics/reports ([stage4d_nextgen_experiments_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4d_nextgen_experiments_metrics.json), [stage4d_nextgen_experiments_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4d_nextgen_experiments_report.md), [stage4d_nextgen_comparison.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/stage4d_nextgen_comparison.png)), and file-integrity preservation across Stages 1–4C.  
**Constraints Respected:** Read-only audit; zero models retrained; zero production checkpoints, scripts, metrics, or Stage 1–4D outputs modified.

---

## Executive Summary & PASS/FAIL Checklist

| # | Audit Requirement | Status | Summary Finding |
|---|---|---|---|
| 1 | Exact 93 features used by `Local-93` and excluded `72` aggregate + `17` domain features | **PASS** | `get_slice_x(x, "local_93")` selects indices `0..92` (`Local_feature_1`–`Local_feature_93`), excluding indices `93..164` (`Aggregate_feature_1`–`Aggregate_feature_72`) and indices `165..181` (17 BTC/tx-degree domain columns). |
| 2 | Zero target labels, future info, or test stats in feature selection, preprocessing, or training | **PASS** | `local_93` is a fixed a priori Elliptic schema slice; training loops strictly over Steps `1–34`; warm-up/validation only inspect Steps `1–41`. |
| 3 | Imputation & `StandardScaler` fitted strictly on Training Steps `1–34` | **PASS** | Verified in [preprocess_graph.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/preprocess_graph.py) and empirically on [temporal_snapshots.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/processed/temporal_snapshots.pt) (`mean_abs_max = 0.0004`, `std ∈ [0.9991, 1.0009]` on Steps `1–34` vs `mean_abs_max = 2.2582`, `std_max = 10.1913` on Val `35–41`). |
| 4 | Architecture, seed & threshold selection strictly on Validation Steps `35–41` | **PASS** | Early stopping, `best_state`, `shortlist_summary`, champion selection, and `0.10–0.90` / `0.01–0.99` / `rec30` thresholds use Validation `Steps 35–41` (`7,829` samples) exclusively; Test `42–49` is rolled out post-selection only. |
| 5 | Reproduce `Graph Champion` and `No-Graph Control` test metrics from saved checkpoints | **PASS** | Both [nextgen_gnn_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/nextgen_gnn_best.pt) and [nograph_control_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4d/nograph_control_best.pt) reproduce identical predictions and metrics with zero discrepancies. |
| 6 | Verify `PR-AUC`, `ROC-AUC`, precision, recall, F1, Brier, ECE, TP, FP, TN, FN | **PASS** | Exact 4-decimal and integer match across Validation `35–41`, Test (`0.10–0.90`, `0.01–0.99`, `rec30`), and all 8 individual Test snapshots (`42–49`, zero diffs). |
| 7 | Selected seed & checkpoint match validation-based selection logic | **PASS** | `E6_Local93_GCNAvg` selected `seed=42` (Val PR-AUC `0.9051` > `0.9035`, `0.9034`) and `E6_Local93_NoGraph` selected `seed=456` (Val PR-AUC `0.8999` > `0.8927`, `0.8911`); saved weights match `E6_Local93_*_seed*.pt` tensor-for-tensor. |
| 8 | Fair comparison vs `HistGradientBoosting` & `Random Forest` (threshold-independent vs threshold-dependent) | **PASS** | Evaluated on identical `8,841`-node Test split (`408` illicit / `8,433` licit). Both tabular models still lead Stage 4D (`PR-AUC` `0.5646` / `0.5458` vs `0.5268`; `Brier` `0.0362` / `0.0365` vs `0.0740`; `F1` `0.6166` / `0.6117` vs `0.5965`). |
| 9 | Investigate whether `Local-93` gain is genuinely attributable to removing aggregate features | **PASS (Causal Caveat)** | Empirical drift shows `Aggregate_72` dominates Train→Val mean shift (`max = 2.2582`, top 8/8 shifted cols) and `>5σ` tail inflation (`4.3×`). However, single-factor causation is **not** isolated because `Domain_17` was dropped simultaneously and regularization/capacity were not varied across `182` vs `93` inputs. |
| 10 | Consistency across Markdown report, JSON metrics, and comparison PNG | **PASS** | All screening (`14`), multi-seed (`5×3`), champion, delta, and plot values match across all three artifacts. |
| 11 | Preservation of all Stage 1–4C checkpoints, scripts, metrics & reports | **PASS** | Stage 1–4C artifacts retain their pre-Stage 4D modification timestamps and byte sizes untouched. |

---

## 1. Feature Partition Verification (`Local-93` vs. Excluded `89` Features)

Verified from [processed/metadata.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/processed/metadata.json) (`182` total `feature_columns`) and `get_slice_x()` in [src/train_nextgen.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L230-L231):

- **Included by `feat_mode == "local_93"` (`x[:, :93]`, 93 columns):**
  - Indices `0..92`: `Local_feature_1` through `Local_feature_93` (verified `LOCAL_93_ALL_MATCH: True`).
- **Excluded by `feat_mode == "local_93"` (`89` columns total = `72` aggregate + `17` domain):**
  - **72 Aggregate features (indices `93..164`):** `Aggregate_feature_1` through `Aggregate_feature_72` (verified `AGG_72_ALL_MATCH: True`).
  - **17 Domain features (indices `165..181`):** `in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, `in_BTC_min`, `in_BTC_max`, `in_BTC_mean`, `in_BTC_median`, `in_BTC_total`, `out_BTC_min`, `out_BTC_max`, `out_BTC_mean`, `out_BTC_median`, `out_BTC_total`.

---

## 2. Zero-Leakage & Preprocessing Audit (Checks 2–4)

1. **A priori feature slice:** `local_93` vs `full_182` is hardcoded by schema index (`x[:, :93]`) in [src/train_nextgen.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L230-L231) and [build_experiment_catalog()](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L461-L652). No target correlations, model feature importances, validation scores, or test statistics were used to choose those 93 columns.
2. **Train-only imputation & scaling:** [preprocess_graph.py](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/preprocess_graph.py) fits `SimpleImputer(strategy="median")` and `StandardScaler()` exclusively on `X_raw[is_train_time]` (`Time step` `1–34`). Independent numerical verification on `temporal_snapshots.pt` (`203,769 × 182` total nodes across 49 steps) confirmed:
   - **Train Steps `1–34` (`snapshots[:34]`, all nodes):** `max(|mean|) = 0.0004`, `std ∈ [0.999107, 1.000931]` (zero-mean, unit-variance up to float32 storage precision).
   - **Validation Steps `35–41` (`snapshots[34:41]`, all nodes):** `max(|mean|) = 2.258219`, `std ∈ [0.112507, 10.191339]` (confirming validation statistics were not pooled into the scaler).
3. **Chronological split counts:** Verified in [tabular_data.npz](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/processed/tabular_data.npz) and snapshot tensors:
   - **Train (Steps `1–34`):** `29,894` labeled nodes
   - **Validation (Steps `35–41`):** `7,829` labeled nodes (`675` illicit, `7,154` licit)
   - **Test (Steps `42–49`):** `8,841` labeled nodes (`408` illicit, `8,433` licit)
4. **Validation-only selection & threshold tuning:**
   - Inside [train_single_experiment()](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L322-L454), optimization iterates strictly over `for t in range(34)` (`Steps 1–34`).
   - Per-epoch checkpointing and early stopping call [rollout_val_only()](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L234-L272), which warms up state on `range(34)` and computes probabilities only on `range(34, 41)` (`Steps 35–41`).
   - Both legacy (`0.10–0.90`, 81 grid points) and recalibrated (`0.01–0.99`, 99 grid points) F1-optimal and recall-constrained (`min_precision=0.30`) thresholds are computed via [select_thresholds_val_only()](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L138-L160) exclusively from `y_val_all` and `y_val_prob_best`.
   - [rollout_test_post_selection()](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L275-L315) is executed only inside [evaluate_champion_on_test()](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/train_nextgen.py#L663-L727) after shortlist multi-seed ranking and champion selection are complete.

---

## 3. Checkpoint Verification & Metric Reproduction (Checks 5–7)

Both saved champion checkpoints were loaded independently on CPU and re-evaluated end-to-end across Validation (`35–41`) and Test (`42–49`). Every reported metric, confusion-matrix count, calibration statistic, and per-snapshot breakdown matched [stage4d_nextgen_experiments_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4d_nextgen_experiments_metrics.json) with **zero differences**.

### 3.1 Seed & Checkpoint Correspondence

- **Graph Champion (`E6_Local93_GCNAvg`):**
  - Validation PR-AUC across seeds `[42, 123, 456]`: **`0.905072` (`0.9051`, seed 42)**, `0.903363` (`0.9034`, seed 123), `0.903525` (`0.9035`, seed 456). Mean ± std = `0.9040 ± 0.0008`.
  - Highest-Validation seed selected: **`seed = 42`**.
  - Tensor-level equality check (`torch.equal` across all keys in `model_state_dict`) between [nextgen_gnn_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/nextgen_gnn_best.pt) and [E6_Local93_GCNAvg_seed42.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4d/E6_Local93_GCNAvg_seed42.pt): **`True`**.
- **No-Graph Control Champion (`E6_Local93_NoGraph`):**
  - Validation PR-AUC across seeds `[42, 123, 456]`: `0.892696` (`0.8927`, seed 42), `0.891138` (`0.8911`, seed 123), **`0.899941` (`0.8999`, seed 456)**. Mean ± std = `0.8946 ± 0.0038`.
  - Highest-Validation seed selected: **`seed = 456`**.
  - Tensor-level equality check (`torch.equal`) between [nograph_control_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4d/nograph_control_best.pt) and [E6_Local93_NoGraph_seed456.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4d/E6_Local93_NoGraph_seed456.pt): **`True`**.
- **Overall Champion:** `E6_Local93_GCNAvg` (`val_prauc_mean = 0.9040` > `0.8946`), correctly selected from the combined graph + no-graph multi-seed shortlist.

### 3.2 Independently Reproduced Metrics (Test Steps `42–49`, `n = 8,841`)

| Champion | Grid / $\tau^*$ | Precision | Recall | F1 | PR-AUC | ROC-AUC | Brier | ECE (10-bin) | TP | FP | TN | FN | JSON Diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Graph (`E6_Local93_GCNAvg`, seed 42)** | `0.10–0.90` / `0.90` | `75.20%` | `46.08%` | `0.5714` | `0.5268` | `0.8188` | `0.0740` | `0.1145` | `188` | `62` | `8,371` | `220` | Exact match |
| **Graph (`E6_Local93_GCNAvg`, seed 42)** | `0.01–0.99` / `0.93` | `85.39%` | `45.83%` | `0.5965` | `0.5268` | `0.8188` | `0.0740` | `0.1145` | `187` | `32` | `8,401` | `221` | Exact match |
| **Graph (`E6_Local93_GCNAvg`, seed 42)** | `rec30` / `0.22` | `13.55%` | `63.48%` | `0.2233` | `0.5268` | `0.8188` | `0.0740` | `0.1145` | `259` | `1,653` | `6,780` | `149` | Exact match |
| **No-Graph (`E6_Local93_NoGraph`, seed 456)** | `0.10–0.90` / `0.90` | `81.82%` | `44.12%` | `0.5732` | `0.4591` | `0.7980` | `0.0685` | `0.1124` | `180` | `40` | `8,393` | `228` | Exact match |
| **No-Graph (`E6_Local93_NoGraph`, seed 456)** | `0.01–0.99` / `0.93` | `88.06%` | `43.38%` | `0.5813` | `0.4591` | `0.7980` | `0.0685` | `0.1124` | `177` | `24` | `8,409` | `231` | Exact match |
| **No-Graph (`E6_Local93_NoGraph`, seed 456)** | `rec30` / `0.18` | `11.59%` | `61.03%` | `0.1948` | `0.4591` | `0.7980` | `0.0685` | `0.1124` | `249` | `1,899` | `6,534` | `159` | Exact match |

All 8 individual Test snapshot checks (`Steps 42–49`, comparing `PR-AUC`, `ROC-AUC`, `illicit_precision`, `illicit_recall`, `illicit_f1`, `tp`, `fp`, `fn`, `tn`) produced `PER_SNAPSHOT_TEST_DIFFS: []`. Notice that on Step 42 (`pre-shutdown`), `E6_Local93_GCNAvg` achieves `PR-AUC = 0.8829` and `F1 = 0.8698` (`TP=187, FP=4, FN=52`), whereas in Steps `43–49` (`post-dark-market-shutdown`), at $\tau^*=0.93$ both champions achieve `TP=0` (with low false-alarm counts: `FP=28` total across Steps 43–49 for `GCNAvg` and `FP=22` for `NoGraph`).

---

## 4. Stage 4D vs. Tabular Baselines on Identical Test Split (`Steps 42–49`) (Check 8)

All models below were evaluated on the exact same `8,841` Test nodes (`408` illicit, `8,433` licit). Tabular Brier/ECE calibration scores were newly computed from [random_forest.joblib](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/random_forest.joblib) and [hist_gradient_boosting.joblib](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/hist_gradient_boosting.joblib) to place them on a strictly equal footing with Stage 4D.

### 4.1 Threshold-Independent Metrics (Ranking & Probability Calibration)

| Model | Features | Graph | Test PR-AUC | Test ROC-AUC | Test Brier ↓ | Test ECE (10-bin) ↓ |
|---|---|---|---|---|---|---|
| **HistGradientBoosting (Stage 3)** | `182` | No | **`0.5646`** | **`0.8848`** | **`0.0362`** | **`0.0396`** |
| **Random Forest (Stage 3)** | `182` | No | `0.5458` | `0.8351` | `0.0365` | `0.0481` |
| **Stage 4D Graph Champion (`E6_Local93_GCNAvg`)** | `93` | Yes | `0.5268` | `0.8188` | `0.0740` | `0.1145` |
| **Stage 4D No-Graph Control (`E6_Local93_NoGraph`)** | `93` | No | `0.4591` | `0.7980` | `0.0685` | `0.1124` |

- **Observation on Graph Value:** Holding architecture and the `Local-93` feature set fixed, directed GCN message passing (`E6_Local93_GCNAvg`) improves **threshold-independent ranking** over `E6_Local93_NoGraph` by **`+0.0677` PR-AUC** (`0.5268` vs `0.4591`) and **`+0.0208` ROC-AUC** (`0.8188` vs `0.7980`).
- **Observation on Calibration:** Stage 4D models train with `pos_weight = 7.63` weighted BCE (`base_bce`), inflating positive log-odds (`mean_prob_licit = 0.1412` on Test vs base rate `0.0461`), which explains their higher `Brier` (`0.0740` vs `0.0362`) and `ECE` (`0.1145` vs `0.0396`) compared to `HistGradientBoosting`.

### 4.2 Threshold-Dependent Metrics (Validation-Tuned Operating Points)

| Model | Threshold Source | $\tau^*$ | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|
| **HistGradientBoosting** | Stage 3 Val `35–41` | `0.79` | **`88.53%`** | **`47.30%`** | **`0.6166`** | `193` | `25` | `8,408` | `215` |
| **Random Forest** | Stage 3 Val `35–41` | `0.59` | `86.55%` | `47.30%` | `0.6117` | `193` | `30` | `8,403` | `215` |
| **Stage 4D Graph Champion** | Val `0.01–0.99` | `0.93` | `85.39%` | `45.83%` | `0.5965` | `187` | `32` | `8,401` | `221` |
| **Stage 4D No-Graph Control** | Val `0.01–0.99` | `0.93` | `88.06%` | `43.38%` | `0.5813` | `177` | `24` | `8,409` | `231` |
| *HistGradientBoosting (default)* | *Uncalibrated 0.50* | *0.50* | *52.63%* | *49.02%* | *0.5076* | *200* | *180* | *8,253* | *208* |
| *Random Forest (default)* | *Uncalibrated 0.50* | *0.50* | *70.50%* | *48.04%* | *0.5714* | *196* | *82* | *8,351* | *212* |

**Key Audit Takeaway:** Stage 4D closed most of the Stage 4C neural/graph gap (lifting Test PR-AUC from Stage 4C's `0.3107` to `0.5268` and F1 from `0.2251` to `0.5965`), **but does not surpass `HistGradientBoosting`** (`PR-AUC 0.5646`, `ROC-AUC 0.8848`, `F1 0.6166`) or `Random Forest` (`PR-AUC 0.5458`, `F1 0.6117`). Meanwhile, on `Local-93`, graph propagation delivers a clear, measurable Test ranking gain (`+0.0677` PR-AUC, `+10` TP, `+0.0152` F1) over its No-Graph twin.

---

## 5. Investigation of `Local-93` vs. `Full-182` Improvement (Check 9 — Causal Evidence Assessment)

In Stage 4D, switching from `Full-182` to `Local-93` produced massive Validation jumps across all `SeparatedGraphNet` topologies:
- `GCNAvg`: Val PR-AUC `0.6967` (`Full-182`) $\rightarrow$ `0.9051` (`Local-93`, seed 42) [`+0.2084`]
- `NoGraph`: Val PR-AUC `0.7196` (`Full-182`) $\rightarrow$ `0.8927` (`Local-93`, seed 42) [`+0.1731`]
- `SAGEConcat`: Val PR-AUC `0.5440` (`Full-182`) $\rightarrow$ `0.8120` (`Local-93`, seed 42) [`+0.2680`]

### 5.1 Pre-Test Feature-Drift Evidence (`Train Steps 1–34` vs. `Val Steps 35–41`, All Nodes)

To avoid touching Test data during diagnostic attribution, feature shift was measured across all nodes in `Train (1–34)` vs `Validation (35–41)`:

| Feature Group | Indices (#) | Median Mean Shift | Max Mean Shift | Median Std Ratio | Max Std Ratio | `frac(\|x\| > 5)` Train → Val |
|---|---|---|---|---|---|---|
| **`Local_93`** | `0..92` (`93`) | `0.0614` | `1.0127` | `0.8070` | `10.1921` | `0.00360 → 0.00493` (`1.37×`) |
| **`Aggregate_72`** | `93..164` (`72`) | `0.0603` | **`2.2582`** | `0.9219` | `7.7089` | `0.00232 →` **`0.01006`** (`4.34×`) |
| **`Domain_17`** | `165..181` (`17`) | `0.0176` | `0.8662` | `0.7298` | `4.1811` | `0.00367 → 0.00500` (`1.36×`) |

- **All 8 largest Train→Val mean shifts across the entire 182-feature set belong to `Aggregate_feature_*`:** `Aggregate_feature_8` (`2.258`), `Aggregate_feature_10` (`2.249`), `Aggregate_feature_7` (`2.180`), `Aggregate_feature_43` (`1.961`), `Aggregate_feature_46` (`1.824`), `Aggregate_feature_13` (`1.771`, `std_ratio = 7.709×`), `Aggregate_feature_16` (`1.730`, `std_ratio = 6.880×`), `Aggregate_feature_52` (`1.661`).
- Extreme tail activations (`|z| > 5`) increased **`4.34×`** in `Aggregate_72` from Train to Validation (compared to `1.37×` in `Local_93` and `1.36×` in `Domain_17`).
- Crucially, `E6_Full182_GCNAvg` *underperformed* `E6_Full182_NoGraph` (`0.6967` vs `0.7196`), whereas `E6_Local93_GCNAvg` *outperformed* `E6_Local93_NoGraph` (`0.9051` vs `0.8927` on Val; `0.5268` vs `0.4591` on Test), consistent with redundant/shifted 1-hop `Aggregate_72` summaries oversmoothing or destabilizing learned GCN aggregation.

### 5.2 Why Strict Single-Factor Causation Cannot Be Claimed Yet

Despite strong corroborating evidence, Stage 4D does **not** prove by controlled experiment alone that removing `Aggregate_72` is the sole cause of the gain:
1. **Confounded removal with `Domain_17`:** `get_slice_x(x, "local_93")` drops **both** `Aggregate_72` (`93:165`) and `Domain_17` (`165:182`) simultaneously. Stage 4D did not test intermediate ablations (`Local93 + Domain17 = 110` features, or `Local93 + Aggregate72 = 165` features), and `Domain_17` itself contains median-imputed missing entries and heavy-tailed BTC features (e.g., `fees` `std_ratio = 4.181×` in Val, alongside `Local_feature_34/37/35` at `10.19×/6.72×/5.99×`).
2. **Confounded optimization / parameter capacity:** `E6_Full182_GCNAvg` early-stopped at **Epoch 4** with `Brier = 0.2115, ECE = 0.3977` (vs Epoch **40** and `Brier = 0.0657, ECE = 0.1274` for `E6_Local93_GCNAvg`) under identical `lr = 0.002` and `weight_decay = 1e-3`. Higher parameter count (`66,051` vs `43,267` trainable parameters for `GCNAvg`; `34,177` vs `22,785` for `NoGraph`), multicollinearity, or insufficient input regularization on `Full-182` may explain part of the gap.

---

## 6. Cross-Artifact Consistency & Stage 1–4C Preservation (Checks 10–11)

- **Cross-Artifact Agreement:**
  - Verified all 14 screening rows, 5 shortlisted multi-seed summaries (`[42, 123, 456]`), per-seed runs, and post-selection Test metrics in [stage4d_nextgen_experiments_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4d_nextgen_experiments_report.md) against [stage4d_nextgen_experiments_metrics.json](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4d_nextgen_experiments_metrics.json) (exact match).
  - Visually inspected [stage4d_nextgen_comparison.png](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/stage4d_nextgen_comparison.png): left panel bars (`14` screening runs, blue Graph vs gray No-Graph, Stage 4C dashed line at `0.6387`) and right panel bars (`5` shortlisted multi-seed mean ± std bars) agree with the JSON and Markdown tables.
- **Stage 1–4C File Preservation:** Verified untouched file sizes and modification timestamps across Stage 1–4C scripts (`train_evolvegcn.py`, `recalibrate_evolvegcn.py`), checkpoints (`evolvegcn_h.pt`, `gcn_lstm.pt`, `gcn_lstm_historical.pt`, `gcn_static_improved.pt`, `hist_gradient_boosting.joblib`, `random_forest.joblib`, `logistic_regression.joblib`), and reports (`stage3_baselines_metrics.json`, `stage4c_evolvegcn_metrics.json`, `stage4c_evolvegcn_recalibrated_metrics.json`, `stage4c_audit_report.md`).
- **Stage 4D Checkpoint Inventory:** Verified `25` checkpoints under `models/stage4d/` (`24` `<exp_id>_seed<seed>.pt` checkpoints covering `14` screening runs + `10` additional shortlist seed runs, plus `nograph_control_best.pt`) and [nextgen_gnn_best.pt](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/nextgen_gnn_best.pt).
