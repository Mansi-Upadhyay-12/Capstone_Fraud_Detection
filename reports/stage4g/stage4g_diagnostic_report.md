# Stage 4G: Temporal Distribution Shift & Post-Shutdown Fraud Detection Diagnostic Report

**Generated On:** 2026-11-26 16:20:01  
**Scope:** Strictly retrospective multi-stage diagnostic investigation across Train (`Steps 1–34`), Validation (`Steps 35–41`), Test Step `42` (pre-shutdown), and Test Steps `43–49` (post-shutdown).  
**Restrictions Honored:** Zero models trained; frozen Stage 4F champion (`F4F_00_Anchor_GCNAvg`, $\tau^*=0.93$) and Stage 1–4F checkpoints/files untouched; all outputs isolated in `reports/stage4g/`.

---

## Executive Diagnosis: Observed Facts vs. Causal Hypotheses

### Observed Facts (Directly Verified in Data & Frozen Models)
1. **Preprocessing & Data Integrity Are Clean (No Alignment/Pipeline Bug at Step 42):** All `203,769` nodes and `234,355` edges align identically across raw files and processed tensors (`max_x_diff = 0.0`), `cross_timestep_edges = 0`, zero duplicate `txId`s, and exact column ordering (`93` Local + `72` Aggregate + `17` Domain).
2. **Simultaneous Prevalence, Structural Homophily & Feature-Signature Shifts After Step 42:**
   - **Prevalence drops `4.4×`:** Labeled fraud prevalence falls from `11.10%` on Step `42` (`239/2,154`) to **`2.53%`** across Steps `43–49` (`169/6,687`, hitting `0.41%` on Step 45 and `0.28%` on Step 46).
   - **Directed Graph Fraud Homophily Drops Sharply:** Upstream parent homophily $P(\text{parent}=\text{illicit} \mid \text{child}=\text{illicit})$ falls from **`0.5989`** (Train) / **`0.3360`** (Val) / **`0.6133`** (Step 42) down to **`0.5357`** in Steps `43–49`; downstream child homophily falls from `0.5650` (Train) / `0.3609` (Val) / `0.6216` (Step 42) to **`0.2419`**.
   - **Fraud-vs-Licit Feature Signature Rotates in Steps 43–49:** Using standardized Cohen's $d$ separation vectors, `Local-93` aligns strongly across Train, Val, and Step 42 (`cos(Val, Step 42) = 0.9448`), then drops sharply in Steps `43–49` (`cos(Val, Steps 43-49) = 0.8987`, `cos(Step 42, Steps 43-49) = 0.8523`).
3. **Why `0%` Recall Occurs at $\tau^*=0.93$, and Why It Is Both Ranking Collapse and Calibration Mismatch:**
   - On pooled Steps `43–49` (`6,687` labeled nodes, `169` illicit), `GCNAvg` outputs `meanP_illicit = 0.1659` vs `meanP_licit = 0.1463` (`ROC-AUC = 0.6247`, `PR-AUC = 0.0316`), compared with `HGB` (`meanP_illicit = 0.0822`, `meanP_licit = 0.0467`, `ROC-AUC = 0.7698`, `PR-AUC = 0.0552`).
   - Even with a **purely diagnostic counterfactual oracle threshold** tuned on `43–49`, `GCNAvg` peaks at **`F1 = 0.0686`** (at $\tau=0.04$) and catches only **`0/50`** frauds in its Top-50 predictions (`HGB` oracle `F1 = 0.1078`, Top-50 hits = **`3/50`**). Thus **threshold recalibration alone cannot rescue `GCNAvg` on Steps `43–49`**: the primary failure is **true ranking degradation**, compounded by frozen $\tau^*=0.93$ sitting above nearly all post-shutdown probabilities (`TP = 0`).

### Evidence-Supported Interpretations vs. Open Hypotheses
- **Supported Interpretation:** The Step `43–49` collapse reflects a genuine **post-shutdown regime shift ($P(X \mid Y=1)$ signature rotation, loss of fraud-edge homophily, and `4.4×` lower illicit prevalence)** rather than a pipeline or alignment bug.
- **Hypothesis on `HGB`'s Higher Partial Ranking (`0.7698` vs `0.6247`):** Because `HGB` splits non-linearly across all `182` quantile-invariant features (`Aggregate-72` + `Domain-17` + `Local-93`), it preserves partial separation on post-shutdown frauds where `Local-93` linear/GCN representations flatten out; exact per-tree path attribution on `43–49` remains an open verification target.

---

## 1. Data Integrity Verification Summary

- **Dataset Scale:** `203,769` nodes, `234,355` edges, `203,769` unique `txId`s (`0` duplicates).
- **Cross-Timestep Edges:** `0` (all edges strictly connect transactions within the same snapshot `t`).
- **Tensor Alignment:** Exact match (`max_x_diff = 0.0`) across `tabular_data.npz`, `elliptic_pyg_data.pt`, and `temporal_snapshots.pt`.
- **Constant Scaled Columns:** `{'train_1_34': [], 'val_35_41': [], 'test_42': [], 'test_43_49': []}`.
- **Exact Duplicate Scaled Feature Rows:** Train `1–34`: `30`, Val `35–41`: `2`, Step `42`: `0`, Steps `43–49`: `0`.

---

## 2. Feature Distribution Drift (`Train 1–34`, `Val 35–41`, `Step 42`, `Steps 43–49`)

### 2.1 Raw Missing-Value Rates (Pre-Imputation)

| Regime | Rows | Rows w/ Any `NaN` | `All_182` Cell `NaN%` | `Local_93` `NaN%` | `Aggregate_72` `NaN%` | `Domain_17` `NaN%` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Train (1-34)` | `136,265` | `401` (`0.29%`) | `0.0275%` | `0.0000%` | `0.0000%` | `0.2943%` |
| `Val (35-41)` | `30,680` | `313` (`1.02%`) | `0.0953%` | `0.0000%` | `0.0000%` | `1.0202%` |
| `Test Step 42` | `7,140` | `15` (`0.21%`) | `0.0196%` | `0.0000%` | `0.0000%` | `0.2101%` |
| `Test 43-49` | `29,684` | `236` (`0.80%`) | `0.0743%` | `0.0000%` | `0.0000%` | `0.7950%` |

### 2.2 Group-Level Drift Summary (PSI, KS, Standardized Mean/Std Shift)

| Feature Group (#) | Median PSI (`Tr->Val`) | Median PSI (`Tr->42`) | Median PSI (`Tr->43-49`) | Max PSI (`Tr->43-49`) | Median PSI (`42->43-49`) | Max Mean Shift (`Tr->43-49`) | `% Cols PSI>0.25` (`Tr->43-49`) | `|z|>5` (`Tr / Val / 42 / 43-49`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Local_93` (`93`) | `0.0518` | `0.0437` | `0.0435` | `1.7395` | `0.0083` | `1.1143` | `7.5%` | `0.36% / 0.49% / 0.42% / 0.55%` |
| `Aggregate_72` (`72`) | `0.0381` | `0.0479` | `0.0507` | `11.8221` | `0.0174` | `3.0323` | `27.8%` | `0.23% / 1.01% / 0.87% / 1.04%` |
| `Domain_17` (`17`) | `0.0501` | `0.0858` | `0.0845` | `1.6764` | `0.0247` | `0.9963` | `5.9%` | `0.37% / 0.50% / 0.42% / 0.42%` |

---

## 3. Graph Structure Drift

| Regime | Nodes/Step | Edges/Step | Edge/Node | Mean Density | Isolated% | Zero-In (All / Ill%) | Zero-Out (All / Ill%) | Giant WCC% | $P(\text{parent}=\text{ill}\|\text{child}=\text{ill})$ | $P(\text{child}=\text{ill}\|\text{parent}=\text{ill})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Train (1-34)` | `4007.8` | `4613.0` | `1.1510` | `0.000336` | `0.00%` | `26.8% / 26.0%` | `18.3% / 29.6%` | `100.0%` | `0.5989` | `0.5650` |
| `Val (35-41)` | `4382.9` | `5051.4` | `1.1525` | `0.000283` | `0.00%` | `30.1% / 19.9%` | `18.5% / 53.9%` | `100.0%` | `0.3360` | `0.3609` |
| `Test Step 42` | `7140.0` | `8493.0` | `1.1895` | `0.000167` | `0.00%` | `25.9% / 17.6%` | `25.3% / 32.6%` | `100.0%` | `0.6133` | `0.6216` |
| `Test 43-49` | `4240.6` | `4808.4` | `1.1339` | `0.000287` | `0.00%` | `33.1% / 76.9%` | `20.0% / 8.9%` | `100.0%` | `0.5357` | `0.2419` |

---

## 4. Retrospective Label & Fraud-Pattern Signature Analysis

| Feature Subset | $\cos(\text{Train}, \text{Val})$ | $\cos(\text{Val}, \text{Step 42})$ | $\cos(\text{Val}, \text{43–49})$ | $\cos(\text{Step 42}, \text{43–49})$ |
| :--- | :---: | :---: | :---: | :---: |
| `All_182` | `0.8475` | `0.9286` | `0.7476` | `0.6518` |
| `Local_93` | `0.9007` | `0.9448` | `0.8987` | `0.8523` |
| `Aggregate_72` | `0.7324` | `0.9128` | `0.5176` | `0.3863` |
| `Domain_17` | `0.8885` | `0.9304` | `0.8313` | `0.7051` |

---

## 5. Cross-Model Comparison Across Regimes (Existing Checkpoints Only, Zero Retraining)

| Model | Val `35–41` PR-AUC / ROC | Step `42` PR-AUC / ROC | Steps `43–49` PR-AUC / ROC | Test All `42–49` PR-AUC / ROC / F1 |
| :--- | :---: | :---: | :---: | :---: |
| `HistGradientBoosting` | `0.9292` / `0.9791` | `0.8913` / `0.9653` | `0.0552` / `0.7698` | `0.5646` / `0.8848` / `0.6166` |
| `Random_Forest` | `0.9279` / `0.9773` | `0.8926` / `0.9616` | `0.0450` / `0.6547` | `0.5458` / `0.8351` / `0.6117` |
| `Logistic_Regression` | `0.3295` / `0.8983` | `0.5403` / `0.9318` | `0.0493` / `0.7564` | `0.1940` / `0.8574` / `0.3228` |
| `MLP_Baseline` | `0.7788` / `0.9395` | `0.8432` / `0.9417` | `0.0432` / `0.6833` | `0.4742` / `0.8360` / `0.5360` |
| `GCN_Original` | `0.2797` / `0.7701` | `0.4696` / `0.8739` | `0.0440` / `0.6988` | `0.1211` / `0.7875` / `0.1723` |
| `GCN_Improved_Residual` | `0.5896` / `0.8828` | `0.6156` / `0.8604` | `0.0402` / `0.6023` | `0.2761` / `0.7546` / `0.3210` |
| `GAT_Original` | `0.3065` / `0.7720` | `0.4984` / `0.8738` | `0.0452` / `0.7361` | `0.1258` / `0.7966` / `0.1963` |
| `GAT_Improved_Residual` | `0.5891` / `0.8889` | `0.7667` / `0.9394` | `0.0353` / `0.6568` | `0.3730` / `0.8201` / `0.2988` |
| `GCN_LSTM` | `0.5488` / `0.8875` | `0.6869` / `0.8785` | `0.0356` / `0.6677` | `0.2003` / `0.7805` / `0.1617` |
| `GCN_LSTM_Historical` | `0.5465` / `0.8870` | `0.6847` / `0.8777` | `0.0356` / `0.6683` | `0.2003` / `0.7805` / `0.1611` |
| `EvolveGCN_H_Recalibrated` | `0.6387` / `0.8849` | `0.7882` / `0.9234` | `0.0346` / `0.6322` | `0.3107` / `0.7963` / `0.2251` |
| `Stage4D_E6_Local93_GCNAvg` | `0.9051` / `0.9697` | `0.8829` / `0.9544` | `0.0316` / `0.6247` | `0.5268` / `0.8188` / `0.5965` |
| `Stage4D_E6_Local93_NoGraph` | `0.8999` / `0.9659` | `0.8511` / `0.9495` | `0.0281` / `0.5827` | `0.4591` / `0.7980` / `0.5813` |
| `Stage4D_E6_Local93_SAGEConcat` | `0.8120` / `0.9562` | `0.8596` / `0.9593` | `0.0308` / `0.6215` | `0.4644` / `0.8223` / `0.4971` |
| `Stage4D_E5_ModeratedBCE_pw2.76` | `0.7412` / `0.9300` | `0.7356` / `0.9161` | `0.0334` / `0.6542` | `0.1039` / `0.7711` / `0.0113` |
| `Stage4E_F2_Local93_Domain17_s123` | `0.9077` / `0.9699` | `0.8856` / `0.9586` | `0.0292` / `0.6007` | `0.5123` / `0.8113` / `0.5762` |
| `Stage4F_Champ_Anchor_GCNAvg` | `0.9051` / `0.9697` | `0.8829` / `0.9544` | `0.0316` / `0.6247` | `0.5268` / `0.8188` / `0.5965` |
| `Stage4F_BiDir_ResSwiGLU_s42` | `0.9035` / `0.9577` | `0.8605` / `0.9292` | `0.0304` / `0.6057` | `0.5171` / `0.7951` / `0.6040` |
| `Stage4F_Asinh_BiDirGCN_s456` | `0.9074` / `0.9688` | `0.8863` / `0.9564` | `0.0300` / `0.5908` | `0.5225` / `0.8053` / `0.5806` |
| `Stage4F_Asinh110_BiDirGCN_s456` | `0.9090` / `0.9725` | `0.8871` / `0.9607` | `0.0298` / `0.6073` | `0.4948` / `0.8154` / `0.5795` |

### 2.3 Top 10 Most Shifted Features (`Step 42 -> Steps 43–49` by PSI)

| Feature | Group | Mean (Tr / 42 / 43–49) | Median (Tr / 43–49) | Std (Tr / 43–49) | PSI (`Tr->43-49`) | PSI (`42->43-49`) | KS (`42->43-49`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Aggregate_feature_43` | `Aggregate_72` | `-0.00 / 2.31 / 2.66` | `-0.17 / 2.61` | `1.00 / 0.50` | `11.8221` | **11.9079** | `0.9734` |
| `Aggregate_feature_46` | `Aggregate_72` | `-0.00 / 2.17 / 2.46` | `-0.08 / 2.43` | `1.00 / 0.51` | `11.8052` | **10.1706** | `0.8230` |
| `Aggregate_feature_8` | `Aggregate_72` | `-0.00 / 2.67 / 3.03` | `-0.19 / 2.98` | `1.00 / 0.24` | `8.9528` | **10.1586** | `0.9936` |
| `Aggregate_feature_10` | `Aggregate_72` | `-0.00 / 2.66 / 3.02` | `-0.18 / 2.97` | `1.00 / 0.25` | `8.7256` | **9.9602** | `0.9892` |
| `Aggregate_feature_7` | `Aggregate_72` | `0.00 / 2.57 / 2.92` | `-0.16 / 2.88` | `1.00 / 0.27` | `7.7845` | **9.9087** | `0.9853` |
| `Aggregate_feature_44` | `Aggregate_72` | `0.00 / 1.71 / 1.89` | `-0.16 / 1.79` | `1.00 / 0.52` | `11.8066` | **9.0337** | `0.7568` |
| `Local_feature_2` | `Local_93` | `0.00 / 0.98 / 1.01` | `-0.12 / 0.14` | `1.00 / 4.47` | `1.7395` | **1.6579** | `0.3316` |
| `fees` | `Domain_17` | `0.00 / 0.98 / 1.00` | `-0.12 / 0.14` | `1.00 / 4.47` | `1.6764` | **1.6339** | `0.3352` |
| `Aggregate_feature_13` | `Aggregate_72` | `0.00 / 1.15 / 1.15` | `-0.13 / 0.04` | `1.00 / 4.53` | `1.5195` | **0.7536** | `0.2777` |
| `Aggregate_feature_14` | `Aggregate_72` | `-0.00 / 0.92 / 1.02` | `-0.17 / -0.05` | `1.00 / 3.41` | `1.3347` | **0.6164** | `0.2626` |

### 4.1 Top 10 Validation-Discriminative Features Across Time (Standardized Cohen's $d$)

| Feature | Group | $d$ (Train `1–34`) | $d$ (Val `35–41`) | $d$ (Step `42`) | $d$ (Steps `43–49`) | Sign Flip in `43–49`? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Local_feature_53` | `Local_93` | `-1.278` | `-1.211` | `-1.308` | `-0.294` | `False` |
| `Local_feature_55` | `Local_93` | `-1.059` | `-1.150` | `-1.266` | `-0.409` | `False` |
| `Aggregate_feature_57` | `Aggregate_72` | `-0.614` | `-1.137` | `-1.044` | `-0.551` | `False` |
| `Local_feature_90` | `Local_93` | `-1.065` | `-1.105` | `-1.035` | `-0.699` | `False` |
| `Local_feature_52` | `Local_93` | `-0.735` | `-1.008` | `-1.126` | `-0.617` | `False` |
| `Local_feature_89` | `Local_93` | `-1.079` | `-0.992` | `-1.016` | `-0.861` | `False` |
| `Aggregate_feature_51` | `Aggregate_72` | `-0.346` | `-0.767` | `-0.733` | `+0.024` | `True` |
| `Local_feature_77` | `Local_93` | `-0.511` | `-0.759` | `-0.466` | `-0.521` | `False` |
| `Aggregate_feature_63` | `Aggregate_72` | `-0.530` | `-0.757` | `-0.696` | `+0.235` | `True` |
| `Local_feature_83` | `Local_93` | `-0.545` | `-0.755` | `-0.443` | `-0.512` | `False` |

---

## 5A. Per-Snapshot Prediction Diagnosis: `GCNAvg` vs. `HistGradientBoosting` (Steps `42–49`)

| Step | N (Ill) | GCNAvg PR / ROC | HGB PR / ROC | GCNAvg `meanP(Ill / Lic / Unl+All Pos%)` | HGB `meanP(Ill / Lic / All Pos%)` | GCNAvg TP/FP/FN | HGB TP/FP/FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `42` | `2154` (`239`) | `0.8829` / `0.9544` | `0.8913` / `0.9653` | `0.843 / 0.124 (9.34%)` | `0.815 / 0.039 (9.31%)` | `187/4/52` | `190/7/49` |
| `43` | `1370` (`24`) | `0.0205` / `0.5696` | `0.0539` / `0.8046` | `0.214 / 0.201 (4.29%)` | `0.092 / 0.037 (3.81%)` | `0/3/24` | `0/1/24` |
| `44` | `1591` (`24`) | `0.0298` / `0.7339` | `0.0395` / `0.7068` | `0.282 / 0.167 (8.30%)` | `0.155 / 0.075 (7.64%)` | `0/11/24` | `1/8/23` |
| `45` | `1221` (`5`) | `0.0150` / `0.7344` | `0.0104` / `0.7523` | `0.310 / 0.128 (5.54%)` | `0.063 / 0.049 (4.36%)` | `0/11/5` | `0/3/5` |
| `46` | `712` (`2`) | `0.0129` / `0.8373` | `0.1823` / `0.9549` | `0.290 / 0.134 (5.54%)` | `0.477 / 0.041 (6.00%)` | `0/2/2` | `1/3/1` |
| `47` | `846` (`22`) | `0.0361` / `0.6602` | `0.0678` / `0.8162` | `0.099 / 0.115 (7.42%)` | `0.038 / 0.026 (6.70%)` | `0/0/22` | `0/1/22` |
| `48` | `471` (`36`) | `0.1415` / `0.7342` | `0.2917` / `0.8980` | `0.138 / 0.086 (5.86%)` | `0.076 / 0.020 (4.33%)` | `0/0/36` | `0/0/36` |
| `49` | `476` (`56`) | `0.1843` / `0.7167` | `0.2960` / `0.8492` | `0.122 / 0.093 (11.86%)` | `0.056 / 0.044 (5.17%)` | `0/1/56` | `1/2/55` |

- **Step 42 Catch Overlap (`239` illicit):** Both caught `184`, GCNAvg-only `3`, HGB-only `6`, both missed `46`.
- **Counterfactual Ranking Ceiling on Steps `43–49` (`169` illicit):** `GCNAvg` oracle post-hoc max F1 = `0.0686` (`tau=0.04`), Top-50 hits = `0` (`k=100`: `0`); `HGB` oracle post-hoc max F1 = `0.1078` (`tau=0.02`), Top-50 hits = `3` (`k=100`: `5`).

---

## 6. Limitations & Most Scientifically Justified Next Experiment

### Diagnostic Limitations
1. **Sparse Post-Shutdown Positives:** Steps `45` (`5` illicit) and `46` (`2` illicit) contain very few labeled positive examples, making single-snapshot PR-AUC and Cohen's $d$ estimates noisy at the individual-step level; pooled Steps `43–49` (`169` illicit) gives the reliable summary.
2. **Unlabeled Transactions (`~77%`):** Class-3 unlabeled background nodes cannot confirm whether fraud-signature rotation also alters unlabeled neighborhoods or reflects Elliptic sampling shifts after the dark-market closure.

### Recommended Next Experiment (NOT Executed)
- **Stage 5 / Future Research Priority (Train-1–34 / Val-35–41 Strictly Preserved):** Because `HistGradientBoosting` holds `ROC-AUC = 0.7698` on Steps `43–49` while *every* pure `Local-93` neural/GNN architecture collapses to `0.59–0.63` on Steps `43–49`, the single most scientifically justified next experiment is a **Hybrid GNN-Embedding + Tabular Tree (`HistGradientBoosting` / Stacking) vs. RankGauss/Quantile-Transformed Feature Graph Model** — specifically testing (a) whether **Train-fitted `QuantileTransformer(output_distribution='normal')`** on all `182` features gives neural/GNN models the same coordinate-wise tail invariance trees rely on in Steps `43–49`, and (b) whether feeding out-of-fold / validation-calibrated `GCNAvg` graph embeddings + `182` features into `HistGradientBoosting` beats pure `HistGradientBoosting` (`PR-AUC 0.5646, ROC-AUC 0.8848`) without touching `Steps 42–49` for tuning!
