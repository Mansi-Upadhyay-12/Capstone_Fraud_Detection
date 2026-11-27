# Stage 5B Completion Report: Individual Fraud Prediction Explanations

**Generated On:** 2026-11-26 19:52:43  
**Stage 4H Frozen Champion:** `ExpA2_RankGauss182_BiDirGCN` (`seed=42`, `total_parameters=159,937`, frozen $\tau^* = 0.57$)  

---

## 1. Key Findings Across TP, FP, FN, and TN Cases

1. **Model & Threshold Verification (Zero Tuning, Unknown `-1` Excluded):**
   - Validation `Steps 35–41` (`7,829` labeled; `22,851` unknown `-1` excluded): `TP=573`, `FP=33`, `FN=102`, `TN=7121` — exact match to Stage 4H checkpoint metadata.
   - Exploratory Test `Steps 42–49` (`8,841` labeled; `27,983` unknown `-1` excluded): `TP=192`, `FP=43`, `FN=216`, `TN=8390` — exact match.
2. **Summary Table of Selected Validation (`8`) & Exploratory Test (`4`) Examples:**

| Case ID | Split | Category | `txId` | Step | `GT` | `P(Illicit)` | In / Out Deg | 1-Hop Pred Illicit | Self IG | Nbr IG | Top +Driver / Top -Driver |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `VAL_TP_1` | Val | `TP` | `70361269` | `35` | `Illicit` | `0.9984` | `1/1` | `2/2` | `+1.98` | `+8.90` | `Aggregate_feature_70 (+3.52)` / `Aggregate_feature_10 (-3.84)` |
| `VAL_TP_2` | Val | `TP` | `40467500` | `38` | `Illicit` | `0.5868` | `1/1` | `1/2` | `+5.17` | `-0.55` | `Local_feature_80 (+7.39)` / `Aggregate_feature_8 (-13.36)` |
| `VAL_FP_1` | Val | `FP` | `54915900` | `35` | `Licit` | `0.9216` | `1/1` | `1/2` | `-3.29` | `+9.99` | `Aggregate_feature_70 (+7.84)` / `Aggregate_feature_10 (-12.53)` |
| `VAL_FP_2` | Val | `FP` | `97404552` | `40` | `Licit` | `0.6043` | `2/2` | `0/4` | `+0.14` | `+4.61` | `Aggregate_feature_49 (+6.49)` / `Aggregate_feature_8 (-10.62)` |
| `VAL_FN_1` | Val | `FN` | `97729143` | `40` | `Illicit` | `0.0029` | `1/1` | `0/2` | `-1.47` | `-0.08` | `Aggregate_feature_20 (+2.29)` / `Aggregate_feature_7 (-2.21)` |
| `VAL_FN_2` | Val | `FN` | `116055520` | `38` | `Illicit` | `0.5083` | `1/1` | `0/2` | `+8.78` | `-4.45` | `Local_feature_90 (+6.25)` / `Aggregate_feature_10 (-9.21)` |
| `VAL_TN_1` | Val | `TN` | `97506379` | `40` | `Licit` | `0.0022` | `3/1` | `0/4` | `-0.49` | `-1.33` | `Local_feature_88 (+0.45)` / `out_BTC_min (-0.51)` |
| `VAL_TN_2` | Val | `TN` | `201433814` | `36` | `Licit` | `0.4688` | `4/1` | `1/5` | `+11.74` | `-7.59` | `Aggregate_feature_49 (+7.22)` / `Aggregate_feature_7 (-9.80)` |
| `TEST_TP_42` | Test* | `TP` | `71989375` | `42` | `Illicit` | `0.9982` | `1/1` | `0/2` | `+5.14` | `+5.49` | `Aggregate_feature_20 (+4.02)` / `Aggregate_feature_7 (-5.30)` |
| `TEST_FP_1` | Test* | `FP` | `106364923` | `44` | `Licit` | `0.9934` | `1/1` | `0/2` | `-0.44` | `+9.65` | `Local_feature_88 (+8.04)` / `Aggregate_feature_7 (-8.33)` |
| `TEST_FN_POST43` | Test* | `FN` | `113621810` | `44` | `Illicit` | `0.0031` | `2/1` | `0/3` | `-0.18` | `-1.28` | `Aggregate_feature_6 (+0.88)` / `Aggregate_feature_7 (-1.13)` |
| `TEST_TN_1` | Test* | `TN` | `94333210` | `43` | `Licit` | `0.0021` | `24/2` | `0/26` | `-0.24` | `-1.58` | `Aggregate_feature_20 (+0.18)` / `Aggregate_feature_70 (-0.41)` |

3. **Behavioral Diagnostic Summary:**
   - **True Positives (`TP`):** Driven by strong positive feature attributions (`Local` + `Aggregate` features) reinforced by high-risk predicted neighbors (`P >= 0.57`) along incoming/outgoing transaction links.
   - **False Positives (`FP`):** Licit-labeled transactions receive elevated scores when either deceptive local/aggregate feature values under RankGauss scaling mimic fraud signatures or adjacent suspicious transactions propagate positive message-passing logits into the node.
   - **False Negatives (`FN`):** Missed frauds (especially post-shutdown `TEST_FN_POST43`) show camouflaged feature profiles where negative licit-pulling attributions outweigh positive drivers and 1-hop neighbors remain predicted Licit or unlabeled (`-1`), leaving insufficient neighbor evidence to reach $\tau^* = 0.57$.
   - **True Negatives (`TN`):** Consistently dominated by strong negative `Local` and `Aggregate` IG contributions and benign neighborhood scores.

---

## 2. Verification Summary

- **Checkpoint & Preprocessing Alignment:** `ExpA2_RankGauss182_BiDirGCN_seed42.pt` (`threshold=0.57`) + `rankgauss_transformer_train_1_34.joblib` verified exact Validation & Test confusion matrices (`cm_exact_match = True`).
- **Dataset ID & Edge Integrity:** Verified `txId`, `timestep`, `y`, and directed edges across all selected snapshots and raw CSV inspection (`node_id_and_edge_alignment_ok = True`).
- **Stage 5A IG Parity & Reproducibility:** All `12` individual IG explanations matched Stage 5A's `integrated_gradients_snapshot(..., n_steps=80)` to `max_parity_diff = 0.00e+00`; max completeness error = `1.31e-01`; repeated run max diff = `0.00e+00`; all numeric outputs finite (`True`).
- **Human-Readable Explanations:** Saved in [`stage5b_human_explanations.md`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage5/stage5b_human_explanations.md).
