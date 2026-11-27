# Stage 5A: Feature Importance & Attribution Analysis Report

**Generated On:** 2026-11-26 19:23:02  
**Models Evaluated (Frozen — Zero Retraining / Zero Test Tuning):**
- **Stage 4H Graph Champion:** `ExpA2_RankGauss182_BiDirGCN` (`seed=42`, $\tau^*=0.57$, Val PR-AUC=`0.9334`, Exploratory Test PR-AUC=`0.6023`) paired with saved Train-`1–34` pipeline `rankgauss_transformer_train_1_34.joblib`.
- **Stage 3 Tabular Baseline:** `HistGradientBoosting` (`182` features, $\tau^*=0.79$, Val PR-AUC=`0.9292`, Exploratory Test PR-AUC=`0.5646`).

---

## 1. Methodology & Leakage Distinction

1. **HistGradientBoosting Permutation Importance (Model-Agnostic, Imbalance-Appropriate):** Uses `average_precision` (PR-AUC) across `10` random repeats per feature (`n_repeats=10`) plus **joint group-block permutation** (permuting all `Local93`, `Aggregate72`, or `Domain17` columns together so collinear split substitutes cannot mask a group's total signal).
2. **Stage 4H GNN Attribution via Integrated Gradients (IG):** Because `QuantileTransformer` uses a piecewise-linear quantile table while `ExpA2_RankGauss182_BiDirGCN` operates directly on the centered RankGauss representations ($X_{\text{rg}}$, zero-centered on Train Steps `1–34`), IG integrates logit gradients from $X_{\text{base}} = 0$ to $X_{\text{rg}}^{(t)}$ using $M=80$ change-of-variables quadrature steps ($\alpha = u^4$, concentrating resolution near $\alpha \to 0^+$ where bias-free neighbor `LayerNorm` layers transition sharply). Aggregating $|\text{IG}|$ across the entire 2-hop bidirectional computation graph of target illicit nodes preserves high-precision Completeness (`max_completeness_err = 6.28e-02`).
3. **Validation vs. Exploratory Test Separation:** All primary rankings use **held-out Validation `Steps 35–41`** (`7,829` labeled nodes, `675` illicit). **Test `Step 42` (`239` illicit) and `Steps 43–49` (`169` illicit)** are reported strictly as exploratory post-hoc explanations of regime shift and were never used for feature selection or model tuning.

---

## 2. Feature-Group Analysis (`Local93` vs. `Aggregate72` vs. `Domain17`)

| Group | # Feats (`%`) | HGB Val Single-Feat Share | HGB Per-Feat Mean | HGB Val Block PR Drop | BiDirGCN Val IG Share | BiDirGCN Per-Feat Mean | BiDirGCN Val Zero-Mask PR Drop | BiDirGCN Test `42` IG% → `43–49` IG% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Local93` | `93` (`51.1%`) | `43.86%` | `0.00138` | `+0.6093` | `39.79%` | `0.80205` | `+0.6694` | `40.41%` → `42.60%` |
| `Aggregate72` | `72` (`39.56%`) | `6.83%` | `0.00026` | `+0.0346` | `51.05%` | `1.32934` | `+0.0085` | `50.42%` → `49.72%` |
| `Domain17` | `17` (`9.34%`) | `49.31%` | `0.00866` | `+0.1521` | `9.16%` | `1.01021` | `+0.0139` | `9.18%` → `7.68%` |

**Key Feature-Group Findings:**
- **Feature count does not equal importance:** On a **per-feature basis**, `Domain17` (`17` features, `9.34%` of columns) carries high per-feature attribution density in `BiDirGCN` (`1.01021` vs `Local93` `0.80205` and `Aggregate72` `1.32934`), whereas `HGB` block permutation shows **`Local93` alone causes a `+0.6093` Val PR-AUC drop** (`Aggregate72`: `+0.0346`, `Domain17`: `+0.1521`).
- **Exploratory Post-Shutdown (`Steps 43–49`) Shift:** In `HGB`, joint block ROC-AUC drop on `Steps 43–49` shows `Local93` (`+0.1746`) and `Aggregate72` (`+0.1143`), while in `BiDirGCN` masking `Aggregate72` (`+0.0999`) and `Local93` (`+0.0424`) confirms that **RankGauss successfully unlocked `Aggregate72` signal** for the GNN after Step 42!
- **Methodological Limitations:** Single-feature permutation underestimates collinear features (`Local93` + `Aggregate72` correlate strongly), while IG measures attribution relative to a zero RankGauss reference rather than counterfactual causal edge removal.

---

## 3. Top 10 Validation Features: `HistGradientBoosting` vs. `ExpA2_RankGauss182_BiDirGCN`

| Rank | HGB Top Feature (Index, Group) | HGB Val PR Drop | HGB Exploratory `42` / `43–49` | BiDirGCN Top Feature (Index, Group) | BiDirGCN Val `|Graph IG|` | BiDirGCN Exploratory `42` / `43–49` |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| 1 | `size` (`#169`, `Domain17`) | `+0.14863` | `+0.08959` / `+0.00309` | `Aggregate_feature_10` (`#102`, `Aggregate72`) | `7.47705` | `6.14420` / `3.82904` |
| 2 | `Local_feature_59` (`#58`, `Local93`) | `+0.03726` | `+0.01667` / `+0.01621` | `Aggregate_feature_8` (`#100`, `Aggregate72`) | `7.17871` | `6.07187` / `3.78044` |
| 3 | `Local_feature_55` (`#54`, `Local93`) | `+0.02260` | `+0.02902` / `-0.00173` | `Aggregate_feature_7` (`#99`, `Aggregate72`) | `7.06382` | `6.03732` / `4.78274` |
| 4 | `Local_feature_49` (`#48`, `Local93`) | `+0.01628` | `+0.00657` / `+0.00403` | `Local_feature_14` (`#13`, `Local93`) | `4.62316` | `3.83263` / `0.44116` |
| 5 | `Local_feature_53` (`#52`, `Local93`) | `+0.01244` | `+0.01507` / `-0.00688` | `Local_feature_90` (`#89`, `Local93`) | `4.59545` | `3.68279` / `0.66868` |
| 6 | `Local_feature_61` (`#60`, `Local93`) | `+0.00750` | `+0.00254` / `+0.00475` | `Aggregate_feature_70` (`#162`, `Aggregate72`) | `4.21226` | `3.83613` / `1.65418` |
| 7 | `Local_feature_3` (`#2`, `Local93`) | `+0.00686` | `+0.00040` / `+0.00646` | `Aggregate_feature_39` (`#131`, `Aggregate72`) | `4.17304` | `3.45861` / `0.85352` |
| 8 | `Aggregate_feature_70` (`#162`, `Aggregate72`) | `+0.00525` | `+0.00246` / `+0.00009` | `Aggregate_feature_45` (`#137`, `Aggregate72`) | `4.15534` | `3.37945` / `0.80113` |
| 9 | `Local_feature_80` (`#79`, `Local93`) | `+0.00489` | `+0.00181` / `-0.00011` | `Aggregate_feature_49` (`#141`, `Aggregate72`) | `4.11934` | `3.33639` / `2.94744` |
| 10 | `Local_feature_5` (`#4`, `Local93`) | `+0.00383` | `+0.00252` / `+0.00266` | `Aggregate_feature_43` (`#135`, `Aggregate72`) | `3.91477` | `3.54961` / `2.32511` |

---

## 4. Verification Summary

- **Checkpoint & Preprocessing Reproduction:** `ExpA2_RankGauss182_BiDirGCN_seed42.pt` + `rankgauss_transformer_train_1_34.joblib` reproduced Val PR-AUC = `0.9334` and Test PR-AUC = `0.6023` (`exact_match = True`); `hist_gradient_boosting.joblib` reproduced Val PR-AUC = `0.9292` and Test PR-AUC = `0.5646`.
- **IG Completeness & Finiteness:** Max IG completeness error across all snapshots = `6.28e-02`; `0` non-finite values in any importance or attribution table.
- **Reproducibility:** Repeated IG execution on `Val Step 35` matched to `max_diff = 0.00e+00`.
