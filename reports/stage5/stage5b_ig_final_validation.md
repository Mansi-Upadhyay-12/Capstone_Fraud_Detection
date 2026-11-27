# Stage 5B Final Read-Only Validation: 3-Zone Composite Midpoint Integrated Gradients

**Validated On:** 2026-11-26  
**Quadrature Under Test:** 3-zone Composite Midpoint Grid (`60` uniform midpoints on `[0, 0.005]` + `60` on `[0.005, 0.05]` + `80` on `[0.05, 1.0]`; total $M = 200$ steps, baseline $X_{\text{base}} = 0$)  
**Baseline Method Compared:** Stage 5B (`poly4` change-of-variables midpoint rule, $M = 80$ steps)  
**Scope & Constraints:** 100% read-only validation across all `12` Stage 5B selected transactions (`8` Validation `Steps 35–41` + `4` Exploratory Test `Steps 42–49`). Frozen model weights, RankGauss preprocessing, threshold (`0.57`), predictions, Stage 5A outputs, and existing Stage 5B artifacts remain completely untouched.

---

## 1. Per-Transaction Completeness, Finiteness, and Reproducibility (All 12 Cases)

| Case ID | Split | Cat | `txId` | Step | $\Delta\text{logit}$ | Old Abs Err | New Abs Err | Abs Improvement | Old Rel Err | New Rel Err | Rel Improvement | Reduction | Finite | Repeated Max Diff |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `VAL_TP_1` | Val | `TP` | `70361269` | `35` | `+10.759424` | `0.116847` | `0.011545` | `+0.105302` | `1.0860%` | `0.1073%` | `+0.9787 pp` | `10.12×` | `True` | `0.00e+00` |
| `VAL_TP_2` | Val | `TP` | `40467500` | `38` | `+4.676081` | `0.055014` | `0.033648` | `+0.021366` | `1.1765%` | `0.7196%` | `+0.4569 pp` | `1.64×` | `True` | `0.00e+00` |
| `VAL_FP_1` | Val | `FP` | `54915900` | `35` | `+6.789657` | `0.087920` | `0.025492` | `+0.062428` | `1.2949%` | `0.3754%` | `+0.9195 pp` | `3.45×` | `True` | `0.00e+00` |
| `VAL_FP_2` | Val | `FP` | `97404552` | `40` | `+4.748868` | `0.000549` | `0.001564` | `-0.001015` | `0.0116%` | `0.0329%` | `-0.0214 pp` | `0.35×` | `True` | `0.00e+00` |
| `VAL_FN_1` | Val | `FN` | `97729143` | `40` | `-1.506028` | `0.040778` | `0.000311` | `+0.040467` | `2.7077%` | `0.0207%` | `+2.6870 pp` | `131.11×` | `True` | `0.00e+00` |
| `VAL_FN_2` | Val | `FN` | `116055520` | `38` | `+4.358511` | `0.026568` | `0.039332` | `-0.012764` | `0.6096%` | `0.9024%` | `-0.2929 pp` | `0.68×` | `True` | `0.00e+00` |
| `VAL_TN_1` | Val | `TN` | `97506379` | `40` | `-1.808382` | `0.004788` | `0.013104` | `-0.008316` | `0.2648%` | `0.7246%` | `-0.4599 pp` | `0.37×` | `True` | `0.00e+00` |
| `VAL_TN_2` | Val | `TN` | `201433814` | `36` | `+4.200376` | `0.041514` | `0.033490` | `+0.008024` | `0.9883%` | `0.7973%` | `+0.1910 pp` | `1.24×` | `True` | `0.00e+00` |
| `TEST_TP_42` | Test* | `TP` | `71989375` | `42` | `+10.625143` | `0.007505` | `0.002238` | `+0.005267` | `0.0706%` | `0.0211%` | `+0.0496 pp` | `3.35×` | `True` | `0.00e+00` |
| `TEST_FP_1` | Test* | `FP` | `106364923` | `44` | `+9.341211` | `0.130915` | `0.028267` | `+0.102648` | `1.4015%` | `0.3026%` | `+1.0989 pp` | `4.63×` | `True` | `0.00e+00` |
| `TEST_FN_POST43` | Test* | `FN` | `113621810` | `44` | `-1.463031` | `0.002624` | `0.001466` | `+0.001158` | `0.1794%` | `0.1002%` | `+0.0791 pp` | `1.79×` | `True` | `0.00e+00` |
| `TEST_TN_1` | Test* | `TN` | `94333210` | `43` | `-1.821304` | `0.007047` | `0.011839` | `-0.004792` | `0.3869%` | `0.6500%` | `-0.2631 pp` | `0.60×` | `True` | `0.00e+00` |

*\*Exploratory Test transactions (`Steps 42–49`); no test labels used for tuning.*

---

## 2. Aggregate Statistical Summary

- **Maximum Absolute Error:** `0.130915` (`TEST_FP_1`) $\rightarrow$ **`0.039332`** (`VAL_FN_2`), a **`3.33×`** worst-case reduction. All `12/12` transactions satisfy `< 0.05` logits (actually `< 0.04` logits).
- **Maximum Relative Error:** `2.7077%` (`VAL_FN_1`) $\rightarrow$ **`0.9024%`** (`VAL_FN_2`), a **`3.00×`** worst-case reduction. All `12/12` transactions satisfy `< 1.0%` relative error (`7/12` are $\le 0.40\%$, vs. `5/12` exceeding `1.0%` previously).
- **Mean / Median Absolute Error:**
  - Old (`poly4 M=80`): mean `0.043506`, median `0.033673`
  - New (`Composite M=200`): mean **`0.016858`** (**`2.58×`** lower), median **`0.012471`**
- **Mean / Median Relative Error:**
  - Old (`poly4 M=80`): mean `0.8481%`, median `0.7990%`
  - New (`Composite M=200`): mean **`0.3962%`** (**`2.14×`** lower), median **`0.3390%`**
- **Behavior on Small-Error Cases:** On `8/12` transactions, error decreased by `1.24×–131.11×`. On `4/12` transactions that already had very small errors under `poly4 M=80` (`VAL_FP_2`, `VAL_FN_2`, `VAL_TN_1`, `TEST_TN_1`), ReLU-kink alignment shifted slightly (`-0.0010` to `-0.0128` abs difference), with every single case remaining well within `< 0.04` logits and `< 1.0%` relative error.
- **Numerical Finiteness & Reproducibility:** All `12/12` repeated `[N_t, 182]` gradient runs produced `100%` finite values (`all_finite = True`) and exact bitwise reproducibility (`max_repro_diff = 0.00e+00` across all `12` cases).

---

## 3. Immutability & Protocol Verification

- **Frozen Stage 4H Checkpoint & Preprocessing:** Verified via SHA-256 digests before and after validation (`ExpA2_RankGauss182_BiDirGCN_seed42.pt`, `rankgauss_transformer_train_1_34.joblib` — untouched).
- **Frozen Decision Threshold:** Confirmed `threshold_recal_01_99 = 0.57` unchanged.
- **Predictions & Confusion Matrices Unchanged:**
  - **Validation (`Steps 35–41`, `7,829` labeled; `22,851` Unknown `-1` excluded):** `TP=573`, `FP=33`, `FN=102`, `TN=7,121` (exact match).
  - **Test (`Steps 42–49`, `8,841` labeled; `27,983` Unknown `-1` excluded):** `TP=192`, `FP=43`, `FN=216`, `TN=8,390` (exact match).
  - All `12` target transaction probabilities matched [`stage5b_individual_explanations.csv`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/stage5/stage5b_individual_explanations.csv) to `< 1e-5`.
- **Stage 5A & Existing Stage 5B Outputs Unchanged:** Verified identical SHA-256 hashes before and after execution across all Stage 5A CSVs and Stage 5B CSV/JSON/MD deliverables.

---

## 4. Final Recommendation

**Explicit Verdict: `PASS`**

The 3-zone Composite Midpoint Grid (`60@[0, 0.005] + 60@[0.005, 0.05] + 80@[0.05, 1.0]`, total $M = 200$) is **approved for subsequent Stage 5 Integrated Gradients explanations**. Across all `12` selected transactions, it cuts worst-case absolute completeness error from `0.130915` to **`0.039332`** (`3.33×` improvement, every case `< 0.04` logits), cuts worst-case relative error from `2.7077%` to **`0.9024%`** (`3.00×` improvement, every case `< 1.0%`), maintains exact `0.00e+00` reproducibility and `100%` numerical finiteness, and preserves all frozen Stage 4H weights, preprocessing, threshold (`0.57`), predictions, and prior Stage 5A/5B outputs.
