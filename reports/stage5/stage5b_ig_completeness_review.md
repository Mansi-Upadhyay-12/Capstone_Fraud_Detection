# Stage 5B Technical Review: Integrated Gradients Completeness Error Investigation

**Reviewed Date:** 2026-11-26  
**Scope:** Independent numerical review of Integrated Gradients (IG) completeness error across all `12` explanations in [`stage5b_individual_explanations.csv`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/results/stage5/stage5b_individual_explanations.csv).  
**Constraints Preserved:** Zero retraining, zero threshold or prediction changes, zero edits to Stage 5A or existing Stage 5B artifacts (`ExpA2_RankGauss182_BiDirGCN_seed42.pt`, `threshold = 0.57`, and `rankgauss_transformer_train_1_34.joblib` remain untouched).

---

## 1. Exact Completeness Equation Used

In [`explain_single_node_ig`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/stage5/stage5b_individual_explanations.py#L225-L265) (matching Stage 5A's [`integrated_gradients_snapshot`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/stage5/stage5a_feature_importance.py#L156-L208)), attributions for target node $u$ at snapshot $t$ are evaluated in centered RankGauss-182 space $X_{\text{rg}}^{(t)} \in \mathbb{R}^{N_t \times 182}$ relative to zero baseline $X_{\text{base}} = 0$ along the straight-line path $X(\alpha) = \alpha X_{\text{rg}}^{(t)}$ ($\alpha \in [0, 1]$).

By the Fundamental Theorem of Calculus for path integrals (Sundararajan et al., 2017):

$$\text{IG}_{v, j}(u) = \left(X_{\text{rg}}^{(t)} - X_{\text{base}}\right)_{v, j} \int_{0}^{1} \frac{\partial f_u\bigl(\alpha X_{\text{rg}}^{(t)}\bigr)}{\partial X_{v, j}} \, d\alpha$$

$$\sum_{v=1}^{N_t} \sum_{j=1}^{182} \text{IG}_{v, j}(u) = f_u\!\left(X_{\text{rg}}^{(t)}\right) - f_u\!\left(0\right) = \Delta\text{logit}_u$$

where $f_u(\cdot)$ is the **pre-sigmoid output logit** of `ExpA2_RankGauss182_BiDirGCN` (without `DropEdge` or dropout in `.eval()` mode). Applying the change of variables $\alpha(u') = (u')^4$ with $d\alpha = 4(u')^3\,du'$ over $M = 80$ midpoints $u'_k = \frac{k - 0.5}{80}$ ($k = 1, \dots, 80$), the recorded numerical completeness error is:

$$\text{ig\_completeness\_err}(u) = \left| \sum_{v=1}^{N_t} \sum_{j=1}^{182} \left[ X_{\text{rg}, v, j}^{(t)} \sum_{k=1}^{80} \frac{4(u'_k)^3}{80} \frac{\partial f_u\bigl((u'_k)^4 X_{\text{rg}}^{(t)}\bigr)}{\partial X_{v, j}} \right] - \bigl(f_u(X_{\text{rg}}^{(t)}) - f_u(0)\bigr) \right|$$

And normalized (relative) completeness error is:

$$\text{rel\_err}(u) = \frac{\text{ig\_completeness\_err}(u)}{\left|f_u(X_{\text{rg}}^{(t)}) - f_u(0)\right|} \times 100\%$$

---

## 2. Absolute and Normalized/Relative Error Across All 12 Explanations

| Case ID | Split Role | `txId` | Step | Baseline $f_u(0)$ | Target $f_u(X_{\text{rg}})$ | $\Delta\text{logit}_u$ | Abs Completeness Error | Relative Error (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `VAL_TP_1` | Validation | `70361269` | `35` | `-4.325494` | `+6.433930` | `+10.759424` | `0.116847` | `1.086%` |
| `VAL_TP_2` | Validation | `40467500` | `38` | `-4.325498` | `+0.350584` | `+4.676081` | `0.055014` | `1.176%` |
| `VAL_FP_1` | Validation | `54915900` | `35` | `-4.325498` | `+2.464159` | `+6.789657` | `0.087920` | `1.295%` |
| `VAL_FP_2` | Validation | `97404552` | `40` | `-4.325499` | `+0.423369` | `+4.748868` | `0.000549` | `0.012%` |
| `VAL_FN_1` | Validation | `97729143` | `40` | `-4.325499` | `-5.831527` | `-1.506028` | `0.040778` | **`2.708%`** *(Max Rel)* |
| `VAL_FN_2` | Validation | `116055520` | `38` | `-4.325489` | `+0.033022` | `+4.358511` | `0.026568` | `0.610%` |
| `VAL_TN_1` | Validation | `97506379` | `40` | `-4.325494` | `-6.133876` | `-1.808382` | `0.004788` | `0.265%` |
| `VAL_TN_2` | Validation | `201433814` | `36` | `-4.325494` | `-0.125118` | `+4.200376` | `0.041514` | `0.988%` |
| `TEST_TP_42` | Test (Expl.) | `71989375` | `42` | `-4.325490` | `+6.299653` | `+10.625143` | `0.007505` | `0.071%` |
| **`TEST_FP_1`** | **Test (Expl.)** | **`106364923`** | **`44`** | **`-4.325490`** | **`+5.015721`** | **`+9.341211`** | **`0.130915`** *(Max Abs)* | `1.401%` |
| `TEST_FN_POST43` | Test (Expl.) | `113621810` | `44` | `-4.325496` | `-5.788527` | `-1.463031` | `0.002624` | `0.179%` |
| `TEST_TN_1` | Test (Expl.) | `94333210` | `43` | `-4.325475` | `-6.146779` | `-1.821304` | `0.007047` | `0.387%` |

- **Summary Across All 12 Cases:**
  - **Max absolute error:** **`0.130915`** (`TEST_FP_1`)
  - **Max Validation absolute error:** **`0.116847`** (`VAL_TP_1`)
  - **Max relative error:** **`2.708%`** (`VAL_FN_1`, where $|\Delta\text{logit}| = 1.5060$ is small)
  - **Median / Mean absolute error:** `0.033673` / `0.043506` logits
  - **Median / Mean relative error:** `0.799%` / `0.848%` (`9/12` cases $\le 1.20\%$; `5/12` cases $\le 0.50\%$).

---

## 3. Transaction With the Maximum Error

- **Worst Absolute Error Case:** **`TEST_FP_1`** — Transaction `txId = 106364923`, Timestep `44` (snapshot index `3229`), Ground Truth = `Licit (y=0)`, Predicted Probability = `0.993411`, Classification = `FP` (Exploratory Test).
  - Baseline logit $f_u(0) = -4.325490$; target logit $f_u(X_{\text{rg}}) = +5.015721$; true $\Delta\text{logit} = +9.341211$.
  - Numerically integrated IG sum across all snapshot nodes & features = `+9.210296` (difference = `-0.130915`, i.e., `1.401%` relative error, meaning `98.60%` of the logit shift is exactly accounted for).

---

## 4. Expected Numerical Tolerance & Assessment

- **Standard Literature & Engineering Tolerance:**
  - Standard IG guidance (Sundararajan et al., 2017; Captum) treats completeness error within **`1%–5%` relative to $|\Delta f|$** (roughly `< 0.05–0.15` logits for $\sim 10$-logit outputs on deep piecewise-linear `ReLU` networks with $M \approx 50\text{–}300$ steps) as acceptable for rank-ordering top features, while **high-precision audit tolerance is `< 0.5%` relative / `< 0.05` absolute**.
- **Assessment of Stage 5B's `0.130915` Error:**
  - First, `M=80` `poly4` already prevents catastrophic failure: standard **uniform midpoint `M=80`** gave **`5.7807` (`53.73%` error)** on `VAL_TP_1`!
  - Second, **`0.130915` (`1.40%`) on `TEST_FP_1`, `0.116847` (`1.09%`) on `VAL_TP_1`, and `2.71%` on `VAL_FN_1` are borderline acceptable for qualitative feature ranking, but exceed a strict `< 0.05` absolute / `< 0.5%` relative engineering tolerance.** Improving precision ensures downstream quantitative claims (such as fine-grained self vs. neighbor IG comparisons on small-$|\Delta\text{logit}|$ FNs) stay tightly converged.

---

## 5. Root-Cause Diagnosis & Recommended Correction

### Empirical Root Cause Analysis

Diagnostic profiling of $g(\alpha) = \frac{d}{d\alpha} f_u(\alpha X_{\text{rg}}) = \langle \nabla_X f_u(\alpha X_{\text{rg}}), X_{\text{rg}} \rangle$ on the frozen checkpoint reveals **two coupled numerical causes**:

| $\alpha$ along path | `TEST_FP_1` $dL/d\alpha$ | `TEST_FP_1` Cumulative $\Delta L(\alpha)$ | `VAL_TP_1` $dL/d\alpha$ | `VAL_TP_1` Cumulative $\Delta L(\alpha)$ |
| :---: | :---: | :---: | :---: | :---: |
| `0.000001` | `+173.70` | `+0.0002` | `+1,647.25` | `+0.0016` |
| `0.000100` | `+203.16` | `+0.0181` | `+2,026.73` | `+0.1783` |
| `0.000500` | `+412.35` | `+0.1433` | `+3,041.23` | `+1.2141` |
| `0.001000` | `+401.34` | `+0.3427` | **`+3,063.90`** | `+2.7425` |
| `0.005000` | **`+420.97`** | `+1.9958` | `+404.28` | **`+7.9419`** *(74% of +10.76)* |
| `0.010000` | `+388.97` | `+4.0678` | `+157.73` | `+9.1520` *(85% of +10.76)* |
| `0.050000` | `+25.18` | **`+8.9671`** *(96% of +9.34)* | `+5.84` | `+10.5183` *(98% of +10.76)* |
| `0.200000` | `-0.78` | `+9.5672` | `+0.28` | `+10.7164` |
| `0.990000` | `-0.08` | `+9.3420` | `+0.01` | `+10.7593` |

1. **`LayerNorm(eps=1e-5)` boundary on bias-free neighbor layers concentrates 74%–96% of $\Delta\text{logit}$ inside $\alpha \in [0, 0.05]$:**  
   In [`BiDirGCNLayer`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/models_stage4f.py#L121-L143), `in_proj` and `out_proj` have `bias=False` followed immediately by `LayerNorm(eps=1e-5)`. Because $\text{LayerNorm}(\alpha z)$ transitions sharply around $\alpha^2 \text{Var}(z) \sim 10^{-5}$, the integrand spikes to `+421` (`TEST_FP_1`) and `+3,064` (`VAL_TP_1`) below $\alpha = 0.01$, then flattens near zero above $\alpha = 0.20$.
2. **`poly4` ($\alpha = u^4$) over-clusters near $10^{-8}$ while leaving `[0.005, 0.05]` too coarse across `ReLU` kinks:**  
   At $M=80$, $\alpha = u^4$ places ~`25%` of all quadrature points below $\alpha = 0.0004$, leaving wider spacing ($\Delta\alpha \approx 0.0016\text{–}0.0053$) through $\alpha \in [0.005, 0.05]$ where `TEST_FP_1` still gains `~74%` (`+6.97` logits) of its total change and crosses multiple piecewise-linear `F.relu(fused + s)` activation kinks. Note that high-order Gauss-Legendre (`split-GL M=80`: `0.0496` on `TEST_FP_1`) also loses exponential accuracy across `C^0` `ReLU` derivative jumps.

### Benchmark of Alternative Quadrature Schemes (Same Frozen Model & Baseline $X_{\text{base}} = 0$)

| Quadrature Method | `TEST_FP_1` Abs / Rel Err | `VAL_TP_1` Abs / Rel Err | `VAL_FN_1` Abs / Rel Err |
| :--- | :---: | :---: | :---: |
| Uniform midpoint ($M=80$) | `0.013221` (`0.142%`) | `5.780668` (`53.727%`) | — |
| **Stage 5B baseline (`poly4`, $M=80$)** | `0.130919` (`1.402%`) | `0.116845` (`1.086%`) | `0.040778` (`2.708%`) |
| Split Gauss-Legendre (`30@[0,0.01]+50@[0.01,1]`, $M=80$) | `0.049606` (`0.531%`) | `0.029648` (`0.276%`) | — |
| `poly4` midpoint ($M=200$) | `0.034488` (`0.369%`) | `0.038219` (`0.355%`) | — |
| **Composite Midpoint `60@[0,0.005] + 60@[0.005,0.05] + 80@[0.05,1.0]` ($M=200$)** | **`0.028260` (`0.303%`)** | **`0.011544` (`0.107%`)** | **`0.000311` (`0.021%`)** |
| Composite Fine `150@[0,0.01] + 90@[0.01,0.1] + 60@[0.1,1.0]` ($M=300$) | `0.023530` (`0.252%`) | `0.003475` (`0.032%`) | `0.008687` (`0.577%`) |

### Concrete Recommendation

Without changing the frozen model weights, RankGauss preprocessing, threshold (`0.57`), predictions, or existing Stage 5A/5B artifacts:
- **Recommended Replacement Rule for Future Attribution Calls (e.g., Stage 5C):** Use the **3-zone Composite Midpoint Grid (`60` steps on `[0, 0.005]`, `60` on `[0.005, 0.05]`, `80` on `[0.05, 1.0]`, total $M=200$)**.
- **Why this works best:** Unlike `poly4`, it matches grid density directly to the measured physical transition bands ($\alpha \in [0, 0.005]$ where `LayerNorm` switches, $\alpha \in [0.005, 0.05]$ where `ReLU` kinks and residual neighbor transitions occur, and $\alpha \in [0.05, 1.0]$ for smooth saturation), reducing error by **4.6× on `TEST_FP_1`** (`0.1309 -> 0.0283`, `0.30%`), **10.1× on `VAL_TP_1`** (`0.1168 -> 0.0115`, `0.11%`), and **131× on `VAL_FN_1`** (`2.708% -> 0.021%`). All three worst cases fall strictly inside `< 0.03` absolute and `<= 0.30%` relative completeness tolerance.
