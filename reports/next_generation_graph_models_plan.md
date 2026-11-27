# Next-Generation Graph Fraud Detection Models: Controlled Experimental Plan

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Plan Created:** 2026-11-26  
**Benchmark Anchor (Current Best Graph Model):** `EvolveGCN_H_Directed_Res` (Stage 4C)  
- **Validation PR-AUC (`35–41`):** `0.6387` (`Val F1 = 0.4810` at $\tau^*=0.90$; `Val F1 = 0.5947` at recalibrated $\tau^*=0.97$)  
- **Test PR-AUC (`42–49`):** `0.3107` | **Test ROC-AUC:** `0.7963`  
- **Test (`tau*=0.97` recalibrated):** `Prec = 14.11%`, `Rec = 55.64%`, `F1 = 0.2251`, `TP = 227`, `FP = 1,382`, `TN = 7,051`, `FN = 181`  
- **Primary Tabular Benchmark:** `HistGradientBoosting` (`Test PR-AUC = 0.5646`, `Test F1 = 0.6166`, `Prec = 88.53%`, `Rec = 47.30%`, `FP = 25`)  

---

## 1. Non-Negotiable Experimental Governance Rules

1. **Zero Modification of Existing Assets:** Existing scripts, checkpoints (`models/*.pt`, `models/*.joblib`), metrics JSONs, and reports remain strictly untouched.
2. **Isolated New Modules:** All new architectures and experiments will live in new files (e.g., `src/models_nextgen.py`, `src/train_nextgen.py`, `train_nextgen.py`) with `models/`, `reports/`, and `results/` outputs under distinct `stage4d_*` filenames.
3. **Strict Chronological Isolation:**
   - **Training (`Steps 1–34`):** `29,894` labeled nodes (`3,462` illicit / `26,432` licit), `136,265` total nodes.
   - **Validation (`Steps 35–41`):** `7,829` labeled nodes (`675` illicit / `7,154` licit), `30,680` total nodes — **exclusively** used for early stopping, hyperparameter tuning, loss/architecture selection, and threshold selection.
   - **Test (`Steps 42–49`):** `8,841` labeled nodes (`408` illicit / `8,433` licit), `36,824` total nodes — **strictly sealed** until the final winning configuration and thresholds are frozen. Never tune or iterate against Test Steps `42–49`.
4. **Selection Criterion:** Primary selection metric is **Validation PR-AUC** (`average_precision_score`), with Validation Illicit Precision, Recall, F1, and ROC-AUC also logged.
5. **Dual Threshold Reporting:** Evaluate thresholds using both the legacy `0.10–0.90` grid (for apples-to-apples comparison with Stages 3–4C original) and the expanded `0.01–0.99` validation-only grid (consistent with Stage 4C recalibrated), plus recall-constrained ($\text{Val Prec} \ge 0.30$).
6. **Reproducibility:** Fixed seeds (`torch.manual_seed(42)`, `np.random.seed(42)`), deterministic CPU execution, `if not smoke_test:` guards on all disk writes, and full hyperparameter logging in JSON/Markdown.

---

## 2. Diagnostic Motivation from Stage 4C Audit

From [reports/stage4c_audit_report.md](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage4c_audit_report.md), three specific architectural bottlenecks explain why `EvolveGCN-H` (`Test PR-AUC = 0.3107`, `F1 = 0.2251`) trails `HistGradientBoosting` (`Test PR-AUC = 0.5646`, `F1 = 0.6166`) and `Improved Residual GAT` (`Test PR-AUC = 0.3730`):

1. **Unbounded Weight-Norm Drift & Skip Suppression:** `||W1||` grew to `75.89–93.90` and `||W2||` grew monotonically (`8.35 -> 17.80`), causing `h1_conv_norm` (`98.63`) to overpower `h1_skip_norm` (`10.41`) by **~9.5×** before `LayerNorm`. Thus, residual identity preservation collapsed under regime shifts.
2. **Degree-Weighted Averaging Dilutes Fraud Signals (Heterophily):** As seen in Step 46 (`txId=75078587`, `in_deg=45` with `8` licit + `37` unlabeled neighbors), symmetric GCN averaging mixes self-features with legitimate/unlabeled neighbors, whereas **GraphSAGE-style self/neighbor separation** (`[H_self || H_neigh]`) preserves node identity.
3. **Aggressive `pos_weight=7.6349` Calibration Shift:** Weighted BCE pushed `40.9%–70.2%` of unlabeled background nodes above `0.90`, forcing threshold selection toward the extreme tail (`0.97`).

---

## 3. Planned Controlled Experiments (Phased on Steps `1–34` Train / `35–41` Val)

### Experiment 1: Stabilizing EvolveGCN-H Weight Evolution
**Objective:** Prevent `||W_t||` explosion and preserve the local feature highway across all temporal snapshots.

- **Controlled Interventions (tested one factor at a time / clean factorial against `EvolveGCN_H_Directed_Res` baseline):**
  1. **Dual-Branch Normalization + Learnable Gate (`DualLN + Gate`):**
     - Replace `LN(conv + skip)` with `LN_conv(conv) * sigmoid(g) + LN_skip(skip)` (or concatenated `[LN_skip(skip) || LN_conv(conv)]`), eliminating the 9.5× norm imbalance between branches.
  2. **Bounded Weight Evolution:**
     - Test (a) Frobenius weight-norm capping/scaling $\tilde{W}_t = W_t / \max(1, \|W_t\|_F / c)$ with $c \in \{10, 20\}$, and (b) anchored GRU update $W_t = (1 - \alpha) W_0 + \alpha \cdot \text{GRU}(Z_t, W_{t-1})$ with $\alpha \in \{0.25, 0.50\}$ or `LayerNorm` on evolved weight columns.
  3. **Regularization & Monitoring:**
     - Sweep `weight_decay` $\in \{1\text{e-}4, 1\text{e-}3, 5\text{e-}3\}$, `dropout_rate` $\in \{0.3, 0.4\}$, and optional `DropEdge(p=0.15)` on training snapshots `1–34` only.
     - Log per-epoch telemetry: `||W1||_F`, `||W2||_F`, pre-clip gradient norm `||grad||_2`, and `||h1_conv|| / ||h1_skip||` ratio on Train and Validation.

---

### Experiment 2: Regime-Shift Diagnostics (Train/Val Pre-Check & Post-Selection Step 44/46 Verification)
**Objective:** Understand why Step 44 (`|x|_mean = 14.66`, `deg_in = 1.64`, `850` FP) triggered false positives without violating the test-set seal.

- **Phase A (Pre-Selection, Leakage-Free — Steps `1–41` ONLY):**
  - Profile per-snapshot feature norms (`||X_t||`), in/out-degree distributions, unlabeled node percentages, and `||h_conv|| / ||h_skip||` across **Steps `1–34` (Train) and `35–41` (Val)**.
  - Check whether validation snapshots exhibit similar feature-norm dips or sparse-illicit regimes (e.g., validation snapshots with lowest illicit ratio or smallest `|x|_mean`), and measure **worst-snapshot Validation PR-AUC / F1 variance** alongside aggregate Validation PR-AUC.
  - Check whether **feature-norm / degree-aware gating** or **GraphSAGE self/neighbor concatenation** reduces validation false positives on low-norm licit nodes.
- **Phase B (Strictly Post-Selection Verification — Steps `42–49`):**
  - Once the final winning model is selected, run the identical diagnostic on Test Steps `42–49` (specifically auditing Step 44 `FP` and `LabMedP`, plus Step 46's 2 illicit nodes) to verify whether the stabilization and architecture fixes resolved Step 44's drift.

---

### Experiment 3: `EvolveGCN-O` (Feature-Independent Weight Evolution)
**Objective:** Test whether removing `TopKSummarizer(H_t)` from weight evolution prevents snapshot background/feature-norm shifts (like Step 44's `0 illicit / 70 licit / 58 unlabeled` Top-$k$ selection) from corrupting $W_t$.

- **Architecture (`FraudEvolveGCNO`):**
  - `EvolveGCNOLayer(d_in, d_out)`: Evolves $W_t^{(l)}$ strictly from $W_{t-1}^{(l)}$ via `GRUCell` (or `LSTMCell`) over the `d_out` filter rows of $W_{t-1}^\top$ without conditioning on `TopK(H_t)`.
  - Uses the same stabilized `DualLN + Residual/Concat` spatial block, directed edges, and `LayerNorm` classification head as stabilized `EvolveGCN-H` for a fair 1-to-1 comparison.
- **Comparison Criterion:** Compare stabilized `EvolveGCN-H` vs. `EvolveGCN-O` strictly on Validation Steps `35–41` PR-AUC, Validation F1, and per-validation-snapshot calibration stability.

---

### Experiment 4: Temporal Attention Across Historical Graph Snapshots
**Objective:** Test whether non-Markovian causal attention across past snapshots outperforms recurrent single-step state evolution (`GRUCell` / `LSTMCell`).

- **Constraint Check (Ephemeral Transaction Graphs):** Transaction IDs do not persist across snapshots (`0` cross-timestep edges), so temporal attention operates at the **snapshot-summary / weight-bank level** using strictly causal indices $\tau \le t$ (or $\tau < t$):
  1. **Model 4A — `TemporalAttn-GCN` (Causal Snapshot-Context Attention + FiLM Conditioning):**
     - Compute per-snapshot unsupervised spatial summary $g_\tau = \text{MeanMaxPool}(H_\tau)$ for $\tau \in [1, t-1]$ within a causal sliding window $W \in \{6, 12\}$.
     - Apply causal multi-head scaled dot-product attention (`Query`: current snapshot spatial summary $g_t^{\text{pre}}$ or node embeddings, `Keys/Values`: $\{g_{\max(1,t-W)}, \dots, g_{t-1}\}$ with relative time-lag embeddings $\Delta t$).
     - Instead of Stage 4B's additive logit bias, condition spatial node embeddings via **FiLM (Feature-wise Linear Modulation)**: $\tilde{h}_i = (1 + \gamma_t) \odot \text{LN}(h_i) + \beta_t$ (bounded by $\tanh$).
  2. **Model 4B — `Attn-EvolveGCN` (Causal Attentive Weight-Bank):**
     - Maintain causal historical weight matrices $\{W_{t-W}, \dots, W_{t-1}\}$ and compute attention-weighted historical weight prior $\bar{W}_{t-1} = \sum_{j=1}^{W} \alpha_{t, j} W_{t-j}$ before applying the stabilized `GRUCell` step.
- **Evaluation:** Compare both temporal attention variants against stabilized `EvolveGCN-H` and `EvolveGCN-O` on Validation PR-AUC (`35–41`).

---

### Experiment 5: Class-Imbalance Loss Function Comparison (Weighted BCE vs. Focal Loss)
**Objective:** Determine whether `pos_weight=7.6349` over-inflates background probabilities and whether `FocalLoss` or moderated weighting yields superior ranking (PR-AUC) and threshold calibration.

- **Controlled Loss Sweep (evaluated on the top stabilized temporal/static GNN architecture from Experiments 1–4, tuned strictly on Validation Steps `35–41`):**
  1. **Weighted BCE (Baseline):** `BCEWithLogitsLoss(pos_weight=7.6349)`
  2. **Moderated Weighted BCE:** `pos_weight` $\in \{1.0, 2.76\ (\sqrt{7.6349}), 4.0\}$
  3. **Binary Focal Loss (`FocalLoss` from `src/train_gnn_static.py`):**
     - `alpha` $\in \{0.25, 0.50, 0.75, 0.85\}$
     - `gamma` $\in \{1.5, 2.0\}$
  4. **Hybrid BCE + Focal / Label-Smoothed Focal:** Optional check if Focal Loss improves probability separation without collapsing recall.
- **Evaluation Metrics:** Validation PR-AUC (primary), Validation ROC-AUC, Validation F1 across both `0.10–0.90` and `0.01–0.99` grids, and Validation unlabeled/licit mean/median probability calibration.

---

### Experiment 6: Value-Added Test of Graph Message Passing Beyond the 182 Features
**Objective:** Rigorously test whether graph message passing provides predictive signal beyond the precomputed 182 Elliptic++ features (`93` Local + `72` 1-hop Aggregate + `17` Domain), and diagnose why `HistGradientBoosting` (`Test PR-AUC = 0.5646`) and `FraudMLP` (`Test PR-AUC = 0.4742`) currently outperform GNNs.

- **Controlled 2 × 3 Diagnostic Matrix (Steps `1–34` Train, `35–41` Val):**

| Feature Regime | `No-Graph (A = I)` (MLP-Equivalent) | `Averaging GCN` ($\hat{A} H W + \text{skip}$) | `Separated Self+Neighbor` (`[H_self \|\| H_neigh]`, GraphSAGE/Skip-Concat) |
| :--- | :---: | :---: | :---: |
| **Local-Only (`93` dims)** (`Local_feature_1..93`) | Tests pure local tabular signal | Tests if message passing reconstructs 1-hop/2-hop neighborhood fraud context | Tests if separated self + neighbor aggregation beats averaging |
| **Full Elliptic++ (`182` dims)** (`Local + Agg + Domain`) | Tests deep tabular ceiling with identical optimizer/loss/head | Tests if standard GCN averaging adds value or degrades the 182 features | Tests if **concatenating** neighbor message passing (`A_no_loop @ H`) onto intact `182` self-features adds complementary 2-hop/ring signal |

- **Scientific Hypothesis:**
  - On `Local-Only (93 dims)`, graph message passing should yield a large Validation PR-AUC gain over `A = I`.
  - On `Full (182 dims)`, `Separated Self+Neighbor` (`GraphSAGE` / `MixHop` / `Concat-Residual` with **neighbor-only** aggregation `mean/attn(N(v))` concatenated to projected `x_v`) avoids diluting precomputed node features and allows the GNN to learn **strictly complementary** higher-order topology (2-hop+ and unlabeled bridge paths) without degrading below `FraudMLP`.

---

## 4. Planned Execution Sequence & Module Structure

### Proposed New Files (Separate from Existing Working Modules)
1. `src/models_nextgen.py` — Clean PyTorch/PyG implementations of:
   - `StabilizedEvolveGCNH` (Dual-LN, gated skip/concat, bounded `W_t` evolution)
   - `StabilizedEvolveGCNO` (Weight-only `GRUCell` evolution)
   - `TemporalAttnGCN` / `AttnEvolveGCN` (Causal snapshot attention + FiLM / weight-bank)
   - `SeparatedGraphNet` / `GatedTemporalSAGE` (Supports `Local-93` vs `Full-182`, `A=I`, `GCN-Avg`, and `[H_self || H_neigh]` modes)
2. `src/train_nextgen.py` + `train_nextgen.py` — Controlled experiment runner with:
   - `--smoke-test` mode (`2` epochs, zero disk writes to production paths)
   - Training/Validation-only loop for Experiments 1–6 (Steps `1–34` train, `35–41` val)
   - Strict Validation PR-AUC model selection + validation-only threshold tuning (`0.10–0.90` & `0.01–0.99`)
   - Single-pass post-selection Test evaluation (`42–49`) ONLY for the final selected champion (plus clean post-selection ablation summary for Experiment 6's graph-value test if explicitly authorized)
3. **Planned Output Artifacts (Stage 4D, non-destructive):**
   - `models/nextgen_gnn_best.pt`
   - `reports/stage4d_nextgen_experiments_metrics.json`
   - `reports/stage4d_nextgen_experiments_report.md`
   - `results/stage4d_nextgen_comparison.png`

---

## 5. Success Criteria & Honest Evaluation Policy
- Every candidate will be compared against:
  - `EvolveGCN_H_Directed_Res` (Stage 4C Original & Recalibrated: Val PR-AUC `0.6387`, Test PR-AUC `0.3107`, Test F1 `0.2251`)
  - `Improved_Residual_GAT` (Test PR-AUC `0.3730`, Test F1 `0.2988`)
  - `Improved_Residual_GCN` (Test PR-AUC `0.2761`, Test F1 `0.3210`)
  - `MLP_Baseline` (Val PR-AUC `0.7788`, Test PR-AUC `0.4742`, Test F1 `0.5360`)
  - `HistGradientBoosting` (Val PR-AUC `0.9292`, Test PR-AUC `0.5646`, Test F1 `0.6166`)
- **Zero-Hype Reporting:** If graph message passing on `182` features fails to beat `MLP_Baseline` or `HistGradientBoosting`, that negative/nuanced finding (and the `Local-93` vs `Full-182` ablation explanation) will be reported accurately rather than overclaimed.

---

**Status:** Plan documented for review. **No training has been started and no existing files were modified.** Awaiting your approval to begin implementing `src/models_nextgen.py` and `src/train_nextgen.py` (with smoke tests first).
