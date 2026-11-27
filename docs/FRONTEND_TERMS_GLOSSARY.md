# FraudLens Beginner-Friendly Technical Glossary

This glossary explains every technical term shown in the **FraudLens** interface in simple language, with concrete examples from the Elliptic++ Bitcoin graph and the built-in `app/sample_data/` bundle.

---

### 1. Transaction
- **Simple Meaning:** A single payment event recorded in the dataset, identified by a unique number (`txId`) and described by 182 numerical measurements (fees, size, Bitcoin amounts, and neighbor statistics).
- **Example:** In `app/sample_data/transactions_sample.csv`, `txId = 73282397` is one Bitcoin transaction in Time Step `42`.

### 2. Node
- **Simple Meaning:** In graph analytics, each transaction is drawn as a circle (a **node** or vertex). FraudLens represents **transactions** as nodes—not individual user bank accounts.
- **Example:** When you open cluster `C_T42_001` in `app/sample_data/`, the graph draws **13 nodes** (one circle for each of the 13 transactions).

### 3. Edge
- **Simple Meaning:** A connection (arrow) between two transactions showing that funds flowed from one transaction (`txId1`) into another (`txId2`).
- **Example:** In `app/sample_data/edges_sample.csv`, the row `73003348,73282397` is one directed edge showing money flowing from transaction `73003348` to `73282397`.

### 4. Directed Graph
- **Simple Meaning:** A network where every connection has a **one-way direction** (`Source -> Target`), shown as an arrowhead. Reversing an arrow changes its financial meaning (sending money vs. receiving money).
- **Example:** In `C_T42_001`, 12 arrows point *into* `73282397` (`fan_in_consolidation`), meaning 12 upstream transactions consolidate outputs into one hub transaction—not the other way around.

### 5. Timestep (Time Step / Snapshot)
- **Simple Meaning:** A discrete 2-week time window (`1` to `49` in Elliptic++). Every edge in this dataset connects two transactions inside the **exact same timestep**, so each timestep acts as an independent graph snapshot.
- **Example:** All 13 transactions in `app/sample_data/transactions_sample.csv` belong to `Time step = 42`.

### 6. Fraud Probability (`pred_prob`)
- **Simple Meaning:** A number between `0.0000` (`0%`) and `1.0000` (`100%`) output by the frozen AI model (`ExpA2_RankGauss182_BiDirGCN`) indicating how strongly this transaction resembles illicit patterns learned from training data. **It is a risk score, not legal proof.**
- **Example:** On `app/sample_data/`, `txId = 73282397` receives a fraud probability of **`0.9940`** (displayed on screen as `0.9940`, or `99.4%` in risk bars) and on full Step 42 receives **`0.9947`**.

### 7. Threshold (`0.57`)
- **Simple Meaning:** The fixed cutoff score used to decide whether a transaction gets flagged as **Illicit (Predicted)**. Any transaction with probability **greater than or equal to `0.57`** is flagged; anything below `0.57` is classified **Licit (Predicted)**.
- **Why `0.57`?** It was calibrated strictly on Validation Steps `35–41` during Stage 4H and frozen permanently so decisions never leak test labels.
- **Example:** Probability `0.62 >= 0.57` -> **Illicit (Predicted)**; probability `0.41 < 0.57` -> **Licit (Predicted)**.

### 8. Predicted Class (`Illicit` / `Licit`)
- **Simple Meaning:** The model's binary decision (`Illicit` if probability `>= 0.57`, else `Licit`). On the Transaction Details page, FraudLens explicitly writes **`Illicit (Predicted)`** or **`Licit (Predicted)`** so you never confuse a model prediction with a confirmed police/legal verdict.
- **Example:** Transaction `73282397` has probability `0.9947 >= 0.57`, so its predicted class is **`Illicit (Predicted)`**.

### 9. Risk Category
- **Simple Meaning:** A human-friendly risk band computed by the backend to help analysts triage alerts from highest urgency to lowest:
  - **High Predicted Risk (`P >= 0.85`):** Red badge — very high model confidence.
  - **Elevated Risk (`Flagged >= 0.57`):** Orange badge — crossed the `0.57` alert threshold (`0.57 <= P < 0.85`).
  - **Moderate Monitor (`0.35 - 0.57`):** Yellow badge — below alert threshold, slightly elevated.
  - **Low Predicted Risk (`< 0.35`):** Green badge — low predicted risk.
- **Example:** `txId = 73282397` (`P = 0.9947`) displays the red **`High Predicted Risk (P >= 0.85)`** badge.

### 10. In-Degree and Out-Degree (`In / Out Deg`)
- **Simple Meaning:**
  - **In-Degree:** How many incoming arrows (`other -> this tx`) enter the transaction inside its timestep.
  - **Out-Degree:** How many outgoing arrows (`this tx -> other`) leave the transaction inside its timestep.
- **Example:** In full Step 42, hub transaction `73282397` has degree **`25 / 0`** (`25` incoming edges feeding into it, `0` outgoing edges leaving it).

### 11. Suspicious Cluster
- **Simple Meaning:** A connected group of **2 or more** flagged transactions (`probability >= 0.57`) linked directly by directed edges in the same timestep. FraudLens finds these groups automatically to uncover coordinated activity instead of isolated single alerts.
- **Example:** Uploading `app/sample_data/` discovers 1 suspicious cluster **`C_T42_001`** containing all `13` connected flagged transactions.

### 12. Tier-A Candidate (`Potential_Fraud_Ring_Candidate`)
- **Simple Meaning:** A stricter, higher-priority suspicious cluster that meets **all three** label-free Stage 5C structural rules:
  1. At least **3** connected flagged nodes (`>= 3`).
  2. Average fraud probability **`>= 0.75`** AND at least **2** high-confidence core nodes (`P >= 0.85`).
  3. At least **2** internal directed edges linking those flagged nodes.
- **Example:** Cluster `C_T42_001` in `app/sample_data/` has `13` nodes (`>= 3`), mean probability `0.9977` (`>= 0.75`), `13` nodes `>= 0.85`, and `12` internal edges (`>= 2`), earning the red **`Ring Candidate`** badge. *(Still an investigative signal, not legal proof.)*

### 13. Integrated Gradients (`IG` / `Composite_Midpoint_IG_3Zone_M200`)
- **Simple Meaning:** An Explainable AI (XAI) technique that answers *"Why did the model give this transaction score X instead of an average baseline score?"* by smoothly interpolating 200 steps (`M = 200`) from a zero RankGauss baseline (`median transaction`) up to the actual transaction and summing each feature's gradient sensitivity.
- **Example:** When you click **Explain Prediction (IG)** on `73282397`, the backend evaluates 200 steps and reports that feature **`Local_feature_3`** added **`+5.23532`** logit units toward Fraud.

### 14. Feature Contribution (`Graph IG`, `Self IG`, `Neighbor IG`)
- **Simple Meaning:** How much a single numerical attribute pushed the model's log-odds score (`logit`) up or down:
  - **Positive (`+`, red):** Pushed the prediction **toward Fraud / Illicit**.
  - **Negative (`-`, green):** Pushed the prediction **away from Fraud / toward Licit**.
  - **Graph IG = Self IG + Neighbor IG** (how much came from the transaction's own features vs. connected neighbors' features).
- **Example:** On full Step 42 `txId = 73282397`, `Local_feature_3` has `Graph IG = +5.23532` (`Self = +4.06241`, `Neighbor = +1.17291`), whereas `Aggregate_feature_10` has `Graph IG = -6.70889` (`Self = -2.27231`, `Neighbor = -4.43657`).

### 15. Completeness Error (Residual & Relative Error `%`)
- **Simple Meaning:** A built-in mathematical accuracy check for Integrated Gradients. Ideally, adding up **all** feature & neighbor IG contributions should equal the exact change in model logit (`Logit(tx) - Logit(baseline)`). Because neural networks with ReLU layers have tiny corners, numerical integration leaves a tiny **Completeness Residual**. Smaller residual means the explanation is mathematically faithful.
- **Example:** On `txId = 73282397` (full Step 42), the logit change is `9.56030`, the sum of all IG values is `9.55166`, giving a tiny residual of **`0.008639 logits`** (**`0.0904%` relative error**, well below the `1.0%` benchmark threshold).

### 16. Ground-Truth Label (`Optional GT`: `Illicit`, `Licit`, `Unknown`)
- **Simple Meaning:** Real historical audit labels from Elliptic++ (only present if uploaded in a `class` column or `classes.csv`).
  - **`Illicit` (`1`):** Confirmed darknet/fraud entity in Elliptic++.
  - **`Licit` (`2` or `0`):** Confirmed legitimate exchange/service in Elliptic++.
  - **`Unknown` (`3` or `-1`):** Unlabeled transaction (`77.15%` of Elliptic++!). FraudLens strictly keeps `Unknown` separate from `Licit` and **never feeds labels into the model during prediction**.
- **Example:** In `app/sample_data/classes_optional_sample.csv`, `73282397` has Ground Truth **`Unknown`** (`class=3`), demonstrating how the GNN can flag an unlabeled hub solely from its features and 12 suspicious neighbors.

### 17. Batch Inference
- **Simple Meaning:** Running the AI model on a complete uploaded snapshot file (`CSV`) all at once on demand—the opposite of continuous live real-time blockchain monitoring. FraudLens strictly performs **batch CSV analysis** (`streaming_supported: false`).
- **Example:** Uploading `transactions_sample.csv` + `edges_sample.csv` scores all `13` transactions in `~0.1–0.3` seconds and stores the results in SQLite.
