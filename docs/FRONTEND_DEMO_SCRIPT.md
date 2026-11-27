# FraudLens 10–15 Minute Live Project Guide Demonstration Script

**Goal:** Give a clear, confident, mistake-free demonstration to your college project guide using real backend inference—showing **zero fabricated data** and explaining every technical concept simply.

**Quick Choice of Demo Dataset:**
- **Fast Demo (Recommended for Live Upload on Stage, `< 2` seconds total):** Upload [`app/sample_data/transactions_sample.csv`](../app/sample_data/transactions_sample.csv), [`app/sample_data/edges_sample.csv`](../app/sample_data/edges_sample.csv), and optional [`app/sample_data/classes_optional_sample.csv`](../app/sample_data/classes_optional_sample.csv) (`13` nodes, `12` edges, `1` complete `fan_in_consolidation` Tier-A candidate ring around `txId = 73282397`).
- **Pre-Loaded Large Snapshot Demo (Already saved in SQLite `fraud_app.db`):** Click existing Step 42 run **`stage6e_e2e_audit_step42`** (`7,140` nodes, `8,493` edges, `612` flagged, `80` clusters, `24` Tier-A candidates) to show large-scale analytics and cached `M=200` IG on hub `73282397` (`0.02s` cache hit!).

---

## Before Your Guide Walks In (2-Minute Setup)
1. Open your `Capstone_Fraud_Detection` project folder and double-click **`start_fraudlens.bat`**.
2. Wait until both windows show Uvicorn (`http://127.0.0.1:8010`) and Vite (`http://127.0.0.1:5173`).
3. Open Chrome/Edge to `http://127.0.0.1:5173` and verify the top-right pill says **`API Online • Threshold 0.57`**.

---

## Demonstration Timeline (12 Minutes Total)

### Step 1 — Dashboard & System Status (Minutes 0:00 – 2:00)
- **What to click:** Start on `/` (**Dashboard**). Hover briefly over the top-right **`API Online • Threshold 0.57`** status pill.
- **What to show on screen:** Left sidebar (`ExpA2_RankGauss182_BiDirGCN`, *"Batch CSV upload only"*), the blue **Batch Offline Evaluation Engine** banner, the 4 KPI cards, the two Recharts bar charts, and **Recent Analysis History**.
- **What to explain verbally:**
  *"Sir/Ma'am, this is **FraudLens**, our web application built on top of our frozen Stage 4H Graph Neural Network (`ExpA2_RankGauss182_BiDirGCN`). Notice the status pill connects live to our FastAPI backend (`GET /api/health`) and confirms our frozen decision threshold `0.57` (calibrated on Validation Steps `35–41`, never test labels). Every number here comes straight from SQLite—there is no live blockchain stream or fake hardcoded data."*
- **What the result demonstrates:** Clean frontend-backend health integration, strict zero-leakage threshold preservation, and honest batch architecture.
- **Likely Question from Guide:** *"Why don't you connect to a live Bitcoin feed?"*  
  **Simple Answer:** *"Elliptic++ transactions have 182 proprietary pre-engineered features (including 1-hop/2-hop neighborhood aggregates) that don't exist in raw public Bitcoin RPC streams, so batch CSV validation ensures exact feature parity with our trained RankGauss pipeline."*

---

### Step 2 — CSV Upload, Schema Integrity & Validation Gate (Minutes 2:00 – 4:30)
- **What to click:** Click **Upload CSV Bundle** in the sidebar. Type Run Label `guide_demo_fanin`. Drag/select `app/sample_data/transactions_sample.csv` into **transactions.csv**, `app/sample_data/edges_sample.csv` into **edges.csv**, expand **Optional File 3**, and select `classes_optional_sample.csv`. Point out that **2. Run Fraud Analysis** is currently **disabled**. Now click **`1. Validate CSV Bundle`**.
- **What to show on screen:** Filenames/sizes, the blue **Optional Class Labels Isolation** banner, the **Required Feature Schema** box, and the green **`Validation Passed`** card (`13` transactions, `12` unique edges, Timestep `42`, `12 Illicit / 0 Licit / 1 Unk`).
- **What to explain verbally:**
  *"We enforce a two-step gate: users cannot run inference until the backend validates all `182` feature columns and checks graph integrity (`POST /api/validate-csv`). Unlike naive scripts, we never silently drop dangling edges or cross-timestep edges—the API rejects them with structured `HTTP 422` diagnostics. Also notice optional `class` labels are strictly ignored during inference."*
- **What the result demonstrates:** Production-grade input validation and strict label leakage prevention.
- **Likely Question from Guide:** *"Wait—if `classes_optional_sample.csv` has the answers, couldn't the model be cheating?"*  
  **Simple Answer:** *"No, in `csv_validator.py` the `class` column is split out completely before creating `X_raw [N, 182]`. Even if we omit `classes.csv` or flip labels, every predicted probability stays bit-identical—we even proved that in `pytest` (`test_optional_class_isolation_from_inference`)."*

---

### Step 3 — Run Frozen Inference & Inspect Run Details (Minutes 4:30 – 6:00)
- **What to click:** Click **`2. Run Fraud Analysis`**. Wait ~1 second for automatic navigation to `/runs/:runId`.
- **What to show on screen:** The **Run Details** KPI cards (`13` nodes, `13` flagged, `1` cluster, `1` Tier-A candidate), the **Execution & Pipeline Metadata** card (`Stage2 Imputer+Scaler -> Stage4H RankGauss`), and the **Post-Hoc Optional Label Comparison** box.
- **What to explain verbally:**
  *"Clicking Analyze executes the exact frozen Stage 2 median imputer & standard scaler, Stage 4H RankGauss quantile normalizer, and our BiDirGCN checkpoint in `.eval()` mode (`no_grad`). Look at the metadata card showing the exact preprocessing order and `Snapshot Tensor Cache: Available`."*
- **What the result demonstrates:** End-to-end frozen inference speed and transparent post-hoc evaluation separation.
- **Likely Question from Guide:** *"Why did you need RankGauss before BiDirGCN?"*  
  **Simple Answer:** *"In Stage 4G/4H we found extreme distribution drift and tails after Step 42; RankGauss maps skewed features onto a smooth normal curve using only training steps `1–34`, improving Test PR-AUC from `0.5268` (`GCNAvg`) up to `0.6023` and slashing calibration ECE from `0.1145` down to `0.0170`."*

---

### Step 4 — Transaction Risk Table, Filtering & Search (Minutes 6:00 – 7:30)
- **What to click:** Click **`Explore Transactions`**. On the Transaction Risk Table, show the Risk Dropdown (**Flagged Illicit (>= 0.57)**) and type `73282397` into **Search numeric txId...** -> Click **Search** -> Click **Clear**. *(Optional: you can also jump to the pre-saved full 7,140-node run `stage6e_e2e_audit_step42` via Dashboard -> Txs to show multi-page pagination `Page 1 of 13` and `612` flagged vs `7,140` total!)*
- **What to show on screen:** Probability bars (`ProbBar`), `RiskBadge` (`High Predicted Risk (P >= 0.85)`), `In/Out Deg` (`12 / 0`), clickable `C_T42_001` cluster badge, and `GtBadge` (`Unknown` vs `Illicit`).
- **What to explain verbally:**
  *"Analysts triage alerts starting from highest predicted probability. Look closely at `txId = 73282397`: its ground truth in Elliptic++ is `Unknown`, yet the model flags it at `99.4%` (`High Predicted Risk`) because `12` illicit transactions all flow into it!"*
- **What the result demonstrates:** Real GNN relational value—catching suspicious unlabeled hubs (`Unknown != Licit`) via neighborhood topology.
- **Likely Question from Guide:** *"Is `Unknown` the same as `Licit`?"*  
  **Simple Answer:** *"Definitely not—77.15% of Elliptic++ transactions are `Unknown (-1)`. Treating Unknown as Licit hides real fraud and ruins evaluation metrics, so FraudLens always isolates `Unknown` cleanly."*

---

### Step 5 — Suspicious Clusters & Interactive Cytoscape Graph (Minutes 7:30 – 9:30)
- **What to click:** Click **`C_T42_001`** (or click **Inspect Suspicious Clusters** -> **Open Cluster** on `C_T42_001`). On the graph page (`/clusters/C_T42_001`), toggle **Show 1-Hop Context** (especially vivid if showing the full Step 42 `C_T42_001` with `38` core + `20` context nodes!), try switching layout (`Force-Directed Layout` vs `Directed Flow Layout`), click **Zoom in** and **Fit graph to screen**, and click node `73282397` (or a neighbor row in **Subgraph Nodes**).
- **What to show on screen:** The yellow **Model-Identified Suspicious Subgraph** caution banner, the visual legend, directed triangle arrowheads pointing into the teal-ringed **Cluster Hub**, the Selected Transaction Node inspector, and the full **Directed Edges (`Src -> Dst`)** table.
- **What to explain verbally:**
  *"Instead of 38 separate confusing alerts, FraudLens groups connected flagged transactions into a single `Potential Fraud-Ring Candidate` (`C_T42_001`). Every edge keeps its exact `source -> target` direction. Notice the yellow warning: we explicitly label these as model-identified suspicious patterns, never confirmed criminal rings."*
- **What the result demonstrates:** Stage 5C directed subgraph motif analysis (`fan_in_consolidation` / `complex_mesh`), zero undirected graph assumptions, and synced UI interactivity.
- **Likely Question from Guide:** *"How do you decide what is a Tier-A Ring Candidate?"*  
  **Simple Answer:** *"Strictly label-free structural rules from Stage 5C: at least 3 connected flagged nodes (`>= 0.57`), mean cluster probability `>= 0.75`, at least 2 high-confidence nodes (`>= 0.85`), and at least 2 directed internal edges."*

---

### Step 6 — On-Demand Integrated Gradients XAI (Minutes 9:30 – 12:00)
- **What to click:** In the Selected Transaction Node card for `73282397`, click **`Inspect & Explain Transaction (IG)`**. Point out the separated **Model Predicted Class** (`Illicit (Predicted)`) vs **Actual Ground-Truth Class (`Unknown`)**, and point out that no XAI ran automatically yet. Now click **`Explain Prediction (IG)`**! *(Tip: on the `13`-node sample run this finishes in `< 1s`; on full `7,140`-node Step 42 it takes `~28s` first time or `< 0.03s` if already cached.)*
- **What to show on screen:**
  1. Top Positive (red) vs. Negative (green) horizontal Recharts BarChart & feature tables (`Graph IG`, `Self`, `Nbr`).
  2. **Feature Group Attribution Summary** (`Local Features (93)`, `Aggregate Features (72)`, `Domain Features (17)`).
  3. **Explanation Quality & Completeness** box (`Method: Composite_Midpoint_IG_3Zone_M200`, `M = 200`, `Completeness Residual < 0.01`, `Relative Error < 0.2%`, `All Finite`, Self % vs Neighbor % shares).
  4. **Directed 1-Hop Neighborhood Influence** table.
- **What to explain verbally:**
  *"We compute Integrated Gradients only when an analyst asks for it, using our Stage 5B 3-zone Composite Midpoint `M=200` quadrature (which reduced worst-case ReLU completeness error over 3× compared to uniform grids). Here we see exact positive fraud drivers like `Local_feature_3`, balanced against negative counter-signals like `Aggregate_feature_10`. We even split each feature into `Self` vs `Neighbor` impact and run a live Completeness Axiom check (`sum(IG) ≈ Δlogit`) right in the UI! Finally, clicking Refresh Explanation hits our SQLite cache instantly."*
- **What the result demonstrates:** Real mathematical transparency, computational completeness verification, and scalable on-demand caching.
- **Likely Question from Guide:** *"Why are some bars green (negative) on a 99.5% fraud prediction?"*  
  **Simple Answer:** *"A deep GNN weighs competing evidence—like a judge hearing pros and cons. Even though `Aggregate_feature_10` pulled the logit down (`-6.71`), the total positive features (`Local93 = +17.01` + `Domain17 = +0.42`) strongly outweighed the negatives, leaving a high net logit (`+5.23` -> `99.47%` probability)."*
