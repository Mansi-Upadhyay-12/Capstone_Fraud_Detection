# FraudLens Complete Frontend User Manual

**Application:** **FraudLens — Intelligent Transaction Risk Analysis** (*Graph-Based Fraud Detection & Explainable AI*)  
**Frozen AI Model:** `ExpA2_RankGauss182_BiDirGCN` (`seed=42`, decision threshold `τ* = 0.57`)

This manual explains **every single component**, **how to operate the full 12-step user workflow**, and **all honest system limitations** in beginner-friendly English.

---

## PART A — COMPONENT-BY-COMPONENT MANUAL

Every component discovered in [`app/frontend/src/`](../app/frontend/src) is explained below using the exact 12-point structure.

---

### 1. Sidebar Navigation, Topbar Header & Backend Status Pill (`AppLayout`)

1. **What is this component?**  
   The permanent navigation shell wrapping every FraudLens screen: a dark-navy left sidebar, a sticky top header bar, a live API health pill, and a mobile slide-out drawer with backdrop.
2. **Why does FraudLens need it?**  
   Users need one-click movement between Dashboard, CSV Upload, Run Overview, Transaction Explorer, and Clusters—plus immediate proof that the Python AI backend is online.
3. **Where can I find it?**  
   Visible on every route (`/`, `/upload`, `/runs/...`) in [`AppLayout.tsx`](../app/frontend/src/components/AppLayout.tsx).
4. **What does each button/control do?**  
   - **Dashboard (`/`), Upload CSV Bundle (`/upload`):** Primary workspace links.
   - **Run Overview, Transaction Risk Table, Suspicious Clusters:** Sub-navigation links targeting the currently viewed run—or the newest run if none is open in the URL.
   - **API Status Pill + Refresh Button (`↺`):** Calls `GET /api/health` (`onRefreshHealth`) and displays green `API Online • Threshold 0.57`, yellow `Checking API...`, or red `Backend Offline`. Hovering shows compute device (`cpu`/`cuda`) and feature count (`182`).
   - **Mobile Menu Toggle (`☰`) & Close (`✕`) / Backdrop:** Opens and dismisses the slide-out sidebar below `820px` screen width.
5. **How do I use it step by step?**  
   Look at the top-right status pill first. If it shows `Backend Offline`, start `start_backend.bat` and click the `↺` refresh icon. Then use the left sidebar links to move across views.
6. **What information must I enter or select?**  
   None—click any navigation link or the `↺` refresh button.
7. **What happens after I click it?**  
   Clicking a link switches the main view lazily without reloading the browser; clicking `↺` fires `GET /api/health`.
8. **What result will appear?**  
   The selected page renders inside `<main className="page-content">`, and the status dot updates to glowing teal (`API Online • Threshold 0.57`).
9. **How should I understand that result?**  
   Glowing teal means the FastAPI server loaded `stage2_imputer_scaler_train_1_34.joblib`, `rankgauss_transformer_train_1_34.joblib`, and `ExpA2_RankGauss182_BiDirGCN_seed42.pt` at frozen threshold `0.57`. The sidebar footer reminds you this is **Batch CSV mode** (no live blockchain stream).
10. **Real example using existing sample dataset:**  
    After uploading `app/sample_data/` and opening run `run_20261126_...`, the sidebar switches its section label from **Latest Run** to **Selected Run** and activates links for `Run Overview`, `Transaction Risk Table`, and `Suspicious Clusters`.
11. **What should I say to my project guide about it?**  
    *"The top status indicator queries `/api/health` on the FastAPI backend and confirms that our frozen Stage 4H BiDirGCN model (`ExpA2_RankGauss182_BiDirGCN`) and calibrated threshold (`0.57`) are loaded in offline batch mode."*
12. **Common mistakes or things I should avoid:**  
    Do not try to upload CSVs while the pill says `Backend Offline`—start `start_backend.bat` first, then click `↺`.

---

### 2. Dashboard KPI Summary Cards (`DashboardPage`)

1. **What is this component?**  
   A 4-card top metrics row summarizing totals across all completed CSV analysis runs stored in SQLite.
2. **Why does FraudLens need it?**  
   Analysts need an instant snapshot of total processed volume, alert volume, cluster volume, and speed without opening individual run pages.
3. **Where can I find it?**  
   Top of `/` ([`DashboardPage.tsx`](../app/frontend/src/pages/DashboardPage.tsx#L162-L202)) whenever at least `1` run exists.
4. **What does each button/control do?**  
   Display-only cards summarizing metrics computed purely by summing `runs` returned from `GET /api/runs`:
   - **Total Analysed Transactions:** Sum of `node_count` across runs.
   - **High-Risk / Flagged Transactions:** Sum of `flagged_count` and aggregate `%` at `0.57`.
   - **Suspicious Clusters:** Sum of `direct_clusters_count` and Tier-A `potential_candidates_count`.
   - **Analysis Run Count:** `runs.length` plus latest `inference_time_sec`.
5. **How do I use it step by step?**  
   Open `/`, verify the backend is online, and read the 4 KPI numbers from left to right.
6. **What information must I enter or select?**  
   None.
7. **What happens after I click it?**  
   Cards are non-interactive; click **New CSV Analysis** above them or links below to inspect a run.
8. **What result will appear?**  
   Exact integer counts formatted with commas (e.g., `7,140`, `612`).
9. **How should I understand that result?**  
   Every figure comes strictly from real SQLite runs (`GET /api/runs`)—nothing is hardcoded or simulated.
10. **Real example using existing sample dataset:**  
    If your only persisted run is the `app/sample_data/` bundle (`transactions_sample.csv` + `edges_sample.csv`), the cards show: **Total Analysed Transactions: `13`**, **High-Risk / Flagged Transactions: `13` (`100.00%`)**, **Suspicious Clusters: `1` (`1 Tier-A ring candidate(s)`)**, **Analysis Run Count: `1`**.
11. **What should I say to my project guide about it?**  
    *"All four KPI cards aggregate real runs stored in our SQLite database (`fraud_app.db`), with flagged counts strictly reflecting the frozen `0.57` decision threshold."*
12. **Common mistakes or things I should avoid:**  
    Remember these cards sum across *all* saved runs (`runs.length`)—for a single dataset's metrics, open **Run Details**.

---

### 3. Dashboard Analytics Charts & Latest Analysis Summary (`DashboardPage`)

1. **What is this component?**  
   Two side-by-side **Recharts** bar charts (`Risk Band Distribution` and `Per-Timestep Flagged & Cluster Counts`) plus the **Latest Analysis Summary** strip for `runs[0]`.
2. **Why does FraudLens need it?**  
   Tables alone hide distribution skew and time-step spikes; bar charts immediately show how many nodes sit in `Low (<0.35)` vs `High (>=0.85)`.
3. **Where can I find it?**  
   Middle section of `/` ([`DashboardPage.tsx`](../app/frontend/src/pages/DashboardPage.tsx#L204-L321)).
4. **What does each button/control do?**  
   - **Run Details →**, **Transactions**, **Clusters** buttons on the Latest Analysis Summary navigate straight to `/runs/{latestRunId}`, `/transactions`, or `/clusters`.
   - Hovering over any Recharts bar displays an exact count tooltip (`band`, `count`, or `flagged` / `clusters`).
5. **How do I use it step by step?**  
   Hover your mouse over the colored bars to read exact numbers, or click **Run Details →** to dive into the newest run.
6. **What information must I enter or select?**  
   None (`DashboardPage` automatically fetches `GET /api/runs/{runs[0].run_id}`).
7. **What happens after I click it?**  
   Clicking one of the 3 summary buttons opens that specific sub-page for the latest run.
8. **What result will appear?**  
   On **Risk Band Distribution**, 4 colored bars (`High >=0.85` red, `Elevated >=0.57` orange, `Moderate 0.35-0.57` yellow, `Low <0.35` green) appear instantly (`isAnimationActive={false}`). On **Per-Timestep**, blue (`Flagged Txs`) and teal (`Clusters`) bars appear per timestep (`T42`, etc.).
9. **How should I understand that result?**  
   In realistic datasets, the green `Low (<0.35)` bar is tall while red/orange bars are small (~8–9%), reflecting Bitcoin's severe class imbalance.
10. **Real example using existing sample dataset:**  
    On full Elliptic++ Step 42 (`run_20261126_201353_32960abf`), `T42` shows **`612` Flagged Txs** (blue bar) and **`80` Clusters** (teal bar), while the Risk Distribution shows `6,443` Low, `85` Moderate, and `612` Elevated/High.
11. **What should I say to my project guide about it?**  
    *"These Recharts visualizations load directly from `GET /api/runs/{latestRunId}` and show both risk-band calibration (`>=0.85`, `>=0.57`) and snapshot-by-snapshot cluster counts."*
12. **Common mistakes or things I should avoid:**  
    Avoid confusing the *latest run* charts with the *all-runs* KPI cards above them—the charts intentionally show only the newest run (`runs[0]`).

---

### 4. Dashboard Empty, Error & Recent Analysis History States (`DashboardPage`)

1. **What is this component?**  
   Three operational states on `/`: (a) a clean Empty State card when `0` runs exist, (b) an Error Banner with `Retry` when the API fails, and (c) the Recent Analysis History table listing all saved runs.
2. **Why does FraudLens need it?**  
   First-time users see guidance instead of broken blank boxes or fake numbers, and investigators can revisit any past run anytime.
3. **Where can I find it?**  
   [`DashboardPage.tsx`](../app/frontend/src/pages/DashboardPage.tsx#L128-L160) and lines `323–385`.
4. **What does each button/control do?**  
   - **Upload CSV Bundle** (Empty State CTA): Opens `/upload`.
   - **Retry** (Error Banner): Re-triggers `GET /api/runs`.
   - Per-row **Overview**, **Txs**, **Clusters** buttons: Jump to any past run's pages.
5. **How do I use it step by step?**  
   If no runs exist, click **Upload CSV Bundle**. Once runs exist, locate any row in **Recent Analysis History** and click **Overview**, **Txs**, or **Clusters**.
6. **What information must I enter or select?**  
   Choose a run row from the history table.
7. **What happens after I click it?**  
   Routes cleanly to `/runs/{run_id}`, `/runs/{run_id}/transactions`, or `/runs/{run_id}/clusters`.
8. **What result will appear?**  
   The chosen historical run loads immediately from SQLite `runs`.
9. **How should I understand that result?**  
   Every uploaded dataset gets a unique timestamped `run_id` (e.g., `run_20261126_201353_32960abf`) persisted permanently in `fraud_app.db`.
10. **Real example using existing sample dataset:**  
    During Stage 6E validation, the table listed `stage6e_e2e_audit_step42` (`7,140` txs, `612` flagged, `80` clusters (`24 cand.`), `1.87s` runtime) above older runs.
11. **What should I say to my project guide about it?**  
    *"When the database is empty, FraudLens refuses to fabricate demo numbers and directs users to `/upload`; once runs exist, every batch is permanently indexed in SQLite."*
12. **Common mistakes or things I should avoid:**  
    Do not manually edit `app/data/fraud_app.db` while the backend is running.

---

### 5. `FileDropZone` & CSV Upload Inputs (`UploadPage`)

1. **What is this component?**  
   Interactive drag-and-drop and click-to-browse file cards (`transactions.csv` required, `edges.csv` required, plus collapsible optional `classes.csv`) paired with an optional **Run Label** input and **Required Feature Schema** info card.
2. **Why does FraudLens need it?**  
   The BiDirGCN GNN needs both node features (`182` numbers per tx) and directed graph links (`txId1 -> txId2`), so users must supply both files cleanly before inference.
3. **Where can I find it?**  
   `/upload` ([`UploadPage.tsx`](../app/frontend/src/pages/UploadPage.tsx#L41-L163) & lines `266–367`).
4. **What does each button/control do?**  
   - **Run Label input (`#dataset-name-input`):** Optional friendly name (trimmed & capped at 120 chars; defaults to transactions filename).
   - **Dropzone card (Click / Drag-and-Drop / Enter-Space key):** Accepts `.csv` or `.txt` files (`> 0 B` and `<= 100 MB`). Shows warning banner on wrong extensions (`.json`, `.xlsx`, etc.) or empty files.
   - **Replace / Remove buttons:** Appears once a file is loaded; swaps or clears the file and resets prior validation.
   - **`Optional File 3: Separate Ground-Truth classes.csv` `<details>`:** Expands the optional third dropzone.
5. **How do I use it step by step?**  
   Optionally type a Run Label -> Drag `transactions_sample.csv` into **transactions.csv** -> Drag `edges_sample.csv` into **edges.csv** -> (Optional) Expand File 3 and drop `classes_optional_sample.csv`.
6. **What information must I enter or select?**  
   Two valid CSV/TXT files matching the Elliptic++ raw column format (`txId`, `Time step`, `182` features; `txId1`, `txId2`).
7. **What happens after I click it?**  
   Each zone turns teal (`ready` border) showing exact filename and formatted size (`KB`/`MB`), and any previous validation summary is cleared automatically.
8. **What result will appear?**  
   A green-teal card checkmark with filename + byte size, plus the hint *"Validate the CSV bundle first to enable frozen model inference."*
9. **How should I understand that result?**  
   Client-side file readiness only checked extension and size (`<= 100 MB`); column names and edge integrity still require clicking Validate.
10. **Real example using existing sample dataset:**  
    Selecting `app/sample_data/transactions_sample.csv` + `edges_sample.csv` + `classes_optional_sample.csv` shows `13` rows / `12` edges ready for validation.
11. **What should I say to my project guide about it?**  
    *"Our upload UI enforces dual-CSV input because Graph Neural Networks need both transaction attributes and directed payment edges, while keeping optional ground-truth labels strictly isolated from inference."*
12. **Common mistakes or things I should avoid:**  
    Do not swap `transactions.csv` and `edges.csv` into the wrong zones, and do not upload `.xlsx` Excel workbooks directly—export to UTF-8 `.csv` first.

---

### 6. CSV Validation & Two-Step Analysis Gate (`UploadPage`)

1. **What is this component?**  
   The two-button control bar (`1. Validate CSV Bundle`, `2. Run Fraud Analysis`), upload progress bar, Validation Passed card, and Backend Rejected (`HTTP 422`) diagnostic card.
2. **Why does FraudLens need it?**  
   Silently dropping malformed edges or wrong columns corrupts graph inference. FraudLens forces strict pre-flight validation (`POST /api/validate-csv`) before unlocking inference (`POST /api/analyze`) and locks out duplicate clicks (`submitLockRef`).
3. **Where can I find it?**  
   Bottom half of `/upload` ([`UploadPage.tsx`](../app/frontend/src/pages/UploadPage.tsx#L369-L519)).
4. **What does each button/control do?**  
   - **`1. Validate CSV Bundle`:** Sends multipart files to `POST /api/validate-csv` with upload `%` progress.
   - **`2. Run Fraud Analysis`:** Disabled until `validation.valid === true`; sends files + Run Label to `POST /api/analyze`, refreshes dashboard runs, and navigates to `/runs/{run_id}`.
5. **How do I use it step by step?**  
   Click **1. Validate CSV Bundle** -> Wait for the green `Validation Passed` card -> Read any yellow warnings -> Click **2. Run Fraud Analysis**.
6. **What information must I enter or select?**  
   Both required CSVs must be loaded first.
7. **What happens after I click it?**  
   During validation or inference, both buttons disable (`Running Frozen BiDirGCN Inference...`), the progress bar tracks HTTP upload `%`, and `submitLockRef` ignores accidental rapid double-clicks.
8. **What result will appear?**  
   - **On Error (`422`):** Red error card showing exact error code (`dangling_edges`, `cross_timestep_edges`, `missing_transaction_columns`, `non_numeric_features`, `duplicate_tx_id`), message, and sample offending column/edge IDs.
   - **On Valid (`200`):** Green `VALID` card with `Validated Transactions`, `Unique Directed Edges`, `Timesteps Covered`, `Median-Imputed NaNs`, `Optional Labels` counts, and yellow warnings (e.g., duplicate edge dedup).
   - **After Analyze (`200`):** Redirects automatically to `/runs/{run_id}`.
9. **How should I understand that result?**  
   `VALID` confirms zero dangling/cross-timestep edges and exact `182`-feature ordering matching `processed/metadata.json`.
10. **Real example using existing sample dataset:**  
    Uploading `transactions_sample.csv` + `edges_sample.csv` + `classes_optional_sample.csv` shows `13` Validated Transactions, `12` Unique Directed Edges, Timestep `42`, `0` NaNs, `12 Illicit / 0 Licit / 1 Unk` optional labels, and warning *"Optional class column detected: retained strictly for post-hoc display and NEVER used in inference."* Clicking **2. Run Fraud Analysis** navigates straight to `/runs/<new_run_id>`.
11. **What should I say to my project guide about it?**  
    *"We enforce zero silent edge dropping: if even one edge references a missing transaction ID or crosses timesteps, the API returns HTTP 422 with the exact offending IDs instead of corrupting the graph."*
12. **Common mistakes or things I should avoid:**  
    Do not refresh the browser tab mid-analysis—wait 1–2 seconds for automatic redirection to `/runs/:runId`.

---

### 7. Run Overview / Run Details (`RunDetailsPage`)

1. **What is this component?**  
   The dedicated summary page for one completed analysis run (`/runs/:runId`), displaying KPIs, Risk Distribution chart, Pipeline Metadata, optional Post-Hoc Evaluation, and the Per-Timestep Snapshot table.
2. **Why does FraudLens need it?**  
   Users need to audit individual batch runs (warnings, NaN imputation count, runtime, optional label metrics, and per-snapshot fraud rates) separately from global dashboard totals.
3. **Where can I find it?**  
   `/runs/:runId` ([`RunDetailsPage.tsx`](../app/frontend/src/pages/RunDetailsPage.tsx)).
4. **What does each button/control do?**  
   - **← Back to Dashboard:** Returns to `/`.
   - **Explore Transactions:** Opens `/runs/:runId/transactions`.
   - **Inspect Suspicious Clusters:** Opens `/runs/:runId/clusters`.
   - **Retry:** Re-fetches `GET /api/runs/{run_id}` if an error occurred.
5. **How do I use it step by step?**  
   Open automatically after CSV analysis or via Dashboard history -> Review the 4 KPI cards, warnings, and Risk Distribution chart -> Scroll to **Per-Timestep Snapshot Summary** -> Click **Explore Transactions** or **Inspect Suspicious Clusters**.
6. **What information must I enter or select?**  
   None (`runId` comes from the URL).
7. **What happens after I click it?**  
   Navigating to Transactions or Clusters preserves the active `runId` context.
8. **What result will appear?**  
   Displays `Transactions Analysed`, `Flagged Transactions`, `Suspicious Clusters`, `Shared Unflagged Bridges`, compliance disclaimer, Risk Distribution BarChart (`Mean P` & `Max P`), frozen `Model ID`, exact `Preprocessing Pipeline` order, `Snapshot Tensor Cache` badge (`Available` or `Expired / Evicted`), optional post-hoc `TP/FP/FN/TN/Prec/Rec/F1` box (only if labels were provided), and per-timestep table.
9. **How should I understand that result?**  
   Pay special attention to **Shared Unflagged Bridges** (low-probability nodes connecting multiple suspicious nodes) and **Post-Hoc Optional Label Comparison** (benchmarking only—labels never touched predictions).
10. **Real example using existing sample dataset:**  
    On full Step 42 (`stage6e_e2e_audit_step42`), Run Details shows `7,140` nodes, `8,493` edges, `612` flagged (`8.57%`), `80` clusters (`24` Tier-A), `52` shared bridges, `255` NaN-imputed cells warning, and `Mean P=0.1007 / Max P=0.9984`.
11. **What should I say to my project guide about it?**  
    *"Run Details documents full provenance for this batch: the frozen `Stage 2 -> RankGauss -> BiDirGCN` pipeline order, threshold `0.57`, NaN imputation count, and snapshot cache availability for Explainable AI."*
12. **Common mistakes or things I should avoid:**  
    If `Snapshot Tensor Cache` shows `Expired / Evicted` (`>50` runs or `>30` days old), you can still view tables and previously cached explanations, but computing *brand-new* IG explanations requires re-uploading the CSV.

---

### 8. Transaction Risk Table (`TransactionsPage`)

1. **What is this component?**  
   A paginated, filterable, sortable table listing transactions scored in the selected run from SQLite (`GET /api/runs/{run_id}/transactions`).
2. **Why does FraudLens need it?**  
   A single snapshot has thousands of transactions; analysts need a structured risk queue sorted from highest probability (`pred_prob DESC`) down to prioritize review.
3. **Where can I find it?**  
   `/runs/:runId/transactions` ([`TransactionsPage.tsx`](../app/frontend/src/pages/TransactionsPage.tsx#L264-L345)).
4. **What does each button/control do?**  
   - **Transaction ID** / **Timestep** / **Fraud Probability** / **In/Out Deg** column header buttons: Toggle ascending/descending sort on the visible page.
   - **Cluster link (`C_T42_001`):** Opens that transaction's cluster network graph.
   - **Detail button:** Opens `/runs/:runId/transactions/:txId` to view details and run XAI.
5. **How do I use it step by step?**  
   Review top high-risk rows (backend sorts highest probability first by default), optionally click **Cluster ID** (`C_T42_001`) or click **Detail** on any row.
6. **What information must I enter or select?**  
   None required to view default top 50 rows; use filters/search above the table to narrow down.
7. **What happens after I click it?**  
   **Detail** opens `TransactionDetailPage`; a **Cluster** link opens `ClusterDetailPage`.
8. **What result will appear?**  
   Each row shows: `Transaction ID`, `Timestep`, visual `ProbBar` + `pred_prob` (4 decimals), `Predicted Class` (`Illicit`/`Licit`), `Risk Category` badge, `In/Out Deg`, `Cluster` link (or `—` if singleton), `Optional Label` (`Illicit`/`Licit`/`Unknown`), and `Detail` button.
9. **How should I understand that result?**  
   Probabilities are fetched directly from the backend—the browser performs zero probability math. `—` in Cluster simply means a flagged transaction has no flagged neighbor (`singleton`) or was not flagged.
10. **Real example using existing sample dataset:**  
    In `app/sample_data/`, `txId = 73000655` appears near the top with `pred_prob = 0.9983`, `Predicted Class = Illicit`, `High Predicted Risk (P >= 0.85)`, cluster `C_T42_001`, `Optional Label = Illicit`, whereas `73282397` shows `12 / 0` degree, `Cluster = C_T42_001`, and `Optional Label = Unknown`.
11. **What should I say to my project guide about it?**  
    *"The explorer queries parameterized SQLite indexes on the backend, keeps `Unknown (-1)` ground-truth labels cleanly separated from `Licit`, and links transactions directly to their suspicious cluster or XAI view."*
12. **Common mistakes or things I should avoid:**  
    Clicking column sort headers sorts the *currently loaded page* (`25/50/100` rows), whereas the default backend query already orders all rows globally by highest `pred_prob` descending.

---

### 9. Transaction Search (`TransactionsPage`)

1. **What is this component?**  
   A numeric transaction ID search form (`Search numeric txId...` text input + `Search` button + `Clear` button + inline non-numeric validation warning).
2. **Why does FraudLens need it?**  
   Lets investigators jump straight to a specific Bitcoin transaction ID inside thousands of rows.
3. **Where can I find it?**  
   Top-left of the filter bar on `/runs/:runId/transactions` ([`TransactionsPage.tsx`](../app/frontend/src/pages/TransactionsPage.tsx#L147-L188)).
4. **What does each button/control do?**  
   - **Input (`Search numeric txId...`):** Accepts numeric ID or substring (matches `CAST(tx_id AS TEXT) LIKE '%...%'` on backend).
   - **Search (`submit`):** Validates digits-only (`^\d+$`); if valid, resets `offset=0` and queries `/transactions?search_tx_id=...`; if non-numeric (e.g. `"abc"`), shows a yellow warning banner without sending a misleading query.
   - **Clear:** Resets input, applied filter, notice banner, and `offset=0`.
5. **How do I use it step by step?**  
   Type `73282397` -> Click **Search** -> Inspect matching row(s) -> Click **Clear** to return to full list.
6. **What information must I enter or select?**  
   Digits only (integer transaction ID or ID fragment).
7. **What happens after I click it?**  
   Backend SQLite filters matching rows and updates `Matching: X transaction(s)`.
8. **What result will appear?**  
   Matching transaction rows, `"No transactions match..."` empty state, or `"Transaction ID search expects digits only..."` warning if letters were typed.
9. **How should I understand that result?**  
   Supports substring ID matching (searching `7328` matches `73282397`).
10. **Real example using existing sample dataset:**  
    Typing `73282397` and clicking **Search** narrows `612` or `7,140` Step 42 rows directly down to `Matching: 1 transaction(s)` (`73282397`).
11. **What should I say to my project guide about it?**  
    *"The search box validates numeric input on the client and triggers a parameterized SQL `LIKE` lookup on the FastAPI backend rather than loading 7,000+ rows into browser memory."*
12. **Common mistakes or things I should avoid:**  
    Avoid typing letters or `txId:` prefix into the box—enter digits only.

---

### 10. Risk-Level Filters (`TransactionsPage`)

1. **What is this component?**  
   A dropdown selector (`aria-label="Filter by risk category"`) filtering the table by backend prediction threshold or minimum probability.
2. **Why does FraudLens need it?**  
   Analysts rarely review thousands of low-risk `P < 0.35` transactions; filtering immediately isolates alert-grade rows (`>= 0.57`) or highest-confidence alerts (`>= 0.85`).
3. **Where can I find it?**  
   Filter bar next to Search on `/runs/:runId/transactions` ([`TransactionsPage.tsx`](../app/frontend/src/pages/TransactionsPage.tsx#L189-L202)).
4. **What does each button/control do?**  
   - `All Risk Levels`: `min_prob=0`, `flagged_only=false`.
   - `Flagged Illicit (>= 0.57)`: Sends `flagged_only=true` (`pred_class = 1`).
   - `High Predicted Risk (>= 0.85)`: Sends `min_prob=0.85`.
   - `Moderate or Higher (>= 0.35)`: Sends `min_prob=0.35`.
5. **How do I use it step by step?**  
   Open the dropdown -> Select **Flagged Illicit (>= 0.57)** -> Observe `Matching:` count and table rows update automatically.
6. **What information must I enter or select?**  
   One of the 4 dropdown choices.
7. **What happens after I click it?**  
   Resets page offset to `0` and queries `GET /api/runs/{run_id}/transactions` with the selected backend query parameter.
8. **What result will appear?**  
   Only rows meeting that probability/flag constraint load in the table.
9. **How should I understand that result?**  
   Selecting `Flagged Illicit (>= 0.57)` gives the exact model alert count matching KPI `flagged_count`.
10. **Real example using existing sample dataset:**  
    On full Step 42 (`7,140` total), switching to **Flagged Illicit (>= 0.57)** immediately updates `Matching:` from `7,140` to **`612 transaction(s)`**.
11. **What should I say to my project guide about it?**  
    *"The risk filter pushes `flagged_only=true` or `min_prob` directly to the FastAPI SQLite query so totals and pagination remain exact."*
12. **Common mistakes or things I should avoid:**  
    Do not forget you left a risk filter active when wondering why low-risk nodes are hidden.

---

### 11. Timestep & Page-Size Filters (`TransactionsPage`)

1. **What is this component?**  
   Two dropdowns (`Filter by timestep` and `Rows per page`) in the Transaction Risk Table toolbar.
2. **Why does FraudLens need it?**  
   Multi-timestep CSV uploads can span multiple weeks (`Time step`); timestep filtering isolates one snapshot at a time, while page-size controls viewing density (`25`/`50`/`100`).
3. **Where can I find it?**  
   Filter bar on `/runs/:runId/transactions` ([`TransactionsPage.tsx`](../app/frontend/src/pages/TransactionsPage.tsx#L204-L234)).
4. **What does each button/control do?**  
   - **Timestep dropdown:** Lists `All Timesteps` plus each uploaded step from `runDetail.summary.timesteps`.
   - **Rows per page dropdown:** Sets `limit` (`25`, `50`, or `100`) and resets `offset=0`.
5. **How do I use it step by step?**  
   Select a specific timestep (e.g., `Timestep 42`) or change `50 / page` to `100 / page`.
6. **What information must I enter or select?**  
   Any available timestep or page size option.
7. **What happens after I click it?**  
   Re-queries `GET /api/runs/{run_id}/transactions` with updated `timestep` / `limit` and `offset=0`.
8. **What result will appear?**  
   Table rows, `Matching:` count, and `Page X of Y` update cleanly.
9. **How should I understand that result?**  
   Every edge and prediction in FraudLens operates strictly within one timestep, so filtering by timestep mirrors the exact snapshot evaluated by BiDirGCN.
10. **Real example using existing sample dataset:**  
    On `app/sample_data/`, the timestep dropdown shows `All Timesteps` and `Timestep 42`.
11. **What should I say to my project guide about it?**  
    *"Because temporal snapshots in Elliptic++ have zero cross-timestep edges, filtering by timestep lets an analyst audit one discrete graph snapshot cleanly."*
12. **Common mistakes or things I should avoid:**  
    Combine carefully with `Search`—if you search a Step 42 `txId` while `Timestep 35` is selected, `0` rows match.

---

### 12. Pagination & Column Sorting (`TransactionsPage`)

1. **What is this component?**  
   The bottom **Previous / Next** pagination bar (`Page X of Y`) and sortable column header buttons (`Transaction ID`, `Timestep`, `Fraud Probability`, `In/Out Deg`).
2. **Why does FraudLens need it?**  
   Rendering 7,000+ DOM table rows at once slows browsers down; server-backed `limit/offset` pagination keeps scrolling snappy.
3. **Where can I find it?**  
   Column headers and bottom footer of `/runs/:runId/transactions` ([`TransactionsPage.tsx`](../app/frontend/src/pages/TransactionsPage.tsx#L268-L290) & `347–378`).
4. **What does each button/control do?**  
   - **Column Header Buttons:** Re-sort currently fetched `items` client-side (`asc` <-> `desc`).
   - **Previous (`offset - limit`) / Next (`offset + limit`):** Request previous/next server page, disabled cleanly at bounds or while loading.
5. **How do I use it step by step?**  
   Click **Next** to advance pages, or click **In/Out Deg** header to see highest-degree transactions on the current page.
6. **What information must I enter or select?**  
   Click buttons directly.
7. **What happens after I click it?**  
   Pagination triggers a backend SQLite `LIMIT ? OFFSET ?` query; sorting reorders the current page array in memory.
8. **What result will appear?**  
   Updated rows and updated `Page X of Y` label.
9. **How should I understand that result?**  
   With `limit=50` on `612` flagged Step 42 rows, `Page 1 of 13` advances to `Page 2 of 13`, etc.
10. **Real example using existing sample dataset:**  
    On `app/sample_data/` (`13` rows), `Page 1 of 1` disables both `Previous` and `Next`.
11. **What should I say to my project guide about it?**  
    *"Pagination is server-driven via SQLite `LIMIT/OFFSET` so even a 250,000-row upload loads smoothly at 50 rows per page."*
12. **Common mistakes or things I should avoid:**  
    Remember column header sorting sorts the *visible page* only, whereas backend SQLite handles global probability ranking.

---

### 13. Shared Fraud Probability Bar & Risk / Ground-Truth Badges (`RiskBadge.tsx`)

1. **What is this component?**  
   Three reusable UI visual components (`ProbBar`, `RiskBadge`, `GtBadge`) shared across Transactions, Transaction Details, Cluster Graph, and Node Tables.
2. **Why does FraudLens need it?**  
   Keeps probability colors, threshold bands, and the visual separation of predicted risk vs. optional ground truth consistent across every page.
3. **Where can I find it?**  
   [`RiskBadge.tsx`](../app/frontend/src/components/RiskBadge.tsx) (used in `TransactionsPage`, `TransactionDetailPage`, `ClusterDetailPage`).
4. **What does each button/control do?**  
   Display-only indicators:
   - `ProbBar`: Horizontal bar (`0%–100%`) + 4-decimal probability (`0.9947`), colored by band.
   - `RiskBadge`: Displays backend `risk_category` (`High Predicted Risk (P >= 0.85)` red, `Elevated Risk (Flagged >= 0.57)` orange, `Moderate Monitor (0.35 - 0.57)` yellow, `Low Predicted Risk (< 0.35)` green).
   - `GtBadge`: Displays `Illicit` (red), `Licit` (green), or `Unknown` (neutral slate).
5. **How do I use it step by step?**  
   Read `RiskBadge` for model alert band, `ProbBar` for exact probability, and `GtBadge` (when labels exist) for historical truth.
6. **What information must I enter or select?**  
   None.
7. **What happens after I click it?**  
   Display-only.
8. **What result will appear?**  
   Consistent color-coded visual badges.
9. **How should I understand that result?**  
   Never confuse **`GtBadge: Unknown`** (no historical label published) with **`Licit`**! Unknown nodes can still be strongly flagged by the model (`ProbBar >= 0.57`).
10. **Real example using existing sample dataset:**  
    For `txId = 73282397` in `app/sample_data/`, `ProbBar` shows `0.9940` (red bar), `RiskBadge` shows `High Predicted Risk (P >= 0.85)`, and `GtBadge` shows `Unknown`.
11. **What should I say to my project guide about it?**  
    *"We render predicted risk and ground-truth labels in separate badges and strictly distinguish `Unknown` from `Licit`, preserving Elliptic++ evaluation hygiene."*
12. **Common mistakes or things I should avoid:**  
    Do not read a green `ProbBar` (`< 0.35`) as proof that `GtBadge` is `Licit`—it only means the *model* predicted low risk.

---

### 14. Suspicious Clusters Explorer (`ClustersPage`)

1. **What is this component?**  
   The cluster list view displaying all connected flagged transaction subgraphs (`|C| >= 2`) discovered in a run, with Tier-A Candidate & Timestep filters plus an explicit investigative caution banner.
2. **Why does FraudLens need it?**  
   Individual transaction alerts can miss coordinated laundering structures (peeling chains, fan-in hubs, dispersion rings). Grouping directly connected flagged nodes exposes collective structure.
3. **Where can I find it?**  
   `/runs/:runId/clusters` ([`ClustersPage.tsx`](../app/frontend/src/pages/ClustersPage.tsx)).
4. **What does each button/control do?**  
   - **Potential Fraud-Ring Candidates Only (Tier A)** checkbox: Queries `GET /api/runs/{run_id}/clusters?candidates_only=true`.
   - **Timestep dropdown:** Filters clusters by snapshot timestep.
   - **Open Cluster button:** Opens `/runs/:runId/clusters/:clusterId`.
   - **Explore Transactions / Back to Run Details / Retry:** Navigation & error recovery.
5. **How do I use it step by step?**  
   Open **Suspicious Clusters** -> Read the yellow Caution banner -> Check **Potential Fraud-Ring Candidates Only (Tier A)** to prioritize top rings -> Click **Open Cluster** on `C_T42_001`.
6. **What information must I enter or select?**  
   Optional Tier-A checkbox and Timestep dropdown.
7. **What happens after I click it?**  
   Filters reload from SQLite `clusters` sorted by `is_ring_candidate DESC, ring_score DESC, num_nodes DESC`.
8. **What result will appear?**  
   Table showing `Cluster ID`, `Step`, `Nodes / Edges`, `Risk Summary` (`Mean`, `Max`, `Ring Score`, `>=0.85` count), `Directed Structure` (`motif_type`, internal in/out max degrees, external 1-hop neighbor counts), `Candidate Status` (`Ring Candidate` red badge or tier), `Hub Tx`, and **Open Cluster** button.
9. **How should I understand that result?**  
   Look for high **Ring Score** + **Ring Candidate** badge + clear directed motif (`fan_in_consolidation`, `complex_mesh`, `peeling_chain`, `fan_out_dispersion`). These are prioritized triage suspects, **not** legally proven criminal rings.
10. **Real example using existing sample dataset:**  
    On `app/sample_data/`, `C_T42_001` appears as `13 nodes / 12 edges`, `Mean 0.998 • Max 0.998`, `Ring Score: 7.31`, motif `fan_in_consolidation`, badge `Ring Candidate`, `Hub Tx: 73282397`. On full Step 42, `80` clusters appear, filtering down to **`24`** Tier-A candidates when checked.
11. **What should I say to my project guide about it?**  
    *"FraudLens adapts Stage 5C subgraph discovery: connected components of flagged nodes are ranked label-free using `RingScore` and classified by directed flow motif without using ground-truth labels."*
12. **Common mistakes or things I should avoid:**  
    Never tell an audience that `Ring Candidate` means "confirmed criminal gang"—always call it a model-identified suspicious group requiring compliance investigation.

---

### 15. Cluster Details Header, Context Toggle & Caution Banner (`ClusterDetailPage`)

1. **What is this component?**  
   The top header strip, context visibility controls, and warning banner on an individual cluster's graph page (`/runs/:runId/clusters/:clusterId`).
2. **Why does FraudLens need it?**  
   Frames exact cluster statistics (timestep, motif, ring score, core vs 1-hop context count) and lets the analyst switch cleanly between inspecting *only* the suspicious core vs seeing external 1-hop neighbors.
3. **Where can I find it?**  
   Top of `/runs/:runId/clusters/:clusterId` ([`ClusterDetailPage.tsx`](../app/frontend/src/pages/ClusterDetailPage.tsx#L323-L380)).
4. **What does each button/control do?**  
   - **Show 1-Hop Context (`N`) checkbox:** Toggles visibility of external boundary nodes (`is_external_context=true`) and dashed boundary edges (`external_boundary_in`/`out`) in both the Cytoscape graph and tables below.
   - **Graph layout selector (`<select>`):** Switches Cytoscape arrangement between **Directed Flow Layout** (`breadthfirst`), **Force-Directed Layout** (`cose`), and **Circular Layout** (`circle`).
   - **← Back to Suspicious Clusters:** Returns to `/runs/:runId/clusters`.
5. **How do I use it step by step?**  
   Start in **Directed Flow Layout** with **Show 1-Hop Context** checked -> Read motif & Ring Score -> Uncheck **Show 1-Hop Context** if you want to focus strictly on flagged core nodes.
6. **What information must I enter or select?**  
   Checkbox state and layout choice.
7. **What happens after I click it?**  
   `visibleNodes` and `visibleEdges` recompute instantly in memory, re-building the Cytoscape view and node/edge tables cleanly without re-fetching from the backend.
8. **What result will appear?**  
   The Cytoscape canvas and bottom tables immediately update node/edge counts (`Core members + 1-hop context`).
9. **How should I understand that result?**  
   External 1-hop context nodes are *adjacent transactions* in the same timestep that fell below `0.57` (or bridge into/out of the cluster)—seeing them reveals how money entered or exited the suspicious core!
10. **Real example using existing sample dataset:**  
    On full Step 42 `C_T42_001`, header shows `38 core nodes, 37 internal edges, +20 1-hop context node(s)` (`Show 1-Hop Context (20)`). Unchecking the toggle cleanly hides the 20 slate dashed boundary nodes, leaving only the 38 orange/red flagged nodes.
11. **What should I say to my project guide about it?**  
    *"The context toggle separates core flagged ring members from adjacent below-threshold boundary transactions returned by `GET /clusters/{id}/graph`, without needing extra API calls."*
12. **Common mistakes or things I should avoid:**  
    Do not assume a cluster looks isolated just because **Show 1-Hop Context** is unchecked—re-enable it to see external boundary connections.

---

### 16. Interactive Cytoscape Graph, Toolbar Controls & Legend (`ClusterDetailPage`)

1. **What is this component?**  
   The live **Cytoscape.js** directed network canvas (`height: 390px`), 4-button zoom/fit/reset toolbar, and 6-item color/edge **Visual Legend**.
2. **Why does FraudLens need it?**  
   Humans spot money-flow patterns (fan-in convergence, peeling chains, hub bottlenecks) in seconds visually—something raw CSV edge lists cannot convey.
3. **Where can I find it?**  
   Left card of `/runs/:runId/clusters/:clusterId` ([`ClusterDetailPage.tsx`](../app/frontend/src/pages/ClusterDetailPage.tsx#L383-L491)).
4. **What does each button/control do?**  
   - **Zoom In (`+`) / Zoom Out (`−`):** Scales Cytoscape zoom by `1.25×` / `0.8×` around canvas center.
   - **Fit Graph to Screen (`⛶`):** Calls `cy.fit(undefined, 28)` so the whole cluster fits inside view.
   - **Reset Graph View (`↺`):** Re-runs the active directed layout algorithm and refits the viewport.
   - **Mouse Wheel / Drag Canvas / Tap Node:** Standard pan/zoom (`minZoom: 0.25`, `maxZoom: 3.0`) and click-to-select a node.
5. **How do I use it step by step?**  
   Look at the Visual Legend -> Zoom in with `+` or scroll wheel -> Click any red/orange circle (especially the teal-ringed **Cluster Hub**) -> Click `Fit to screen` if you zoom too far.
6. **What information must I enter or select?**  
   Click toolbar controls or tap a node on the canvas.
7. **What happens after I click it?**  
   Cytoscape updates zoom/position immediately; tapping a node triggers Cytoscape's `'tap'` listener, highlights the node with a bright border, and updates the **Selected Transaction Node** panel on the right. On navigation away, Cytoscape removes all listeners and calls `cy.destroy()` to prevent memory leaks.
8. **What result will appear?**  
   Directed arrows (`target-arrow-shape: triangle`, orange solid for internal suspicious links, slate dashed for 1-hop external links) connect color-coded transaction nodes labeled by `txId`.
9. **How should I understand that result?**  
   - **Red dot:** `P >= 0.85` (High Predicted Risk)
   - **Orange dot:** `0.57 <= P < 0.85` (Flagged)
   - **Teal outer ring + largest circle:** **Cluster Hub** (highest internal degree inside cluster)
   - **Slate dashed small circle:** **1-Hop Context** node outside core cluster
10. **Real example using existing sample dataset:**  
    Opening `C_T42_001` (`app/sample_data/` or full Step 42) in `Directed Flow Layout` clearly draws all incoming arrows converging downward/inward into the large teal-ringed Hub node **`73282397`**.
11. **What should I say to my project guide about it?**  
    *"We render only the selected cluster subgraph via Cytoscape.js, preserving exact directed payment arrows (`src -> dst`) and cleaning up the Cytoscape instance on unmount so navigation stays leak-free."*
12. **Common mistakes or things I should avoid:**  
    Avoid reading the graph as an *undirected* web—always follow where the triangle arrowheads point!

---

### 17. Node Selection, Inspector Panel, Subgraph Nodes & Directed Edges Tables (`ClusterDetailPage`)

1. **What is this component?**  
   The synchronized **Selected Transaction Node** inspector card (top right) plus two full-width tables below: **Subgraph Nodes** and **Directed Edges** (`src_tx_id -> dst_tx_id`).
2. **Why does FraudLens need it?**  
   Small labels on crowded graphs can be hard to click or read on mobile; clicking a table row guarantees exact node selection, while the Directed Edges table lets analysts audit exact edge types and endpoint probabilities.
3. **Where can I find it?**  
   Right card and bottom full-width cards on `/runs/:runId/clusters/:clusterId` ([`ClusterDetailPage.tsx`](../app/frontend/src/pages/ClusterDetailPage.tsx#L493-L692)).
4. **What does each button/control do?**  
   - **Subgraph Nodes Table Row (`onClick`):** Highlights that row in blue, selects the matching node inside Cytoscape, and updates the **Selected Transaction Node** card.
   - **Inspect & Explain Transaction (IG)** (or **Open Transaction Detail** for external context nodes): Navigates to `/runs/:runId/transactions/{selectedNode.tx_id}`.
5. **How do I use it step by step?**  
   Either click a node on the Cytoscape canvas OR click a row in **Subgraph Nodes** -> Read the inspector card (`Role`, `Probability`, `Degree`, `Logit`, `Optional GT`, `Connected Subgraph Flows`) -> Click **Inspect & Explain Transaction (IG)**.
6. **What information must I enter or select?**  
   Select any node row or canvas circle.
7. **What happens after I click it?**  
   `selectedNode` updates synchronously, showing exact deterministic cluster role (`Cluster Hub`, `Cluster Source` [0 internal in, >=1 out], `Cluster Sink` [>=1 internal in, 0 out], `Internal Relay / Member`, or `1-Hop External Boundary`).
8. **What result will appear?**  
   Full node details + incoming/outgoing subgraph edge counts + full directed edges table (`Directed Flow`, `Edge Type`, `Src Prob`, `Dst Prob`).
9. **How should I understand that result?**  
   Notice how `Internal Hub` / `Cluster Sink` transactions receive converging funds from multiple `Cluster Source` nodes.
10. **Real example using existing sample dataset:**  
    Clicking **`73282397`** shows `Cluster Hub`, `Probability 0.9947`, `Illicit`, `Snapshot Degree 25 / 0`, `Raw Model Logit 5.23492`, `Optional GT Unknown`, and `Connected Subgraph Flows: 25 incoming edge(s), 0 outgoing edge(s)`.
11. **What should I say to my project guide about it?**  
    *"The node table and Cytoscape graph stay two-way synchronized; structural roles like `Cluster Source`, `Sink`, and `Hub` are derived directly from the returned directed edges without inventing metrics."*
12. **Common mistakes or things I should avoid:**  
    Note that external 1-hop context nodes display `Raw Model Logit: N/A (Context Node)` because `routes.py` only sends full SQLite logits for internal cluster rows—yet clicking `Open Transaction Detail` still loads its full SQLite record!

---

### 18. Transaction Risk Summary (`TransactionDetailPage`)

1. **What is this component?**  
   The top summary card on `/runs/:runId/transactions/:txId` presenting the target transaction's frozen prediction alongside its optional historical ground-truth label.
2. **Why does FraudLens need it?**  
   Before explaining *why* a prediction happened, analysts must clearly see *what* the model predicted vs. what historical ground truth (if any) was attached.
3. **Where can I find it?**  
   Top card of `/runs/:runId/transactions/:txId` ([`TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx#L195-L262)).
4. **What does each button/control do?**  
   - **View Cluster {cluster_id}** (Header button) & **Cluster Association link:** Jumps to the cluster graph if the transaction belongs to a cluster.
   - **← Back to Transaction Risk Table:** Returns to `/runs/:runId/transactions`.
5. **How do I use it step by step?**  
   Inspect `Predicted Fraud Probability`, `Raw Logit Score`, `Model Predicted Class`, `Actual Ground-Truth Class (Optional)`, `Directed Degree (In / Out)`, and `Cluster Association`.
6. **What information must I enter or select?**  
   None (`runId` and `txId` come from the URL).
7. **What happens after I click it?**  
   Header/cluster links navigate to the cluster view or transaction list.
8. **What result will appear?**  
   Clear 6-field grid explicitly labeling `Illicit (Predicted)` separately from `Unknown` / `Illicit` / `Licit`.
9. **How should I understand that result?**  
   A positive raw logit (`> 0`) corresponds to probability `> 0.50`; specifically `logit >= ~0.2819` crosses frozen threshold `0.57` (`sigmoid(0.2819) = 0.57`).
10. **Real example using existing sample dataset:**  
    For **`73282397`**, Risk Summary shows `Predicted Fraud Probability = 0.9947`, `Raw Logit Score = 5.234924`, `Model Predicted Class = Illicit (Predicted)`, `Actual Ground-Truth Class = Unknown`, `Directed Degree = 25 / 0`, `Cluster Association = C_T42_001`.
11. **What should I say to my project guide about it?**  
    *"Notice we explicitly separate `Model Predicted Class` from `Actual Ground-Truth Class`—for `73282397`, ground truth is `Unknown`, yet the BiDirGCN flags it with `99.47%` probability because of its features and converging neighbors."*
12. **Common mistakes or things I should avoid:**  
    Do not skip this separation when presenting—your project guide will specifically look to see that `Unknown` labels are not treated as `Licit` or leaked into predictions.

---

### 19. On-Demand Transaction Explainability Trigger & UX States (`TransactionDetailPage`)

1. **What is this component?**  
   The on-demand XAI control flow: **Explain Prediction (IG)** / **Refresh Explanation** header button, empty prompt card with CTA button, animated loading spinner card, and retryable error card.
2. **Why does FraudLens need it?**  
   Running 200-step Integrated Gradients (`M=200`) across 7,140 nodes on upload would take hours! Computing explanations **strictly on demand** when an analyst clicks **Explain** and caching them in SQLite makes the app both fast and faithful.
3. **Where can I find it?**  
   Header and central panel of `/runs/:runId/transactions/:txId` ([`TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx#L177-L191) & `264–313`).
4. **What does each button/control do?**  
   - **Explain Prediction (IG):** Fires `POST /api/runs/{run_id}/transactions/{tx_id}/explain` (protected by `explainInFlightRef` so double-clicking cannot fire two gradient jobs).
   - **Refresh Explanation:** Appears once loaded; re-queries the endpoint (which hits SQLite cache immediately).
   - **Retry Explanation:** Appears inside the red error card if the backend fails (e.g., `409 snapshot_cache_expired` if snapshot `.pt` tensor cache was pruned).
5. **How do I use it step by step?**  
   Open a transaction -> Confirm empty prompt `On-Demand Feature Attribution Ready` -> Click **Explain Prediction (IG)** -> Wait during spinner -> Inspect results -> (Optional) Click **Refresh Explanation** to verify instant cache retrieval.
6. **What information must I enter or select?**  
   Simply click the **Explain Prediction (IG)** button.
7. **What happens after I click it?**  
   Backend loads `step_{t}.pt` tensor cache, executes 200 forward+backward passes through frozen `BiDirRelationalGCN`, decomposes attributions across features and nodes, stores the JSON payload in SQLite `explanations`, and returns it (`cache_hit: false` first time, `cache_hit: true` thereafter).
8. **What result will appear?**  
   Badge transitions from `Freshly Computed` (teal) on initial calculation (~0.3s on 13-node sample, ~28s on 7,140-node snapshot) to `Cached SQLite Result` (green, ~0.02s) on repeat clicks.
9. **How should I understand that result?**  
   Waiting ~28s on a 7,140-node snapshot is normal on CPU because `M=200` composite midpoint quadrature executes 200 gradient passes; SQLite caching ensures you only ever wait once per `(run_id, tx_id)`.
10. **Real example using existing sample dataset:**  
    On full Step 42 `73282397`, first click computed fresh `M=200` IG in `28.38s` (`Freshly Computed`), and second click (`Refresh Explanation`) returned in `0.0230s` (`Cached SQLite Result`).
11. **What should I say to my project guide about it?**  
    *"Explainability is strictly on-demand rather than computed eagerly during upload, guarded against duplicate clicks, and persisted in SQLite so repeated inspections return in ~20 milliseconds."*
12. **Common mistakes or things I should avoid:**  
    Do not spam-click or close the tab during the 20–30s cold calculation on a large snapshot—wait for the loading spinner to finish.

---

### 20. Integrated Gradients Feature Contribution Chart & Tables (`TransactionDetailPage`)

1. **What is this component?**  
   A horizontal Recharts `BarChart` (red positive bars right of `0`, green negative bars left of `0`) paired with two detailed 10-row tables: **Top Positive Contributors** and **Top Negative Contributors**.
2. **Why does FraudLens need it?**  
   Shows *which exact Elliptic++ features* (out of 182) drove this transaction's fraud score up vs. pulled it down.
3. **Where can I find it?**  
   First major card after clicking **Explain Prediction (IG)** on `/runs/:runId/transactions/:txId` ([`TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx#L321-L447)).
4. **What does each button/control do?**  
   Hovering any horizontal bar displays the exact 6-decimal `IG Logit Contribution`; tables display all 10 top positive and 10 top negative features cleanly with `Group` badges (`Local93`, `Aggregate72`, `Domain17`) and `Self / Nbr` split.
5. **How do I use it step by step?**  
   Scan the red bars (top positive drivers) and green bars (top negative drivers), then read exact `Graph IG = Self + Nbr` values in the two tables below.
6. **What information must I enter or select?**  
   None (hover bars for tooltips).
7. **What happens after I click it?**  
   Display/hover only.
8. **What result will appear?**  
   Up to 6 top positive + 6 top negative features plotted in the chart, plus full 10 + 10 tables.
9. **How should I understand that result?**  
   - **Positive (`+`, red):** Feature pushed the predicted logit higher (more suspicious).
   - **Negative (`-`, green):** Feature pushed the logit lower (more legitimate).
   - **`Self / Nbr` column:** Shows how much of that feature's attribution came from the transaction's *own* feature value (`Self`) vs. its *neighbors'* values via GCN message passing (`Nbr`).
10. **Real example using existing sample dataset:**  
    For `73282397`, top positive driver is **`Local_feature_3`** (`+5.23532`: `Self = +4.06241`, `Nbr = +1.17291`), followed by `Local_feature_14` (`+4.77030`); top negative counterweight is **`Aggregate_feature_10`** (`-6.70889`: `Self = -2.27231`, `Nbr = -4.43657`).
11. **What should I say to my project guide about it?**  
    *"The horizontal bar chart clearly separates positive risk drivers (red) from negative counter-evidence (green), and the `Self / Nbr` split proves our BiDirGCN relies on both transaction attributes and graph context."*
12. **Common mistakes or things I should avoid:**  
    Do not read a negative green bar as an "error"—complex neural models always balance positive fraud evidence against negative licit signals; the *net sum* determines the final logit (`5.23492`).

---

### 21. Feature Group Contributions & Self-vs-Neighbor Shares (`TransactionDetailPage`)

1. **What is this component?**  
   The **Feature Group Attribution Summary** card breaking all 182 features into **Local Features (`93`)**, **Aggregate Features (`72`)**, and **Domain Features (`17`)**, alongside **Self vs. Neighbor** attribution `%` shares inside the diagnostics box.
2. **Why does FraudLens need it?**  
   Individual feature names (`Local_feature_3`) are anonymized in Elliptic++, whereas feature **groups** tell you *what type* of evidence dominated (transaction-level local signals vs 1-hop aggregates vs interpretable Bitcoin BTC/fee domain metrics).
3. **Where can I find it?**  
   Left card of the second row after explanation loads ([`TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx#L450-L501)).
4. **What does each button/control do?**  
   Display-only breakdown cards showing `Signed` sum, `|IG| Magnitude`, and `Self-node signed sum`.
5. **How do I use it step by step?**  
   Compare the 3 group rows (`Local Features (93)`, `Aggregate Features (72)`, `Domain Features (17)`) and read the `Self vs Neighbor` share sentence in the adjacent card.
6. **What information must I enter or select?**  
   None.
7. **What happens after I click it?**  
   Display only.
8. **What result will appear?**  
   Three group cards with red (`>= 0`) or green (`< 0`) signed sums plus absolute `|IG|` magnitudes.
9. **How should I understand that result?**  
   A group can have huge `|IG| Magnitude` even if its net `Signed` sum is smaller (when positive and negative features within that group partly cancel).
10. **Real example using existing sample dataset:**  
    For `73282397`: **Local Features (93)** = **`+17.01454`** (`|IG| 50.44140`), **Aggregate Features (72)** = **`-7.88440`** (`|IG| 66.65639`), **Domain Features (17)** = **`+0.42152`** (`|IG| 13.29288`). Self-features contribute **`74.6%`** and graph neighbors contribute **`25.4%`**.
11. **What should I say to my project guide about it?**  
    *"Because Elliptic++ anonymizes `Local_feature_1..93` and `Aggregate_feature_1..72`, grouping attributions into Local93, Aggregate72, and Domain17 tells analysts whether intrinsic transaction properties or Bitcoin flow metrics drove the alert."*
12. **Common mistakes or things I should avoid:**  
    Do not say `Local Features` has zero neighbor contribution—neighbors *also* carry their own `Local_feature_*` values which propagate into the target during GNN convolution (`Self-node signed sum: 11.71862` vs total `+17.01454`)!

---

### 22. Explanation Quality & Completeness Diagnostics (`TransactionDetailPage`)

1. **What is this component?**  
   The **Explanation Quality & Completeness** card showing numerical verification metrics (`Method`, `Integration Steps`, `Completeness Residual`, `Relative Error`, `Logit Δ`, `Total IG Attribution Sum`, `All Finite` badge, and plain-English explanation).
2. **Why does FraudLens need it?**  
   XAI explanations are useless if numerically inaccurate. Stage 5B identified high errors with naive `M=80` grids and validated the 3-Zone Composite Midpoint `M=200` grid (`< 0.05` abs / `< 1.0%` rel error)—this card proves accuracy live for every clicked transaction.
3. **Where can I find it?**  
   Right card of the second row after explanation loads ([`TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx#L503-L568)).
4. **What does each button/control do?**  
   Display-only audit box.
5. **How do I use it step by step?**  
   Check that the green **`All Finite`** badge appears, check **`Relative Error` (< `1.0%`)**, and verify `Total IG Attribution Sum` matches `Logit Δ (x - base)` closely.
6. **What information must I enter or select?**  
   None.
7. **What happens after I click it?**  
   Display only.
8. **What result will appear?**  
   Exact numerical values computed from the 200-step backward pass.
9. **How should I understand that result?**  
   According to the Integrated Gradients **Completeness Axiom**, summing every feature's IG across the snapshot should equal `logit(tx) - logit(baseline)`. Near-zero residual proves the explanation faithfully accounts for the prediction.
10. **Real example using existing sample dataset:**  
    On `73282397`: `Method = Composite_Midpoint_IG_3Zone_M200`, `Integration Steps = M = 200`, `Logit Δ = 9.56030`, `Total IG Sum = 9.55166`, **`Completeness Residual = 0.008639 logits`**, **`Relative Error = 0.0904%`**, badge **`All Finite`**.
11. **What should I say to my project guide about it?**  
    *"Every explanation includes a live mathematical self-check: our Stage 5B 3-zone Composite Midpoint `M=200` quadrature keeps completeness error under `0.1%` (`0.0086` logits on `73282397`) with zero non-finite gradients."*
12. **Common mistakes or things I should avoid:**  
    Note that `Logit Δ` (`9.56030`) equals `logit_x (5.23492) - logit_baseline (-4.32538)`, which differs from raw `logit_x (5.23492)` alone because the zero/median baseline already has a strong negative licit baseline logit!

---

### 23. Directed 1-Hop Neighborhood Influence & 2-Hop Summary (`TransactionDetailPage`)

1. **What is this component?**  
   The bottom table listing every direct incoming (`neighbor -> tx`) and outgoing (`tx -> neighbor`) 1-hop neighbor in the same timestep along with its predicted probability, optional GT label, and signed/`|IG|` logit influence—plus a 2-hop cohort header summary.
2. **Why does FraudLens need it?**  
   In a Graph Neural Network, a transaction's prediction depends on its neighbors' features! This table identifies *which specific connected transactions* pushed the target node toward Fraud or Licit.
3. **Where can I find it?**  
   Bottom card of `/runs/:runId/transactions/:txId` after explanation loads ([`TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx#L571-L637)).
4. **What does each button/control do?**  
   Display-only audit table (sorted by largest absolute `|IG| Logit` influence descending); shows an empty message (`0 incoming/outgoing edges`) if the node is isolated.
5. **How do I use it step by step?**  
   Read `Direction` (`incoming` vs `outgoing`), `Directed Edge`, `Neighbor Prob`, `Neighbor GT`, `Signed IG Logit`, and `|IG| Logit` (magnitude of influence).
6. **What information must I enter or select?**  
   None.
7. **What happens after I click it?**  
   Display only.
8. **What result will appear?**  
   All unique `(direction, neighbor)` edges + 2-hop summary (`count`, `pred_illicit_count`, `mean_pred_prob`).
9. **How should I understand that result?**  
   Because `BiDirRelationalGCN` has **2 graph convolution layers**, information travels up to **2 hops**. Even a neighbor with negative `Signed IG` can have large `|IG| Logit` (showing strong structural sensitivity).
10. **Real example using existing sample dataset:**  
    For `73282397`, the table lists **25 `incoming` edges** (all feeding into `73282397`, such as `73000840 -> 73282397`, `Neighbor Prob = 0.9978`, `Signed IG = +0.01035`, `|IG| = 3.57147`) plus `2-hop snapshot context: 23 node(s), 8 predicted illicit (mean P=0.413)`.
11. **What should I say to my project guide about it?**  
    *"Since our BiDirGCN uses separate incoming and outgoing directed message-passing matrices over 2 layers, this panel exposes exact 1-hop directional attribution and 2-hop risk context."*
12. **Common mistakes or things I should avoid:**  
    Do not overlook **`|IG| Logit`**: a neighbor can have positive + negative feature attributions that nearly cancel in `Signed IG Logit`, yet still strongly sway specific features (`|IG| Logit = 3.57`).

---

### 24. Root Router, Suspense Fallback & API Error Normalization (`App.tsx`, `fraudApi.ts`)

1. **What is this component?**  
   The top-level React Router setup with `React.lazy()` code-splitting (`<Suspense fallback={<RouteLoadingFallback />}>`) and centralized Axios client `fraudApi.ts` (`BackendRequestError`).
2. **Why does FraudLens need it?**  
   Keeps initial bundle fast (`13.05 kB` entry JS, deferring `Cytoscape` `451 kB` and `Recharts` `395 kB` until needed) and ensures any HTTP error (`422`/`404`/`409`/`network_error`) shows a clean human-readable message instead of crashing React.
3. **Where can I find it?**  
   [`App.tsx`](../app/frontend/src/App.tsx) and [`fraudApi.ts`](../app/frontend/src/api/fraudApi.ts).
4. **What does each button/control do?**  
   Automatically redirects unknown URLs (`*`) back to `/`, displays `"Loading view..."` briefly while chunks load, and formats backend `detail.message` payloads cleanly.
5. **How can I use it / what does it show?**  
   Works silently behind the scenes across every route.

---

## PART B — COMPLETE 12-STEP USER WORKFLOW

1. **Start the application:** Double-click `start_fraudlens.bat` in the `Capstone_Fraud_Detection` project root (or run `start_backend.bat` and `start_frontend.bat` in two terminals). Wait until both windows are ready, then open `http://127.0.0.1:5173`.
2. **Open the Dashboard (`/`):** Look at the top-right status pill and verify it says **`API Online • Threshold 0.57`**. Review existing runs (or empty state).
3. **Upload transaction and edge CSV files (`/upload`):** Click **Upload CSV Bundle** in the left sidebar (or **New CSV Analysis** button). Type a Run Label (optional, e.g., `demo_step42_sample`). Drag `app/sample_data/transactions_sample.csv` into **transactions.csv** and `app/sample_data/edges_sample.csv` into **edges.csv** (optionally expand **File 3** and drop `classes_optional_sample.csv`).
4. **Validate the files (`/upload`):** Click **`1. Validate CSV Bundle`**. Confirm the green **`Validation Passed — Ready for Fraud Analysis`** card appears (`13` transactions, `12` unique edges, Timestep `42`).
5. **Start an analysis (`/upload` -> `/runs/:runId`):** Click the newly unlocked **`2. Run Fraud Analysis`** button. Wait during inference; FraudLens automatically navigates to `/runs/{new_run_id}`.
6. **Open the generated run (`/runs/:runId`):** Inspect the 4 KPI cards (`Transactions Analysed`, `Flagged Transactions`, `Suspicious Clusters`, `Shared Unflagged Bridges`), Risk Distribution chart, warnings, and Per-Timestep Snapshot table.
7. **Explore transactions (`/runs/:runId/transactions`):** Click **`Explore Transactions`**. Review rows sorted from highest `Fraud Probability` downward.
8. **Filter and search for high-risk transactions (`/runs/:runId/transactions`):** Choose **`Flagged Illicit (>= 0.57)`** or **`High Predicted Risk (>= 0.85)`** in the risk dropdown, or type digits (like `73282397`) in **`Search numeric txId...`** and click **Search**. Click **Clear** when finished.
9. **Open transaction details (`/runs/:runId/transactions/:txId`):** Click **`Detail`** on the target row (`73282397`) to examine its predicted score vs. optional ground-truth label.
10. **Investigate suspicious clusters (`/runs/:runId/clusters`):** Click **Suspicious Clusters** in the sidebar (or **View Cluster C_T42_001** button on the transaction page). Notice the caution banner, optionally check **`Potential Fraud-Ring Candidates Only (Tier A)`**, and click **`Open Cluster`** on `C_T42_001`.
11. **Explore the graph (`/runs/:runId/clusters/:clusterId`):** Inspect the directed Cytoscape network (`source -> target`), toggle **Show 1-Hop Context**, try **Force-Directed Layout** vs **Directed Flow Layout**, use **Zoom in / Fit / Reset**, and click the teal-ringed hub node `73282397` (or its row in **Subgraph Nodes**) to verify its `Cluster Hub` role and incoming links.
12. **Generate and understand an explanation (`/runs/:runId/transactions/:txId`):** Click **`Inspect & Explain Transaction (IG)`** in the node inspector, then click **`Explain Prediction (IG)`**. Read the red/green horizontal bar chart, **Local93 / Aggregate72 / Domain17** group sums, **Completeness Residual** (`< 1%`), and **Directed 1-Hop Neighborhood Influence** table.

---

## PART C — LIMITATIONS AND HONEST WARNINGS

- **What FraudLens CAN Do:** Validate raw 182-feature transaction CSVs and directed edge lists without silent edge drops; run deterministic frozen `Stage 2 -> RankGauss -> BiDirGCN` inference; cluster connected flagged transactions (`>= 0.57`) into directed subgraphs & Tier-A candidates; compute & cache faithful `M=200` Integrated Gradients attributions; and display optional post-hoc ground-truth metrics separately.
- **What FraudLens CANNOT Do:** Monitor live real-time blockchain mempools (`streaming_supported: false`); prove criminal intent or guilt; link cross-timestep multi-week chains when uploaded CSVs lack cross-timestep edges; run multi-user logins/RBAC (single-analyst local workstation mode); or retrain/fine-tune the neural network (`0.57` threshold is frozen).
- **Live vs. Batch Data:** Strictly **offline batch CSV upload**.
- **Do Predictions Prove Actual Fraud?** **NO.** Flagged transactions (`P >= 0.57`) and Tier-A `Ring Candidates` are probabilistic risk prioritization alerts for human compliance analysts—never confirmed criminal fraud.
- **Label Availability:** Optional in CSV uploads. In Elliptic++, `77.15%` of transactions are `Unknown (-1)` (`class = 3`), and FraudLens preserves `Unknown` separately from `Licit`.
- **Frozen Model & Threshold:** `ExpA2_RankGauss182_BiDirGCN` (`models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt`), threshold **`0.57`**. Under severe post-Step-42 darknet market shutdown drift (`Steps 43–49`), model recall drops significantly compared to `Step 42` (`stage4h_report.md` & `stage5c_fraud_ring_report.md`).
- **Performance & Cache Boundaries:** Cold `M=200` IG on a `7,140`-node CPU snapshot takes `~28s` (`~23ms` once cached in SQLite); snapshot `.pt` tensor caches auto-prune after `50` runs or `30` days (`HTTP 409` on expired runs for uncached IG requests).
- **Testing Status:** Full 12-step Chrome Playwright E2E passed on dual-CSV uploads (`Step 42`), while optional `classes.csv` multipart UI isolation was verified in backend `pytest` (`test_optional_class_isolation_from_inference`) and sample CSV validation (`verify_stage6f_packaging.py`) rather than inside Playwright UI.
