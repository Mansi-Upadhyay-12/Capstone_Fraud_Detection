# FraudLens Complete Frontend Inventory

**Application:** **FraudLens — Intelligent Transaction Risk Analysis** (*Graph-Based Fraud Detection & Explainable AI*)  
**Frontend Stack:** React 18 (`react-router-dom` v6), TypeScript, Vite 6, Recharts 2, Cytoscape.js 3, Axios, Lucide React  
**Backend API Base:** `http://127.0.0.1:8010` (via `VITE_API_BASE_URL` in [`app/frontend/src/api/fraudApi.ts`](../app/frontend/src/api/fraudApi.ts))

---

## 1. Summary Counts (Observed Directly From Source Files)

- **Pages / Routes Inspected (`7` routes + `1` wildcard redirect):**
  1. `/` — `DashboardPage`
  2. `/upload` — `UploadPage`
  3. `/runs/:runId` — `RunDetailsPage`
  4. `/runs/:runId/transactions` — `TransactionsPage`
  5. `/runs/:runId/transactions/:txId` — `TransactionDetailPage`
  6. `/runs/:runId/clusters` — `ClustersPage`
  7. `/runs/:runId/clusters/:clusterId` — `ClusterDetailPage`
  *(Unknown paths `*` redirect to `/` via `<Navigate to="/" replace />` in [`App.tsx`](../app/frontend/src/App.tsx).)*
- **Source Modules & UI Components Inspected (`14` runtime files, `25` documented component groups/blocks):**
  - App Shell & Shared Components: `App`, `RouteLoadingFallback`, `AppLayout` (Sidebar, Topbar, Status Pill, Mobile Drawer + Backdrop), `RiskBadge`, `ProbBar`, `GtBadge`.
  - Page Feature Sections & Helper Sub-components: Dashboard KPIs, Empty/Error States, Latest Run Summary, Risk Band Chart, Per-Timestep Chart, Recent History Table, `FileDropZone` (`transactions.csv`, `edges.csv`, `classes.csv`), Upload Validation/Analysis Gating & Progress Bar, Run Details KPIs/Metadata/Post-Hoc Eval/Per-Timestep Table, Transaction Search/Filters/Sort/Pagination/Table, Suspicious Clusters Filters/Banner/Table, Cluster Cytoscape Canvas/Toolbar/Legend/Node Inspector/Nodes Table/Edges Table, Transaction Risk Summary/Explain Trigger/Feature Contribution Chart & Tables/Feature Group Breakdown/Completeness Diagnostics/1-Hop & 2-Hop Neighborhood Context.
- **Interactive Controls Inspected (`65` distinct UI controls across layout and pages):**
  - Navigation Links (`16` static/template route links + dynamic per-row links), Buttons (`22`), File/Text/Checkbox Inputs (`9`), Dropdown `<select>` Controls (`5`), Expandable `<details>` Section (`1`), Interactive Canvas & Clickable Rows (`12` click/drag/drop handlers across dropzones, Cytoscape canvas, node rows, and mobile drawer backdrop).

---

## 2. Page-by-Page Routing & Backend API Inventory

### Page 1: Dashboard (`DashboardPage`)
- **Route / URL:** `/`
- **Source File:** [`app/frontend/src/pages/DashboardPage.tsx`](../app/frontend/src/pages/DashboardPage.tsx)
- **Purpose:** Gives the user an immediate bird's-eye overview of all previously completed CSV analysis runs, plus visual charts for the latest run.
- **Main Components:** Batch Offline Notice Banner, 4 KPI Summary Cards, Latest Analysis Summary Card, Risk Band Distribution BarChart (Recharts), Per-Timestep Flagged & Cluster BarChart (Recharts), Recent Analysis History Table, Empty State Card (when `0` runs exist), and Error Retry Banner.
- **Available User Actions:**
  - Click **New CSV Analysis** (`/upload`).
  - Click **Upload CSV Bundle** in the empty state (`/upload`).
  - Click **Retry** if loading runs fails.
  - Click **Run Details**, **Transactions**, or **Clusters** on the Latest Analysis Summary card.
  - Click **Overview**, **Txs**, or **Clusters** on any row in Recent Analysis History.
- **Backend API Endpoints Used:**
  - `GET /api/health` (via `App.tsx`)
  - `GET /api/runs` (via `App.tsx`)
  - `GET /api/runs/{latestRunId}` (via `fetchRunDetail` for chart data)
- **Information Displayed:** Cumulative analysed transaction count, cumulative flagged transaction count (`>= 0.57`), cumulative suspicious cluster & Tier-A candidate counts, total run count, latest latency, latest run ID/dataset/timestamp/edges/labels status, Risk Band counts, per-timestep flagged & cluster counts, and full run history table.
- **What the User Should Do Here:** Confirm **API Online • Threshold 0.57** in the top bar, review past runs, or click **New CSV Analysis** to upload and score a new CSV dataset.

---

### Page 2: Upload CSV Bundle (`UploadPage`)
- **Route / URL:** `/upload`
- **Source File:** [`app/frontend/src/pages/UploadPage.tsx`](../app/frontend/src/pages/UploadPage.tsx)
- **Purpose:** Provides a safe two-step workflow (`1. Validate` -> `2. Run Fraud Analysis`) so malformed, missing-column, dangling-edge, or cross-timestep CSV files are caught before frozen BiDirGCN inference runs.
- **Main Components:** Optional Class Isolation Info Banner, Run Label text input (`#dataset-name-input`), Two Required `FileDropZone` cards (`transactions.csv`, `edges.csv`), Collapsible Optional `classes.csv` `FileDropZone`, Schema Guidance Card, Step Buttons (`1. Validate CSV Bundle`, `2. Run Fraud Analysis`), Upload Progress Track, Client Notice Warning Banner, Structured Backend Error Card, and Validation Passed Summary Card.
- **Available User Actions:**
  - Enter an optional Run Label (`dataset_name`).
  - Drag-and-drop or browse for `transactions.csv`, `edges.csv`, and optionally `classes.csv`.
  - **Replace** or **Remove** any selected file.
  - Click **1. Validate CSV Bundle** (enabled when both required CSVs are selected).
  - Click **2. Run Fraud Analysis** (enabled ONLY when `validation.valid === true`).
- **Backend API Endpoints Used:**
  - `POST /api/validate-csv`
  - `POST /api/analyze`
- **Information Displayed:** Selected filenames and sizes, client file-extension/empty/100MB warnings, real-time upload progress `%`, HTTP `422`/`413` error codes/messages/offending column or edge samples, validated node/edge/timestep/NaN-imputation counts, optional class counts, and deduplication warnings.
- **What the User Should Do Here:** Select both required CSVs, click **1. Validate CSV Bundle**, read any warnings or fix any `HTTP 422` errors, then click **2. Run Fraud Analysis** and wait to be redirected automatically to `/runs/:runId`.

---

### Page 3: Run Overview / Run Details (`RunDetailsPage`)
- **Route / URL:** `/runs/:runId`
- **Source File:** [`app/frontend/src/pages/RunDetailsPage.tsx`](../app/frontend/src/pages/RunDetailsPage.tsx)
- **Purpose:** Summarizes everything produced by a single analysis run and acts as the launchpad into its transactions and suspicious clusters.
- **Main Components:** Back link, Page Action Buttons (`Explore Transactions`, `Inspect Suspicious Clusters`), 4 KPI Cards, Compliance Disclaimer Banner, Warning Banners (if NaNs imputed or duplicate edges removed), Risk Distribution Recharts BarChart, Execution & Pipeline Metadata Card, Post-Hoc Optional Label Comparison Box (only shown when optional labels exist), and Per-Timestep Snapshot Summary Table.
- **Available User Actions:**
  - Click **Back to Dashboard** (`/`).
  - Click **Explore Transactions** (`/runs/:runId/transactions`).
  - Click **Inspect Suspicious Clusters** (`/runs/:runId/clusters`).
  - Click **Retry** on API failure.
- **Backend API Endpoints Used:**
  - `GET /api/runs/{run_id}`
- **Information Displayed:** `dataset_name`, `run_id`, timestamp, total nodes/edges, flagged count & rate `%` at threshold `0.57`, direct cluster & Tier-A candidate counts, shared unflagged bridges count, inference latency, mean/max predicted probability, `model_id`, preprocessing order, uploaded timesteps, NaN imputation count, snapshot cache availability (`Available` / `Expired / Evicted`), optional post-hoc `TP/FP/FN/TN/Precision/Recall/F1`, and per-timestep table.
- **What the User Should Do Here:** Verify dataset counts and warnings, see whether any clusters were detected, and click **Explore Transactions** or **Inspect Suspicious Clusters**.

---

### Page 4: Transaction Risk Table (`TransactionsPage`)
- **Route / URL:** `/runs/:runId/transactions`
- **Source File:** [`app/frontend/src/pages/TransactionsPage.tsx`](../app/frontend/src/pages/TransactionsPage.tsx)
- **Purpose:** Lets users search, filter, sort, and paginate through all scored transactions in a run to prioritize suspicious items for deep investigation.
- **Main Components:** Search Bar Form (`input`, `Search`, `Clear`), Risk Category Dropdown, Timestep Dropdown, Rows-Per-Page Dropdown, Matching Count Readout, Non-Numeric Search Warning Banner, Error Retry Banner, Sortable Data Table (`ProbBar`, `RiskBadge`, `GtBadge`), and Server Pagination Bar (`Previous`, `Next`).
- **Available User Actions:**
  - Search by exact/partial numeric `txId` and click **Search** or **Clear**.
  - Filter by Risk Level (`All Risk Levels`, `Flagged Illicit (>= 0.57)`, `High Predicted Risk (>= 0.85)`, `Moderate or Higher (>= 0.35)`).
  - Filter by `Timestep`.
  - Change page size (`25`, `50`, `100` rows per page).
  - Sort currently loaded table rows by `Transaction ID`, `Timestep`, `Fraud Probability`, or `In/Out Deg`.
  - Click a `Cluster ID` link to open the cluster's graph (`/runs/:runId/clusters/:clusterId`).
  - Click **Detail** on any row to inspect the transaction and run Explainable AI (`/runs/:runId/transactions/:txId`).
  - Page backward/forward with **Previous** / **Next**.
- **Backend API Endpoints Used:**
  - `GET /api/runs/{run_id}` (for available timesteps list)
  - `GET /api/runs/{run_id}/transactions` (with `timestep`, `flagged_only`, `min_prob`, `search_tx_id`, `limit`, `offset`)
- **Information Displayed:** `tx_id`, `timestep`, `pred_prob` (`ProbBar`), `pred_label` (`Illicit`/`Licit`), `risk_category`, `in_deg / out_deg`, clickable `cluster_id` (or `—`), `optional_gt_label` (`Illicit`/`Licit`/`Unknown`), and total matched rows.
- **What the User Should Do Here:** Filter to `Flagged Illicit (>= 0.57)` or search a `txId`, click **Detail** on a high-risk row for XAI, or click its **Cluster ID** link to inspect how it connects to neighbors.

---

### Page 5: Transaction Details & XAI (`TransactionDetailPage`)
- **Route / URL:** `/runs/:runId/transactions/:txId`
- **Source File:** [`app/frontend/src/pages/TransactionDetailPage.tsx`](../app/frontend/src/pages/TransactionDetailPage.tsx)
- **Purpose:** Shows an individual transaction's prediction vs optional ground truth, and runs **on-demand Integrated Gradients (`M=200`)** to explain *why* the BiDirGCN model assigned its score.
- **Main Components:** Header Actions (`Back to Transaction Risk Table`, `View Cluster`, `Explain Prediction (IG)` / `Refresh Explanation`), Transaction Risk Summary Card, On-Demand Empty Prompt Card, Explanation Loading Spinner Card, Error + Retry Card, Disclaimer Banner, Feature Contribution Horizontal BarChart (`Recharts`) + Positive/Negative Feature Tables, Feature Group Attribution Summary (`Local93`, `Aggregate72`, `Domain17`), Explanation Quality & Completeness Card (with Self-vs-Neighbor shares), and Directed 1-Hop Neighborhood Influence Table (+ 2-Hop Summary).
- **Available User Actions:**
  - Click **Explain Prediction (IG)** (from header or empty prompt) to compute explanation on demand.
  - Click **Refresh Explanation** to fetch/confirm SQLite cached result.
  - Click **Retry Explanation** if an error occurs (e.g., `HTTP 409` expired tensor cache).
  - Click **View Cluster** or inline `cluster_id` to inspect the cluster graph.
  - Click **Back to Transaction Risk Table**.
- **Backend API Endpoints Used:**
  - `GET /api/runs/{run_id}/transactions?search_tx_id={txId}&limit=50`
  - `POST /api/runs/{run_id}/transactions/{tx_id}/explain`
- **Information Displayed:** Probability, raw logit, predicted class, optional GT class, degree, cluster link, disclaimer, SQLite cache status (`Freshly Computed` vs `Cached SQLite Result`), top 10 positive & top 10 negative features (graph/self/neighbor IG breakdown), Local/Aggregate/Domain signed & `|IG|` totals, integration method `Composite_Midpoint_IG_3Zone_M200`, `M=200`, completeness residual (`abs_error`), relative error `%`, logit delta, `All Finite` badge, self vs neighbor attribution `%` shares, directed 1-hop neighbors table, and 2-hop summary.
- **What the User Should Do Here:** First examine the Transaction Risk Summary (comparing predicted risk against optional ground truth), then click **Explain Prediction (IG)**, wait ~0.1–30s depending on snapshot size/cache, and inspect which features and neighbors pushed the fraud logit upward or downward.

---

### Page 6: Model-Identified Suspicious Clusters (`ClustersPage`)
- **Route / URL:** `/runs/:runId/clusters`
- **Source File:** [`app/frontend/src/pages/ClustersPage.tsx`](../app/frontend/src/pages/ClustersPage.tsx)
- **Purpose:** Lists all connected groups of `>= 2` flagged transactions discovered in the run, ranked by candidacy and composite ring score.
- **Main Components:** Header & `Explore Transactions` link, Investigative Caution Warning Banner, Filters Bar (`Potential Fraud-Ring Candidates Only (Tier A)` checkbox, `Timestep` dropdown, count readout), Error Retry Banner, Empty State Card, and Clusters Summary Table.
- **Available User Actions:**
  - Check/uncheck **Potential Fraud-Ring Candidates Only (Tier A)**.
  - Select a **Timestep** filter.
  - Click **Retry** on error.
  - Click **Open Cluster** on any cluster row to launch the Cytoscape network graph.
- **Backend API Endpoints Used:**
  - `GET /api/runs/{run_id}` (for timesteps dropdown)
  - `GET /api/runs/{run_id}/clusters` (with `candidates_only` and `timestep`)
- **Information Displayed:** `cluster_id`, `timestep`, `num_nodes / num_internal_edges`, `mean_pred_prob`, `max_pred_prob`, `ring_score`, high-confidence node count (`>= 0.85`), `motif_type` (`fan_in_consolidation`, `fan_out_dispersion`, `peeling_chain`, `directed_pair`, `complex_mesh`), internal & external 1-hop degree stats, candidacy badge (`Ring Candidate` or tier), and `hub_tx_id`.
- **What the User Should Do Here:** Read the caution banner, filter to Tier-A Ring Candidates if analyzing large snapshots, pick a cluster with high `Ring Score` or interesting motif, and click **Open Cluster**.

---

### Page 7: Cluster Directed Network Graph & Details (`ClusterDetailPage`)
- **Route / URL:** `/runs/:runId/clusters/:clusterId`
- **Source File:** [`app/frontend/src/pages/ClusterDetailPage.tsx`](../app/frontend/src/pages/ClusterDetailPage.tsx)
- **Purpose:** Visualizes an individual suspicious cluster's directed payment topology (`source -> target`) using **Cytoscape.js**, shows optional 1-hop context neighbors, and lets users inspect any node or jump straight to its XAI explanation.
- **Main Components:** Header Badges & Stats, Show 1-Hop Context Checkbox, Graph Layout Dropdown, Model-Identified Subgraph Caution Banner, Cytoscape Interactive Canvas, Graph Zoom/Fit/Reset Toolbar, Graph Visual Legend, Selected Transaction Node Inspector Panel, Subgraph Nodes Table, and Directed Edges Table.
- **Available User Actions:**
  - Toggle **Show 1-Hop Context** to include/hide adjacent non-cluster neighbor nodes and dashed boundary edges.
  - Switch graph layout (`Directed Flow Layout`, `Force-Directed Layout`, `Circular Layout`).
  - Pan (drag canvas), wheel-zoom, or use toolbar buttons (**Zoom in**, **Zoom out**, **Fit graph to screen**, **Reset graph view**).
  - Click any node in the Cytoscape graph OR click any row in the **Subgraph Nodes** table to inspect it.
  - Click **Inspect & Explain Transaction (IG)** (or **Open Transaction Detail** on context nodes) to open `/runs/:runId/transactions/:txId`.
- **Backend API Endpoints Used:**
  - `GET /api/runs/{run_id}/clusters/{cluster_id}/graph`
- **Information Displayed:** Cluster ID, candidacy badge, timestep, motif type, ring score, core node & internal edge counts, external 1-hop count, directed Cytoscape canvas, visual legend, selected node `txId`, derived cluster role (`Cluster Hub`, `Cluster Source`, `Cluster Sink`, `Internal Relay / Member`, `1-Hop External Boundary`), `pred_prob`, `RiskBadge`, predicted class, snapshot in/out degree, raw logit (core nodes), optional GT badge, connected incoming/outgoing edge counts in subgraph, full nodes table, and explicit `src_tx_id -> dst_tx_id` directed edges table.
- **What the User Should Do Here:** Trace the direction of arrows (`source -> target`) to identify hubs, sources, and sinks; click nodes to inspect probabilities; and click **Inspect & Explain Transaction (IG)** on the hub or suspicious nodes to understand model reasoning.

---

## 3. Code-Only Support Components Discovered

- `RouteLoadingFallback` ([`App.tsx`](../app/frontend/src/App.tsx#L29-L35)): Shown automatically by React `<Suspense>` while lazy-loaded route bundles download (`"Loading view..."`).
- `FileDropZone` ([`UploadPage.tsx`](../app/frontend/src/pages/UploadPage.tsx#L41-L163)): Reusable accessible drop/click/keyboard zone enforcing `.csv`/`.txt`, non-empty (`> 0 B`), and `<= 100 MB` checks.
- `RiskBadge`, `ProbBar`, `GtBadge` & Formatters ([`RiskBadge.tsx`](../app/frontend/src/components/RiskBadge.tsx)): Shared badges and formatters ensuring consistent colors (`>= 0.85` red, `0.57–0.85` orange, `0.35–0.57` yellow, `< 0.35` green) and strict separation of `Unknown` vs `Licit` vs `Illicit`.
- `BackendRequestError` & `normalizeApiError` ([`fraudApi.ts`](../app/frontend/src/api/fraudApi.ts#L23-L75)): Centralized Axios error translator handling offline network failures, Pydantic/FastAPI `HTTP 422` structured error details, `413` size limits, `404` missing resources, and `409` expired tensor caches.
