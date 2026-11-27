# Stage 6E — End-to-End Application Validation & User Experience Audit Report

**Date:** 2026-11-27  
**Application:** **FraudLens — Intelligent Transaction Risk Analysis** (*Graph-Based Fraud Detection & Explainable AI*)  
**Audit Scope:** Full user-journey verification (`12` workflow steps), real headless Google Chrome browser execution via Playwright, responsive audit (`1440x900`, `1366x768`, `768x1024`, `390x844`), basic application-level security/data-handling audit, performance review, bug fixes, and full regression re-execution.

---

## 1. Executive Summary

Stage 6E end-to-end validation is **COMPLETE (`PASS`)**.  
All `8/8` Vitest React component tests, TypeScript + Vite production build (`9.11s`, zero warnings), full `12`-step real headless Chrome Playwright E2E journey on full Elliptic++ Step 42 (`7,140` transactions, `8,493` edges, `80` clusters, `24` candidates, on-demand `M=200` Integrated Gradients), multi-breakpoint responsive checks (`0 px` overflow across all 4 viewports), and `10/10` Stage 6B backend regression + numerical parity tests passed with **0 console errors, 0 uncaught page errors, and 0 HTTP 5xx server failures**.

During the audit, `6` real bugs / UX defects were discovered (`ClusterDetailPage` Cytoscape `useEffect` race with `loading`, external Google Fonts CDN DNS console errors + missing `/favicon.ico` `404`, non-numeric search silent ignore on `TransactionsPage`, Recharts chart render delay during static inspection, clipped two-column node/edge tables on desktop `ClusterDetailPage`, and missing mobile drawer backdrop + synchronous double-submit lock). All `6` were fixed and verified cleanly.

---

## 2. User Workflows Tested

| Workflow Step | Verified Path & Outcome | Status |
| :--- | :--- | :---: |
| **A. Startup & Health** | `uvicorn` (`127.0.0.1:8010`) + `vite preview` (`127.0.0.1:4173`), CORS verified, topbar displays `API Online • Threshold 0.57` | **PASS** |
| **B. CSV Upload & Validation** | Dual file picker + drag-and-drop (`Vitest` & Playwright), wrong file extension blocked client-side, dangling edges trigger `HTTP 422` (`dangling_edges`), valid Step 42 (`7,140` nodes, `8,493` edges) unlocks Analyze only after validation pass | **PASS** |
| **C. Analysis Execution** | `dataset_name = stage6e_e2e_audit_step42` locked against double-submit, executes frozen `Stage 2 -> RankGauss -> BiDirGCN` (`1.10s`), redirects to `/runs/<run_id>` | **PASS** |
| **D. Dashboard & Run Details** | Aggregated KPI totals, Recharts Risk Distribution & Per-Timestep charts, Recent History links, and Vitest `0`-run empty state + error retry | **PASS** |
| **E. Transaction Explorer** | Numeric search validation (`"abc"` blocked with clear notice), risk filter (`Flagged >= 0.57` -> `612` rows), degree & probability sorting, pagination, optional ground-truth separation | **PASS** |
| **F. Suspicious Cluster Explorer** | `80` Step 42 clusters (`24` Tier-A candidates filter), directed motif & ring scores, prominent non-criminal-proof disclaimer | **PASS** |
| **G. Directed Cluster Graph** | Cytoscape live graph (`C_T42_001`: `58` nodes, `60` edges), legend, zoom/fit/reset toolbar, layout switching (`cose`/`breadthfirst`), node selection + deterministic cluster role, safe unmount destroy | **PASS** |
| **H. Transaction-Level XAI** | Strictly on-demand (zero auto-calls), real `Composite_Midpoint_IG_3Zone_M200` on `73282397` (`Local93`, `Aggregate72`, `Domain17`, completeness residual + rel error, 1-hop/2-hop neighbors), SQLite `Cached SQLite Result` on repeat refresh | **PASS** |

---

## 3. Exact Commands Executed (Final Clean Run After All Fixes)

```powershell
# 1. Frontend Vitest component & state tests
npm --prefix app/frontend test

# 2. TypeScript type-check & production Vite bundle
npm --prefix app/frontend run build

# 3. Full real headless Chrome Playwright E2E + 4-viewport responsive + screenshot audit
.\venv\Scripts\python.exe C:\Users\mansi\.gemini\antigravity\brain\8eefce37-2f35-4ee2-bb75-29c88d910ff8\scratch\run_stage6e_e2e_playwright.py

# 4. Backend Stage 6B API, validation, XAI & Step 35/42 numerical parity regression suite
.\venv\Scripts\python.exe -m pytest -v app/backend/tests/test_stage6b_backend.py

# 5. Protected Stage 1-5 git status verification
git status --short app/ src/ models/ processed/ reports/stage5/ reports/stage6/
```

---

## 4. Test Results & Counts

| Suite / Audit Gate | Status | Counts & Timing |
| :--- | :---: | :--- |
| **Frontend Component Tests (`vitest run`)** | **PASS** | **`3` test files, `8/8` tests passed (`0` failed)** in `7.02s` (`ClusterDetailPage`: 3, `TransactionDetailPage`: 2, `DashboardAndUpload`: 3) |
| **TypeScript + Vite Production Build (`tsc && vite build`)** | **PASS** | **`0` TS errors, `✓ 2,066 modules transformed`**, built in `9.11s`, `0` chunk warnings (initial JS `13.05 kB`) |
| **Playwright Headless Chrome E2E (`7,140`-node Step 42)** | **PASS** | **All `7` multi-page checks + `4` viewports passed; `0` console errors, `0` page errors, `0` HTTP 5xx errors** |
| **Backend Pytest Regression (`test_stage6b_backend.py`)** | **PASS** | **`10/10` tests passed (`0` failed)** in `71.41s` (including Step 35 & Step 42 `0.00e+00` raw tensor parity) |
| **Optional `classes.csv` Multipart Upload via Browser UI** | **NOT TESTED** | Core dual-CSV UI path tested in browser; optional `classes` isolation verified directly in backend `pytest` (`test_optional_class_isolation_from_inference`). |
| **Destructive Stress / Multi-User Load Testing** | **NOT TESTED** | Excluded per Stage 6E rules ("Do not run destructive load tests"). |
| **JSDOM Recharts Container & React Router v7 Notices** | **WARNING** | Standard harmless `vitest` jsdom warnings (`width(0)/height(0)` on `ResponsiveContainer` in headless DOM; React Router v7 future flags). Zero runtime impact in real Chrome. |

---

## 5. Browser Screenshots & Evidence Paths (`reports/stage6/evidence/stage6e/`)

All `11` real Chromium screenshots captured cleanly after fixes:
1. [`01_dashboard_desktop.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/01_dashboard_desktop.png) — Populated Dashboard (`API Online • Threshold 0.57`, KPI cards, instant Recharts Risk Band & T42 bars, Recent Analysis History).
2. [`02_upload_validation_error.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/02_upload_validation_error.png) — `HTTP 422 dangling_edges` rejection card with Analyze disabled.
3. [`03_upload_validated.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/03_upload_validated.png) — Valid Step 42 pre-flight check (`7,140` transactions, `8,493` unique edges, `255` NaN cells).
4. [`04_run_details.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/04_run_details.png) — Run Details (`7,140` nodes, `612` flagged, `80` clusters, `52` bridges, Recharts tooltip & Per-Timestep table).
5. [`05_transactions_explorer.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/05_transactions_explorer.png) — Filtered flagged transaction explorer (`612` matching rows).
6. [`06_clusters_explorer.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/06_clusters_explorer.png) — Suspicious Cluster Explorer (`24` filtered Tier-A Ring Candidates + disclaimer).
7. [`07_cluster_directed_graph.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/07_cluster_directed_graph.png) — Cytoscape directed graph (`C_T42_001`: `58` nodes, `60` edges), legend, node inspector (`73282397`), and unclipped full-width Node & Edge tables.
8. [`08_transaction_xai.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/08_transaction_xai.png) — On-demand `M=200` Integrated Gradients attribution waterfall (`Local_feature_3 +5.23532`), group summaries (`Local93`, `Aggregate72`, `Domain17`), completeness residual (`0.008639 logits` / `0.0904%`), and directed 1-hop neighbors.
9. [`09_tablet_768x1024.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/09_tablet_768x1024.png) — Tablet (`768x1024`) layout check.
10. [`10_mobile_390x844.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/10_mobile_390x844.png) — Mobile (`390x844`) stacked KPIs, charts, and responsive topbar.
11. [`11_mobile_drawer_open.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/11_mobile_drawer_open.png) — Mobile navigation drawer open with dimmed click-outside backdrop.

---

## 6. Responsive Audit Results

| Viewport | Page(s) Checked | Horizontal Overflow (`scrollWidth - innerWidth`) | Status |
| :--- | :--- | :---: | :---: |
| **Desktop (`1440x900`)** | All `7` routes + Cluster Graph + XAI | `0 px` | **PASS** |
| **Laptop (`1366x768`)** | Dashboard | `0 px` | **PASS** |
| **Tablet (`768x1024`)** | Dashboard + Sidebar Drawer Collapse | `0 px` | **PASS** |
| **Mobile (`390x844`)** | Dashboard + Drawer + `ClusterDetailPage` + `TransactionDetailPage` | `0 px` across all 3 pages | **PASS** |

---

## 7. Bugs Discovered & 8. Fixes Implemented

1. **External Google Fonts DNS & Missing Favicon Errors (`index.html`):** Headless Chrome reported `5` `ERR_NAME_NOT_RESOLVED` console errors on external `fonts.googleapis.com` links in air-gapped execution plus `404` on `/favicon.ico`.  
   - *Fix:* Removed external Google Fonts CDN `<link>` tags (relying on clean local system sans/mono stacks) and embedded an inline SVG `<link rel="icon">` in [`app/frontend/index.html`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/index.html) -> **Console errors dropped from `6` to `0`**.
2. **Cytoscape Initialization Timing / `loading` Dependency Gap (`ClusterDetailPage.tsx`):** `loading` was updated in `.finally()` while the Cytoscape initialization `useEffect` omitted `loading` from its dependency array, risking skipped canvas mount if React committed `graphData` prior to `loading=false` (e.g., Retry transitions).  
   - *Fix:* Batched `setLoading(false)` inside `.then`/`.catch` and added `loading` to the `useEffect` dependencies in [`ClusterDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClusterDetailPage.tsx#L57-L243).
3. **Side-by-Side Table Clipping in Cluster Detail (`ClusterDetailPage.tsx`):** Placing the `Subgraph Nodes` and `Directed Edges` tables inside `.grid-2` squeezed columns horizontally at `1440px`.  
   - *Fix:* Converted both tables into full-width stacked cards ([`07_cluster_directed_graph.png`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/evidence/stage6e/07_cluster_directed_graph.png)).
4. **Non-Numeric Transaction Search Silent Ignore (`TransactionsPage.tsx`):** Entering non-digit characters in `searchTxId` previously passed no `search_tx_id` parameter, silently returning all transactions.  
   - *Fix:* Added upfront numeric check displaying `"Transaction ID search expects digits only (e.g. 71989375)."` warning in [`TransactionsPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionsPage.tsx#L106-L116).
5. **Recharts Animation Delay Blank Frames (`DashboardPage`, `RunDetailsPage`, `TransactionDetailPage`):** Default 400ms grow animations delayed immediate chart visibility on initial paint and screenshots.  
   - *Fix:* Set `isAnimationActive={false}` on `<Bar>` series for instant crisp chart rendering.
6. **Mobile Topbar Spacing & Slide-Out Drawer Backdrop + Upload Double-Click Lock (`AppLayout.tsx`, `index.css`, `UploadPage.tsx`):** Added `.mobile-backdrop` click-outside dismissal + compact mobile topbar rules in [`index.css`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/styles/index.css#L636-L654) and synchronous `submitLockRef` + dataset label length cap (`120` chars) in [`UploadPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/UploadPage.tsx#L179-L237).

---

## 9. Security & Data-Handling Observations (Application-Level Review)

- **Parameterized SQL & Output Escaping:** All SQLite queries in [`db.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/database/db.py) use `?` placeholders; React JSX renders all fields (`dataset_name`, `run_id`, `cluster_id`) auto-escaped with zero `dangerouslySetInnerHTML`.
- **Path Traversal & File Naming:** Uploaded filenames never map to filesystem paths; cached tensor snapshots resolve strictly via server-generated UUID `run_id` and integer `timestep` only after `db.get_run(run_id)` succeeds (`404` otherwise).
- **Strict CSV & Label Isolation:** Non-numeric features, `±Inf`, duplicate `txId`, dangling edges, and cross-timestep edges trigger structured `HTTP 422`; optional `class` labels never enter `X_raw` inference inputs.
- **Honest Security Scope:** Not a formal penetration test or multi-tenant authenticated deployment; [`routes.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/api/routes.py) reads multipart upload bytes into RAM before checking `MAX_CSV_BYTES` (`100 MB`).

---

## 10. Performance Observations

- **Frontend Bundle Splitting:** Initial entry JS stays at **`13.05 kB` (`4.85 kB` gzip)** via `React.lazy()` and Rollup `manualChunks` (`graph-vendor` `451.17 kB`, `charts-vendor` `395.55 kB`, `react-vendor` `165.01 kB`).
- **End-to-End Inference & XAI Latency:** Full Step 42 batch scoring + clustering (`7,140` nodes, `8,493` edges) executes in `~1.1–1.9s` on CPU; fresh `M=200` Integrated Gradients computes on demand in `~28s` and repeats return instantly from SQLite cache (`~23ms`). Ephemeral `.pt` tensor caches auto-prune at `50` runs / `30` days.

---

## 11. Remaining Limitations

- Built for local single-analyst / workstation demonstrations without user authentication or live streaming (`streaming_supported: false`).
- Expired snapshot tensor caches (`> 50` runs or `> 30` days) return `HTTP 409` for uncached IG requests while preserving SQLite predictions and already-cached explanations.

---

## 12. Confirmation of Protected Research Artifact Integrity

Confirmed unchanged via `git status --short` and `pytest`:
- Stage 1–5 source code (`src/`, `preprocess_graph.py`), [`ExpA2_RankGauss182_BiDirGCN_seed42.pt`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt), frozen threshold `0.57`, Stage 2 [`stage2_imputer_scaler_train_1_34.joblib`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/artifacts/stage2_imputer_scaler_train_1_34.joblib), Stage 4H [`rankgauss_transformer_train_1_34.joblib`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4h/rankgauss_transformer_train_1_34.joblib), validation/test splits, and Stage 1–5 research reports/results/figures remain **100% untouched**.

---

## 13. Exact Startup Instructions

**Terminal 1 — FastAPI Backend (`http://127.0.0.1:8010`):**
```powershell
.\venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8010
```

**Terminal 2 — React Vite Frontend (`http://127.0.0.1:5173`):**
```powershell
npm --prefix app/frontend run dev
```

*(Production preview after `npm --prefix app/frontend run build`: `npm --prefix app/frontend run preview` at `http://127.0.0.1:4173`.)*
