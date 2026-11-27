# Stage 6D — Interactive Graph Visualization & Explainable AI (XAI) Completion Report

**Date:** 2026-11-27  
**Application:** **FraudLens — Intelligent Transaction Risk Analysis** (*Graph-Based Fraud Detection & Explainable AI*)  
**Scope:** Stage 6D interactive directed cluster network visualization (`ClusterDetailPage`), on-demand Integrated Gradients XAI workspace (`TransactionDetailPage`), React lazy-loading + vendor chunk splitting, component testing (`Vitest` + `React Testing Library`), and full Step 42 live backend integration verification.  
**Boundary:** **STOPPED cleanly after Stage 6D.** Stage 1–5 research code, frozen `ExpA2_RankGauss182_BiDirGCN_seed42.pt` checkpoint, threshold `0.57`, preprocessing pipelines, and Stage 6B backend inference logic remain completely unmodified.

---

## 1. Files Created and Modified

### Modified (`app/frontend/` only)
- [`app/frontend/package.json`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/package.json): Added `"test": "vitest run"` script and devDependencies `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom`.
- [`app/frontend/vite.config.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/vite.config.ts): Configured Rollup `manualChunks` (`react-vendor`, `charts-vendor`, `graph-vendor`, `icons-api`) and `vitest` jsdom test environment.
- [`app/frontend/src/types/api.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/types/api.ts): Added exact typed interfaces (`FeatureAttributionItem`, `FeatureGroupBreakdownItem`, `OneHopNeighborItem`, `TransactionExplanationResponse`) matching `POST /api/runs/{run_id}/transactions/{tx_id}/explain`.
- [`app/frontend/src/api/fraudApi.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/api/fraudApi.ts): Added `explainTransaction(runId, txId)`.
- [`app/frontend/src/App.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/App.tsx): Added route-level `React.lazy()` and `<Suspense>` code-splitting across all `7` routes.
- [`app/frontend/src/pages/ClusterDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClusterDetailPage.tsx): Upgraded Cytoscape directed graph viewer with zoom/pan/fit/reset toolbar, visual legend, 1-hop context toggle, deterministic cluster role derivation, edge metadata table, and safe Cytoscape unmount cleanup.
- [`app/frontend/src/pages/TransactionDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionDetailPage.tsx): Replaced Stage 6C placeholder with complete on-demand `M=200` Integrated Gradients analytics (risk summary separating predicted vs ground truth, top positive/negative feature horizontal chart & tables, Local93/Aggregate72/Domain17 group breakdown, self-vs-neighbor logit shares, completeness residual diagnostics, directed 1-hop/2-hop context, caching badge, and retryable error states).

### Created
- [`app/frontend/src/tests/setupTests.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/tests/setupTests.ts)
- [`app/frontend/src/tests/ClusterDetailPage.test.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/tests/ClusterDetailPage.test.tsx)
- [`app/frontend/src/tests/TransactionDetailPage.test.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/tests/TransactionDetailPage.test.tsx)
- [`reports/stage6/stage6d_graph_xai_completion_report.md`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/stage6d_graph_xai_completion_report.md)

---

## 2. Cluster Graph Implementation Details

Implemented in [`ClusterDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClusterDetailPage.tsx):
- **Targeted Cluster-Only Directed Graph:** Renders strictly nodes and directed edges returned by `GET /api/runs/{run_id}/clusters/{cluster_id}/graph` using `cytoscape` (`target-arrow-shape: 'triangle'`, `curve-style: 'bezier'`, `directed: true`).
- **Interactive Controls:** Dedicated toolbar with `Zoom In` (`1.25x`), `Zoom Out` (`0.8x`), `Fit to Screen`, `Reset Graph View`, drag-to-pan, wheel zoom (`minZoom: 0.25`, `maxZoom: 3.0`), `Show 1-Hop Context` toggle, and layout switcher (`breadthfirst`, `cose`, `circle`).
- **Visual Encoding & Legend:** Distinguishes `High Risk (>= 0.85)` red nodes, `Flagged (>= 0.57)` orange nodes, `Moderate / Low` yellow/green nodes, `Cluster Hub` teal border highlight, `1-Hop External Boundary` slate dashed nodes, and solid internal vs dashed boundary directed edges.
- **Node Selection & Roles:** Tapping a Cytoscape node or clicking a table row updates the Selected Transaction Node panel with `tx_id`, `pred_prob` bar, `risk_category`, `pred_label`, deterministic cluster role (`Cluster Hub`, `Cluster Source`, `Cluster Sink`, `Internal Relay / Member`, or `1-Hop External Boundary`), snapshot `in_deg` / `out_deg`, `logit`, `optional_gt_label`, connected subgraph edge counts, and link to `TransactionDetailPage`.
- **Lifecycle Cleanup:** Effect cleanup calls `cy.removeAllListeners()` and `cy.destroy()` cleanly on unmount or data change.
- **Non-Criminal Disclaimer:** Prominent warning clearly communicates that clusters represent model-identified suspicious structural patterns at threshold `0.57`, not confirmed proof of criminal fraud.

---

## 3. Transaction Explanation (XAI) Implementation Details

Implemented in [`TransactionDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionDetailPage.tsx):
- **Risk vs Ground-Truth Separation:** Clearly separates `Model Predicted Class` (`Illicit (Predicted)` / `Licit (Predicted)` + `pred_prob` + `logit`) from `Actual Ground-Truth Class (Optional)` (`Unknown` / `Illicit` / `Licit`).
- **Strictly On-Demand Execution & Concurrency Guard:** Never auto-requests `/explain` on page load; uses an in-flight ref (`explainInFlightRef`) + disabled button state to prevent duplicate concurrent submissions while showing loading feedback (`200` path steps on CPU).
- **Feature Attribution Chart & Tables:** Recharts vertical `BarChart` plotting top `6` positive (fraud-increasing, `#ef4444`) and top `6` negative (score-decreasing, `#10b981`) RankGauss features from `top_positive_features` and `top_negative_features`, accompanied by full 10-row positive & negative breakdown tables (`graph_ig_attribution`, `self_ig_attribution`, `neighbor_ig_attribution`).
- **Feature Group Analysis:** Displays exact backend sums (`signed_sum`, `abs_sum`, `self_signed_sum`) organized into `Local Features (93)`, `Aggregate Features (72)`, and `Domain Features (17)` from `feature_group_breakdown`, plus `self_abs_share` / `neighbor_abs_share` percentages.
- **Explanation Quality & Completeness:** Shows `method` (`Composite_Midpoint_IG_3Zone_M200`), integration steps (`M = 200 (3-zone Composite Midpoint Grid)`), `abs_error` (`logits`), `rel_error_pct` (`%`), `logit_delta`, `ig_sum`, `all_finite` status badge, SQLite `cache_hit` badge (`Cached SQLite Result` vs `Freshly Computed`), and a plain-English summary of completeness.
- **1-Hop & 2-Hop Neighborhood Context:** Lists exact directed `one_hop_neighbors` (`edge_representation`, neighbor probability, neighbor optional GT, neighbor signed & `|IG|` contributions) alongside `two_hop_summary` statistics.

---

## 4. Exact API Endpoints Integrated

1. `GET /api/runs/{run_id}/clusters/{cluster_id}/graph` -> [`fetchClusterGraph`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/api/fraudApi.ts#L228-L240)
2. `POST /api/runs/{run_id}/transactions/{tx_id}/explain` -> [`explainTransaction`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/api/fraudApi.ts#L242-L255)
3. All Stage 6C routes retained (`GET /api/health`, `POST /api/validate-csv`, `POST /api/analyze`, `GET /api/runs`, `GET /api/runs/{run_id}`, `GET /api/runs/{run_id}/transactions`, `GET /api/runs/{run_id}/clusters`).

---

## 5. Tests Executed & Exact Verification Status

| Check / Suite | Outcome | Observed Evidence |
| :--- | :---: | :--- |
| **Vitest Frontend Component Suite** (`npm --prefix app/frontend test`) | **PASS** | **`2` files, `5/5` tests passed (`0` failed)** in `7.35s`: `ClusterDetailPage` (loading, directed Cytoscape rendering, legend, zoom/fit/reset controls, node selection role update, cleanup destroy; API error + Retry; empty graph state) and `TransactionDetailPage` (no auto-call before click, risk vs GT separation, on-demand IG chart/groups/completeness/1-hop neighbors; `409` error + Retry). |
| **TypeScript & Vite Production Build** (`npm --prefix app/frontend run build`) | **PASS** | **`0` TypeScript errors**, built cleanly in `11.44s` (`✓ 2066 modules transformed`), **zero `>500 kB` chunk warnings**. |
| **Live Full Step 42 Graph & `M=200` XAI Backend Integration** (`verify_stage6d_live.py`) | **PASS** | Uploaded real Step 42 snapshot (`7,140` nodes, `8,493` edges -> `run_id=run_20261126_194531_f460f6fb`, `612` flagged, `80` clusters, `24` candidates). Verified `C_T42_001` graph (`motif=complex_mesh`, `58` nodes, `60` edges, `hub_tx=73282397`) and `POST /explain` on `73282397` (`method=Composite_Midpoint_IG_3Zone_M200`, `abs_err=0.008639`, `rel_err=0.0904%`, `top_pos=Local_feature_3 (+5.23532)`, `28.38s` fresh -> `0.0230s` SQLite cache hit). |
| **Stage 6B Backend Regression Suite** (`pytest -v app/backend/tests/test_stage6b_backend.py`) | **PASS** | **`10/10` tests passed (`0` failed)** in `50.29s`, re-verifying full Step 35 (`5,507` nodes) & Step 42 (`7,140` nodes) `0.00e+00` model parity and `0.57` threshold. |
| **Chrome DevTools Automated Browser Run & Screenshots** | **NOT TESTED** | `chrome_devtools` could not connect because Chrome remote debugging (`C:\Users\mansi\AppData\Local\Google\Chrome\User Data\DevToolsActivePort`) was unavailable on the host (`ENOENT`). Automated screenshot artifacts were therefore not captured; DOM rendering and live Vite preview + FastAPI HTTP integration were verified instead via Vitest/jsdom + `vite preview` + live `httpx` integration tests. |
| **JSDOM Recharts ResponsiveContainer & Vitest/C8 Warnings** | **WARNING** | During Vitest jsdom execution (which has `0x0` layout geometry), Recharts logged `The width(0) and height(0) of chart should be greater than 0`, and React Router v6 emitted standard `v7_startTransition` / `v7_relativeSplatPath` future-flag notices. Neither occurs in error state or breaks builds. |

---

## 6. Production Bundle-Size Comparison (Stage 6C vs Stage 6D)

| Artifact / Chunk | Stage 6C (`vite build`) | Stage 6D (`vite build`) | Change |
| :--- | :---: | :---: | :---: |
| **Initial Entry JS (`index-*.js`)** | `1,154.84 kB` (`349.54 kB` gzip) | **`12.98 kB` (`4.83 kB` gzip)** | **`-98.88%`** (`89×` smaller initial load) |
| **Entry CSS (`index-*.css`)** | `9.37 kB` (`2.65 kB` gzip) | `9.37 kB` (`2.65 kB` gzip) | Unchanged |
| **`react-vendor-*.js`** | Bundled in entry | `165.01 kB` (`53.97 kB` gzip) | Shared core vendor chunk |
| **`icons-api-*.js`** | Bundled in entry | `74.76 kB` (`25.37 kB` gzip) | Shared icons + axios chunk |
| **`charts-vendor-*.js` (`recharts`)** | Bundled in entry | `395.55 kB` (`108.81 kB` gzip) | Deferred to chart routes |
| **`graph-vendor-*.js` (`cytoscape`)** | Bundled in entry | `451.17 kB` (`144.50 kB` gzip) | Loaded ONLY on `ClusterDetailPage` |
| **Lazy Page Chunks (each route)** | Bundled in entry | `1.42 kB – 17.02 kB` | On-demand per route |
| **Vite >500 kB Chunk Warning** | **1 Warning** (`1,154.84 kB`) | **0 Warnings** (largest chunk `451.17 kB`) | **Eliminated** |

---

## 7. Known Limitations & Remaining Issues

- **Host Browser Screenshots Not Captured:** Live desktop Chrome visual screenshots remain uncaptured until Chrome remote debugging (`chrome://inspect/#remote-debugging`) is enabled locally (`NOT TESTED` row above).
- **CPU On-Demand Latency on Large Snapshots:** First-time `M=200` Integrated Gradients computation on a `7,140`-node snapshot ran in `28.38s` on CPU (cached lookups take `0.023s`), mitigated by loading feedback and SQLite caching; multi-minute uploads or expired caches (`HTTP 409` after `50`-run / `30`-day eviction) display clear user retry/re-upload guidance.
- **Responsive Cytoscape Interaction in Unit Tests:** Cytoscape layout/zoom APIs were verified using spies in JSDOM (`Vitest`), while pixel layout at mobile breakpoints relies on CSS grid stacking rules in [`index.css`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/styles/index.css) (`@media (max-width: 1080px)` and `@media (max-width: 820px)`).

---

## 8. Research Artifact Protection Confirmation

Confirmed **100% unchanged** via `git status --short` and `test_stage6b_backend.py`:
- All Stage 1–5 research source code (`src/`, `preprocess_graph.py`), frozen model checkpoint [`ExpA2_RankGauss182_BiDirGCN_seed42.pt`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt), threshold `0.57`, Stage 2 artifact [`stage2_imputer_scaler_train_1_34.joblib`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/artifacts/stage2_imputer_scaler_train_1_34.joblib), Stage 4H [`rankgauss_transformer_train_1_34.joblib`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4h/rankgauss_transformer_train_1_34.joblib), validation/test splits, Stage 1–5 reports/results/figures, and Stage 6B inference logic are untouched.

---

## 9. Startup Commands

**Terminal 1 — FastAPI Backend (`http://127.0.0.1:8010`):**
```powershell
.\venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8010
```

**Terminal 2 — React Vite Frontend (`http://127.0.0.1:5173`):**
```powershell
npm --prefix app/frontend run dev
```

*(Run frontend unit tests: `npm --prefix app/frontend test`; production build & preview: `npm --prefix app/frontend run build` and `npm --prefix app/frontend run preview` at `http://127.0.0.1:4173`.)*
