# Stage 6C — React Frontend Dashboard Development Completion Report

**Date:** 2026-11-27  
**Application Title:** **FraudLens — Intelligent Transaction Risk Analysis** (*Graph-Based Fraud Detection & Explainable AI*)  
**Scope:** Stage 6C React + TypeScript + Vite web application under [`app/frontend/`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend) connected to the Stage 6B FastAPI backend at `http://127.0.0.1:8010`.  
**Boundary:** Frontend integration only. Stage 1–5 source code, checkpoints, frozen threshold (`0.57`), preprocessing artifacts, backend inference logic, and research reports remain untouched. **Stage 6D** (full individual XAI visualizations & interactive graph explanation overlays) has not been started.

---

## 1. Pages Implemented

| Route | Component | Implemented Functionality |
| :--- | :--- | :--- |
| `/` | [`DashboardPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/DashboardPage.tsx) | Aggregated KPI cards across persisted runs (`Total Analysed Transactions`, `High-Risk / Flagged Transactions`, `Suspicious Clusters`, `Analysis Run Count`), batch offline engine banner (explicitly stating no live blockchain stream), Latest Analysis Summary card, Recharts Risk Band bar chart (`High >=0.85`, `Elevated >=0.57`, `Moderate 0.35-0.57`, `Low <0.35`), Recharts Per-Timestep Flagged & Cluster chart, Recent Analysis History table, loading/error retry states, and empty-state CTA when 0 runs exist. |
| `/upload` | [`UploadPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/UploadPage.tsx) | Dual drag-and-drop & file-picker CSV upload for required `transactions.csv` (`txId`, `Time step`, exact `182` features) and `edges.csv` (`txId1`, `txId2`), plus collapsible optional `classes.csv` with explicit post-hoc-only isolation banner. Client checks (`.csv`/`.txt`, non-empty, `<= 100 MB`), filename/size preview, remove/replace, upload progress bar, `POST /api/validate-csv` diagnostics (warnings, `NaN` counts, structured `HTTP 422` error panel), gated `POST /api/analyze` lock preventing duplicate submissions, and automatic redirect to `/runs/:runId`. |
| `/runs/:runId` | [`RunDetailsPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/RunDetailsPage.tsx) | Run metadata, model ID & preprocessing pipeline summary, KPI cards (`node_count`, `edge_count`, `flagged_count`, `flagged_rate_pct`, `direct_clusters_count`, `potential_candidates_count`, `shared_bridges_count`), Recharts Risk Distribution bar chart, optional post-hoc ground-truth comparison card (strictly hidden unless optional labels exist), warnings/disclaimers, and Per-Timestep metrics table. |
| `/runs/:runId/transactions` | [`TransactionsPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionsPage.tsx) | Server-filtered & paginated Transaction Risk Table (`tx_id`, `timestep`, probability bar + `pred_prob`, `pred_label`, `risk_category`, `in_deg`/`out_deg`, clickable `cluster_id`, `optional_gt_label` preserving `Unknown` vs `Illicit` vs `Licit`), numeric `search_tx_id`, risk filter (`All`, `Flagged >= 0.57`, `High >= 0.85`, `Moderate >= 0.35`), timestep selector, sortable headers, and Detail navigation. |
| `/runs/:runId/transactions/:txId` | [`TransactionDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionDetailPage.tsx) | Stored prediction metadata (`tx_id`, `pred_prob`, `logit`, `pred_label`, `risk_category`, `in_deg`/`out_deg`, `cluster_id`, `optional_gt_label`) and clean **Stage 6D Individual Transaction Explainability Foundation** route linked to `POST /api/runs/{run_id}/transactions/{tx_id}/explain` with zero fabricated attributions. |
| `/runs/:runId/clusters` | [`ClustersPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClustersPage.tsx) | Investigative caution banner stating clusters are model-identified suspicious groups rather than confirmed criminal fraud rings; `candidates_only` & `timestep` filters; displays `cluster_id`, `timestep`, `num_nodes`, `num_internal_edges`, `mean_pred_prob`/`max_pred_prob`, `ring_score`, directed `motif_type`, internal/external degree stats, candidate status badge, `hub_tx_id`, and Open Cluster navigation. |
| `/runs/:runId/clusters/:clusterId` | [`ClusterDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClusterDetailPage.tsx) | Live **Cytoscape.js** directed network visualization driven by `GET /api/runs/{run_id}/clusters/{cluster_id}/graph` (Directed Flow / Force-Directed layout switcher, internal suspicious edges vs dashed 1-hop external context, risk-colored nodes, hub highlight), interactive Selected Node Inspector linking to Transaction Detail, and Member/Boundary Nodes table. |

---

## 2. Components, Styling & Centralized API Integration

- [`src/components/AppLayout.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/components/AppLayout.tsx): Persistent dark-navy sidebar + sticky top bar displaying live backend status via `GET /api/health`, frozen model & threshold metadata, manual refresh button, active run sub-navigation links, and responsive mobile drawer toggle.
- [`src/components/RiskBadge.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/components/RiskBadge.tsx): Shared `RiskBadge`, `ProbBar`, `GtBadge`, and number/byte/timestamp formatters aligned cleanly to backend risk strings (`High Predicted Risk (P >= 0.85)`, `Elevated Risk (Flagged >= 0.57)`, `Moderate Monitor (0.35 - 0.57)`, `Low Predicted Risk (< 0.35)`).
- [`src/types/api.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/types/api.ts) & [`src/api/fraudApi.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/api/fraudApi.ts): Centralized Axios client (`API_BASE_URL` defaulting to `http://127.0.0.1:8010`) with typed response models matching backend schemas (`GET /api/health`, `POST /api/validate-csv`, `POST /api/analyze`, `GET /api/runs`, `GET /api/runs/{run_id}`, `GET /api/runs/{run_id}/transactions`, `GET /api/runs/{run_id}/clusters`, `GET /api/runs/{run_id}/clusters/{cluster_id}/graph`) and normalized `BackendRequestError` handling network offline and `HTTP 413`/`422`/`404`/`409` responses cleanly.
- [`src/styles/index.css`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/styles/index.css): Custom fintech dark navy/charcoal theme (`#070d19` canvas, `#0d1628` cards, restrained `#38bdf8` blue and `#2dd4bf` teal accents) with responsive breakpoints at `1080px` and `820px`.

---

## 3. Dependencies Installed (`app/frontend/package.json`)

- **Runtime (`dependencies`):** `react@^18.3.1`, `react-dom@^18.3.1`, `react-router-dom@^6.28.0`, `axios@^1.7.9`, `recharts@^2.15.0`, `lucide-react@^0.468.0`, `cytoscape@^3.30.4` (`140` total packages installed in `app/frontend/node_modules`).
- **Build / Type-checking (`devDependencies`):** `typescript@^5.7.2`, `vite@^6.0.7`, `@vitejs/plugin-react@^4.3.4`, `@types/react@^18.3.12`, `@types/react-dom@^18.3.1`, `@types/cytoscape@^3.21.9`.

---

## 4. Build, Live Integration & Verification Results

1. **TypeScript & Production Vite Build (`npm --prefix app/frontend run build`):**
   - Exited with code `0` (`✓ 2065 modules transformed`, built in `15.78s`):
     - `dist/index.html` (`0.75 kB`)
     - `dist/assets/index-Ds3iCs9H.css` (`9.37 kB`)
     - `dist/assets/index-Cb62Ydhg.js` (`1,154.84 kB`)
2. **Live HTTP, CORS & End-to-End API Integration (`verify_stage6c_live.py`):**
   - Verified built frontend `vite preview` (`http://127.0.0.1:4173/`) serving `FraudLens` HTML (`HTTP 200`, `748` bytes).
   - Verified CORS `Access-Control-Allow-Origin` headers on `GET /api/health` (`threshold_used = 0.57`, `streaming_supported = false`) for both `:4173` and `:5173` Vite origins.
   - Verified invalid dangling-edge Step 42 CSV upload cleanly returns `HTTP 422` (`dangling_edges`).
   - Verified valid `500`-node, `223`-edge Step 42 slice CSV validation (`HTTP 200`, `valid = true`), analysis completion (`HTTP 200`, `run_id = run_20261126_191741_71b498c1`, `flagged = 14`), `/api/runs`, `/api/runs/{run_id}`, `/api/runs/{run_id}/transactions` (`total = 14`), and `/api/runs/{run_id}/clusters`.
3. **Stage 6B Backend Regression Check (`.\venv\Scripts\python.exe -m pytest -v app/backend/tests/test_stage6b_backend.py`):**
   - **`10 passed`, `0 failed`** in `47.09s` (exit code `0`), confirming zero regressions in backend schemas, CSV validation, SQLite persistence, and Step 35 / Step 42 model parity.

---

## 5. Known Limitations & Remaining Scope for Stage 6D

- **Automated Chrome DevTools Screenshots Unavailable:** Live headless/desktop Chrome UI automation via `chrome_devtools` could not connect because Chrome remote debugging (`DevToolsActivePort`) was not active on the host machine (`ENOENT`). Build compilation, bundle serving, CORS headers, and all API request/response flows were verified directly via CLI/HTTP tools instead.
- **Deferred to Stage 6D (XAI & Advanced Network Interactivity):**
  - Calling `POST /api/runs/{run_id}/transactions/{tx_id}/explain` from [`TransactionDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionDetailPage.tsx) to render the `M=200` Integrated Gradients positive/negative feature attribution bar/waterfall charts, `Local93` / `Aggregate72` / `Domain17` group summaries, self-vs-neighbor logit shares, completeness badges, and directed 1-hop/2-hop neighbor tables.
  - Interactive Cytoscape node-level IG overlays, edge filtering (internal vs external boundary), and cluster triage export controls in [`ClusterDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClusterDetailPage.tsx).
- **Large Vite Single-Chunk Notice:** Vite emitted a build-time warning that `dist/assets/index-Cb62Ydhg.js` (`1,154.84 kB`) exceeds `500 kB` due to bundled `recharts` + `cytoscape`; route-level `React.lazy()` code-splitting can be added in Stage 6D if desired.

---

## 6. Exact Startup Commands

**Terminal 1 — FastAPI Backend (`http://127.0.0.1:8010`):**
```powershell
.\venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8010
```

**Terminal 2 — React Vite Frontend (`http://127.0.0.1:5173`):**
```powershell
npm --prefix app/frontend run dev
```

*(Production preview alternative after `npm --prefix app/frontend run build`: `npm --prefix app/frontend run preview` at `http://127.0.0.1:4173`.)*

---

## 7. Files Created Under Stage 6C

- [`app/frontend/package.json`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/package.json)
- [`app/frontend/tsconfig.json`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/tsconfig.json)
- [`app/frontend/vite.config.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/vite.config.ts)
- [`app/frontend/index.html`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/index.html)
- [`app/frontend/src/vite-env.d.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/vite-env.d.ts)
- [`app/frontend/src/main.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/main.tsx)
- [`app/frontend/src/App.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/App.tsx)
- [`app/frontend/src/styles/index.css`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/styles/index.css)
- [`app/frontend/src/types/api.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/types/api.ts)
- [`app/frontend/src/api/fraudApi.ts`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/api/fraudApi.ts)
- [`app/frontend/src/components/AppLayout.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/components/AppLayout.tsx)
- [`app/frontend/src/components/RiskBadge.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/components/RiskBadge.tsx)
- [`app/frontend/src/pages/DashboardPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/DashboardPage.tsx)
- [`app/frontend/src/pages/UploadPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/UploadPage.tsx)
- [`app/frontend/src/pages/RunDetailsPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/RunDetailsPage.tsx)
- [`app/frontend/src/pages/TransactionsPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionsPage.tsx)
- [`app/frontend/src/pages/TransactionDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/TransactionDetailPage.tsx)
- [`app/frontend/src/pages/ClustersPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClustersPage.tsx)
- [`app/frontend/src/pages/ClusterDetailPage.tsx`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/frontend/src/pages/ClusterDetailPage.tsx)
- [`reports/stage6/stage6c_frontend_completion_report.md`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/reports/stage6/stage6c_frontend_completion_report.md)
