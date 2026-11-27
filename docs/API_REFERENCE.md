# FraudLens Backend REST API & Architecture Reference

Base URL (local development): `http://127.0.0.1:8010`  
Interactive Swagger/OpenAPI Docs: `http://127.0.0.1:8010/docs`

---

## Endpoints Summary

| Method & Path | Purpose | Key Parameters / Payload |
| :--- | :--- | :--- |
| `GET /api/health` | Verifies frozen model bundle, threshold (`0.57`), artifact paths, device (`cpu`/`cuda`), and batch-only status (`streaming_supported: false`). | None |
| `POST /api/validate-csv` | Validates schema, types, duplicate `txId`, optional `class`, dangling edges, and cross-timestep edges (`HTTP 422` on violation). | Multipart: `transactions_file`, `edges_file`, optional `classes_file` |
| `POST /api/analyze` | Runs frozen `Stage 2 -> RankGauss -> BiDirGCN` inference + Stage 5C clustering, persists run to SQLite (`app/data/fraud_app.db`), caches snapshot `.pt` tensors, returns `run_id`. | Multipart: `transactions_file`, `edges_file`, optional `classes_file`, optional `dataset_name` |
| `GET /api/runs` | Lists persisted analysis runs sorted newest-first. | None |
| `GET /api/runs/{run_id}` | Returns complete run summary, risk band distribution, per-timestep breakdown, shared bridge samples, and post-hoc optional label evaluation (if supplied). | Path: `run_id` |
| `GET /api/runs/{run_id}/transactions` | Paginated & filtered transaction risk predictions ordered by `pred_prob DESC, tx_id ASC`. | Query: `timestep`, `flagged_only`, `min_prob`, `search_tx_id`, `limit` (`1..500`), `offset` |
| `GET /api/runs/{run_id}/clusters` | Lists weakly connected suspicious clusters (`|C| >= 2`, `P >= 0.57`) and Tier-A Fraud-Ring Candidates. | Query: `candidates_only`, `timestep` |
| `GET /api/runs/{run_id}/clusters/{cluster_id}/graph` | Returns cluster metadata, member nodes, internal directed edges, and up to `40` external 1-hop context nodes/boundary edges (when cache exists). | Path: `run_id`, `cluster_id` |
| `POST /api/runs/{run_id}/transactions/{tx_id}/explain` | Computes on-demand (or returns SQLite-cached) 3-zone `M=200` Composite Midpoint Integrated Gradients explanation + 1-hop/2-hop context. | Path: `run_id`, `tx_id` |

---

## Frozen Inference & Persistence Behavior

1. **Deterministic Transform Order:** Raw `[N, 182]` features pass strictly through `app/backend/artifacts/stage2_imputer_scaler_train_1_34.joblib` (`SimpleImputer(median)` + `StandardScaler()`), then `models/stage4h/rankgauss_transformer_train_1_34.joblib` (`QuantileTransformer(normal)` + `StandardScaler()`), then `models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt` (`.eval()`, `torch.no_grad()`, threshold `0.57`).
2. **Cache & Data Retention Policy:** Uploaded raw CSV files are processed in memory and **never** written to permanent storage. Ephemeral processed `.pt` snapshot tensors (`app/data/runs_cache/{run_id}/`) support on-demand `M=200` IG calculations and external 1-hop graph boundaries up to **`50` runs / `30` days** (`HTTP 409 snapshot_cache_expired` thereafter while SQLite predictions and already-cached IG JSON payloads remain queryable).
3. **Access & Streaming Scope:** Designed as a local single-analyst workstation application without user authentication and without live blockchain streaming (`streaming_supported: false`).
