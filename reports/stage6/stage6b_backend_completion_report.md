# Stage 6B — Backend Core Development and Frozen Model Integration Report

**Date:** 2026-11-26  
**Scope:** Stage 6B FastAPI backend implementation under `app/backend/` — frozen Stage 4H `ExpA2_RankGauss182_BiDirGCN` integration, exact Stage 2 + RankGauss preprocessing, SQLite persistence (`app/data/fraud_app.db`), Stage 5C suspicious cluster discovery, on-demand Stage 5B 3-zone Composite Midpoint Integrated Gradients (`M=200`), and end-to-end verification (`pytest` + Step 35 / Step 42 numerical parity + live Uvicorn server probe).  
**Boundary:** React frontend (`Stage 6C`) was intentionally NOT implemented in this stage. All Stage 1–5 source code, checkpoints, processed data, results, reports, and figures remain unmodified.

---

## 1. Backend Directory & Module Structure

Created under [`app/backend/`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend):

| Module | Purpose |
| :--- | :--- |
| [`app/backend/config.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/config.py) | Read-only paths to frozen artifacts (`CHECKPOINT_PATH`, `RANKGAUSS_PATH`, `STAGE2_ARTIFACT_PATH`, `METADATA_PATH`), SQLite/cache settings (`MAX_CACHED_RUNS=50`, `RETENTION_DAYS=30`), upload limits (`100 MB`, `250,000` nodes, `500,000` edges), CORS origins, expected threshold `0.57`. |
| [`app/backend/schemas.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/schemas.py) | Pydantic response models (`HealthResponse`, `ValidationResponse`, `AnalyzeResponse`, `RunSummaryItem`). |
| [`app/backend/scripts/export_stage2_artifact.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/scripts/export_stage2_artifact.py) | Fits `Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])` strictly on raw `dataset/txs_features.txt` train Steps `1–34` sorted by `['Time step', 'txId']`, exports `app/backend/artifacts/stage2_imputer_scaler_train_1_34.joblib` (`6,113` bytes), and verifies `0.00e+00` parity vs `processed/temporal_snapshots.pt` on Steps `1, 35, 42`. |
| [`app/backend/ml/frozen_model.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/ml/frozen_model.py) | Singleton `FrozenFraudModelBundle` applying the exact frozen transform order (`Stage 2 Imputer+Scaler -> Stage 4H RankGauss -> BiDirRelationalGCN.eval()`), reading `threshold_recal_01_99 = 0.57` directly from the checkpoint, with GPU-with-CPU-fallback selection and snapshot `torch.no_grad()` inference. |
| [`app/backend/services/csv_validator.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/services/csv_validator.py) | Strict multipart CSV validator enforcing exact `txId`, `Time step`, and `182` feature column schema, unique integer `txId`, non-numeric/`Inf` rejection (`HTTP 422`), `NaN` count reporting for median imputation, optional `class` column isolation, and strict **no-silent-drop** edge validation (`dangling_edges` and `cross_timestep_edges` raise `HTTP 422`; exact duplicate intra-timestep edges deduplicated with warning). |
| [`app/backend/services/subgraph_service.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/services/subgraph_service.py) | Reuses Stage 5C [`classify_directed_motif`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/stage5/stage5c_fraud_ring_analysis.py#L82-L105) and [`gt_label_str`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/src/stage5/stage5b_individual_explanations.py#L55-L62) to extract intra-snapshot weakly connected predicted suspicious clusters (`|C| >= 2`, `probs >= 0.57`), Tier-A fraud-ring candidates, directed roles, post-hoc ground-truth summaries (`Unknown = -1` preserved strictly), and unflagged shared bridges across arbitrary uploaded timesteps (`>= 1`). |
| [`app/backend/services/explainability_service.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/services/explainability_service.py) | On-demand Stage 5B 3-zone Composite Midpoint Integrated Gradients (`60` pts `[0, 0.005]` + `60` pts `[0.005, 0.05]` + `80` pts `[0.05, 1.0]`, zero baseline in RankGauss space, `logit[node_idx]` target), returning completeness diagnostics, `self_vs_neighbor` shares, `Local93`/`Aggregate72`/`Domain17` group sums, top +/- features, 1-hop directed neighbors, and 2-hop stats, cached in SQLite `explanations`. |
| [`app/backend/services/analyze_service.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/services/analyze_service.py) | Orchestrates validation, frozen preprocessing, strictly intra-timestep BiDirGCN inference (`drop_edge_p=0.0`), Stage 5C clustering, ephemeral snapshot tensor caching (`app/data/runs_cache/{run_id}/step_{t}.pt`), and SQLite persistence. |
| [`app/backend/database/db.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/database/db.py) | Parameterized SQLite persistence (`runs`, `transactions`, `clusters`, `cluster_edges`, `explanations`) initialized at [`app/data/fraud_app.db`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/data/fraud_app.db) with bounded retention (`50` cached runs / `30` days). |
| [`app/backend/api/routes.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/api/routes.py) & [`app/backend/main.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/main.py) | FastAPI application with lifespan warmup (`init_db()`, `get_model_bundle()`), CORS configuration, and all `9` REST API endpoints. |
| [`app/backend/tests/test_stage6b_backend.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/tests/test_stage6b_backend.py) & [`app/backend/scripts/verify_parity_and_server.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/scripts/verify_parity_and_server.py) | Comprehensive `pytest` suite (`10` test cases including full Step 35 & Step 42 parity) + live Uvicorn verification harness. |

---

## 2. Implemented REST API Endpoints

1. **`GET /api/health`** — Reports loaded frozen checkpoint path, `model_id = "ExpA2_RankGauss182_BiDirGCN"`, `threshold_used = 0.57`, preprocessing pipeline order, compute device, artifact readiness flags, `feature_count = 182`, and `streaming_supported = false`.
2. **`POST /api/validate-csv`** — Multipart upload (`transactions_file`, `edges_file`, optional `classes_file`). Returns schema summary, deduplicated edge stats, imputed `NaN` count, optional class counts, or structured `HTTP 422` diagnostics.
3. **`POST /api/analyze`** — Runs full validation -> frozen Stage 2 -> RankGauss -> BiDirGCN inference -> Stage 5C clustering -> SQLite persistence + snapshot tensor cache creation; returns `AnalyzeResponse` with `run_id`, flagged counts, cluster/candidate counts, warnings, and investigative disclaimer.
4. **`GET /api/runs`** — Lists persisted runs sorted newest-first.
5. **`GET /api/runs/{run_id}`** — Returns full run metadata, risk-band counts, per-timestep breakdown, bridge samples, and optional post-hoc evaluation (when optional `class` was provided).
6. **`GET /api/runs/{run_id}/transactions`** — Supports parameterized filtering (`timestep`, `flagged_only`, `min_prob`, `search_tx_id`) and pagination (`limit`, `offset`), sorted by descending fraud probability.
7. **`GET /api/runs/{run_id}/clusters`** — Lists Stage 5C weakly connected suspicious clusters (`>= 2` flagged nodes) with optional `candidates_only` and `timestep` filters.
8. **`GET /api/runs/{run_id}/clusters/{cluster_id}/graph`** — Returns cluster metadata, member nodes, directed internal edges, plus up to `40` 1-hop external boundary nodes and boundary edges when the snapshot tensor cache is present.
9. **`POST /api/runs/{run_id}/transactions/{tx_id}/explain`** — Computes (or serves SQLite-cached) Stage 5B 3-zone Composite Midpoint IG (`M=200`) explanation and directed 1-hop/2-hop neighborhood breakdown (`HTTP 409 snapshot_cache_expired` if tensor cache was evicted).

---

## 3. Numerical Parity Verification (Step 35 & Step 42)

Verified via [`test_stage6b_backend.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/tests/test_stage6b_backend.py) and [`verify_parity_and_server.py`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/backend/scripts/verify_parity_and_server.py) against untouched Stage 2 [`processed/temporal_snapshots.pt`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/processed/temporal_snapshots.pt), Stage 4H [`rankgauss_transformer_train_1_34.joblib`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4h/rankgauss_transformer_train_1_34.joblib), and frozen [`ExpA2_RankGauss182_BiDirGCN_seed42.pt`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt):

| Snapshot | Nodes | Intra-Step Edges | Stage-2 Max Abs Diff | RankGauss Max Abs Diff | Logit Max Abs Diff | Prob Max Abs Diff | Binary Match (`@0.57`) | Flagged Count | Direct Clusters | Ring Candidates | Shared Bridges |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 35** (Val) | `5,507` | `6,351` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **`True` (100%)** | `515` | `49` | `19` | `61` |
| **Step 42** (Test) | `7,140` | `8,493` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `0.00e+00` | **`True` (100%)** | `612` | `80` | `24` | `52` |

In addition, all `10/10` tests in `pytest` passed (`147.06s`, exit code `0`) covering:
- `GET /api/health` contract & threshold check (`0.57`, `streaming_supported=false`)
- Valid CSV parsing + `NaN` feature median imputation (`200`)
- Strict `HTTP 422` rejections for missing feature columns, non-numeric text values, `+/-Inf` entries, duplicate `txId`s, dangling edge endpoints, and cross-timestep edges (zero silent dropping)
- Exact invariance of predicted `logit` and `pred_prob` when optional `class` labels are omitted vs flipped
- End-to-end persistence, filtering, cluster graph retrieval, IG completeness (`abs_error < 0.05`, `rel_error_pct < 1.0%`, `all_finite=True`), SQLite explanation caching (`cache_hit=True`), and `HTTP 409 snapshot_cache_expired` handling after cache eviction
- Full Step 35 & Step 42 end-to-end API upload parity (`< 1e-5` SQLite rounding tolerance vs raw tensor outputs).

---

## 4. Environment, Startup Command & Live Probe Results

Installed backend packages into [`./venv`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/venv): `fastapi-0.165.0`, `uvicorn-0.54.0`, `python-multipart-0.0.34`, `httpx-0.29.0`, `pytest-9.2.0`, `pydantic-2.14.4` (retaining existing PyTorch `2.14.1`, PyG `2.8.0.post1`, scikit-learn `1.9.1`).

**Run backend locally:**
```powershell
.\venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

**Run backend verification suite:**
```powershell
.\venv\Scripts\python.exe -m pytest -v app/backend/tests/test_stage6b_backend.py
```

**Verified Live Uvicorn Probe (`127.0.0.1:8010/api/health`):**
```json
{
  "status": "ok",
  "model_id": "ExpA2_RankGauss182_BiDirGCN",
  "checkpoint_path": "C:\\Users\\mansi\\Downloads\\Capstone_Fraud_Detection\\models\\stage4h\\ExpA2_RankGauss182_BiDirGCN_seed42.pt",
  "threshold_used": 0.57,
  "preprocessing_order": [
    "Stage2 SimpleImputer(median) + StandardScaler (Steps 1-34)",
    "Stage4H QuantileTransformer(normal) + StandardScaler (Steps 1-34)"
  ],
  "device": "cpu",
  "stage2_artifact_ready": true,
  "rankgauss_ready": true,
  "feature_count": 182,
  "streaming_supported": false
}
```
SQLite database verified initialized at [`app/data/fraud_app.db`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/app/data/fraud_app.db).

---

## 5. Scope Notes & Honest Limitations

- **Batch CSV Upload Only:** Live blockchain ingestion / streaming is not implemented (`streaming_supported: false`); inference operates synchronously on uploaded CSV snapshots.
- **Probabilistic Risk & XAI Scope:** Integrated Gradients quantifies logit attribution in RankGauss-normalized feature space relative to the zero/median baseline, and predicted flags/clusters require human analyst review rather than serving as criminal proof.
- **Ephemeral Tensor Cache Retention:** Raw CSV uploads are discarded after request completion; cached per-timestep `.pt` tensors used for new IG calculations retain at most `50` runs / `30` days (`HTTP 409` on expired runs while SQLite prediction tables & previously cached IG explanations remain queryable).
