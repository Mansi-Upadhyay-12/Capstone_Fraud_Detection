# FraudLens CSV Upload Format & User Workflow Guide

This guide specifies the exact CSV schemas, validation rules, error codes, and step-by-step workflow for analysing transaction snapshots in **FraudLens — Intelligent Transaction Risk Analysis**.

---

## 1. Quick-Try Sample Templates (`app/sample_data/`)

Three ready-to-upload minimal sample templates (`13` connected transactions and `12` directed edges centered around Step 42) are included in `app/sample_data/` without modifying the original Elliptic++ dataset:
- [`app/sample_data/transactions_sample.csv`](../app/sample_data/transactions_sample.csv) — Required transactions input (`184` columns: `txId`, `Time step`, + exact `182` numerical features).
- [`app/sample_data/edges_sample.csv`](../app/sample_data/edges_sample.csv) — Required directed edge list (`txId1`, `txId2`).
- [`app/sample_data/classes_optional_sample.csv`](../app/sample_data/classes_optional_sample.csv) — Optional post-hoc ground-truth labels (`txId`, `class`).

---

## 2. Exact CSV File Requirements

### File 1: Transactions CSV (Required)
- **Format:** UTF-8 CSV (`.csv` or `.txt`), max `100 MB`, max `250,000` rows per upload.
- **Required Columns (`184` columns + optional `class`):**
  1. `txId` — Unique non-null integer transaction identifier (`int64`, no duplicates).
  2. `Time step` — Non-null integer temporal snapshot index (`>= 1`).
  3. **Exact `182` raw unscaled numerical feature columns** matching `processed/metadata.json["feature_columns"]`:
     - **Local Features (`93`):** `Local_feature_1` through `Local_feature_93`
     - **Aggregate Features (`72`):** `Aggregate_feature_1` through `Aggregate_feature_72`
     - **Bitcoin Domain Features (`17`):** `in_txs_degree`, `out_txs_degree`, `total_BTC`, `fees`, `size`, `num_input_addresses`, `num_output_addresses`, `in_BTC_min`, `in_BTC_max`, `in_BTC_mean`, `in_BTC_median`, `in_BTC_total`, `out_BTC_min`, `out_BTC_max`, `out_BTC_mean`, `out_BTC_median`, `out_BTC_total`.
- **Missing Values & Numeric Policy:**
  - Empty/`NaN` cells in the `182` feature columns are permitted and are automatically median-imputed using Stage 2 Train `Steps 1–34` statistics (the count of imputed cells is reported in validation warnings).
  - Non-numeric strings or `+/-Inf` entries in any feature column are **rejected cleanly (`HTTP 422`)**.
  - Duplicate headers or duplicate `txId` rows are **rejected cleanly (`HTTP 422`)**.

### File 2: Directed Edges CSV (Required)
- **Format:** UTF-8 CSV (`.csv` or `.txt`), max `100 MB`, max `500,000` edges per upload.
- **Required Columns (`2`):**
  - `txId1` — Non-null integer source transaction ID.
  - `txId2` — Non-null integer destination transaction ID (directed link `txId1 -> txId2`).
- **Strict Graph Integrity Policy (No Silent Dropping):**
  - **Dangling Edges:** Every `txId1` and `txId2` must exist in File 1 (`transactions.csv`). Missing endpoints trigger `HTTP 422 (dangling_edges)` with sample offending pairs — edges are never silently discarded.
  - **Cross-Timestep Edges:** Both endpoints of every edge must share the exact same `Time step`. Cross-snapshot edges trigger `HTTP 422 (cross_timestep_edges)`.
  - **Duplicate Edges:** Exact duplicate directed pairs `(txId1, txId2)` are deduplicated after integrity validation and counted in warnings (`duplicate_edges_removed`).

### File 3: Optional Class Labels CSV (Optional — Post-Hoc Only)
- May be supplied either as a `class` column inside `transactions.csv` or as a separate `classes.csv` (`txId`, `class`).
- **Accepted Values:** `1` / `illicit` -> Illicit (`1`); `2` / `0` / `licit` -> Licit (`0`); `3` / `-1` / `unknown` / blank -> Unknown (`-1`). Unlabeled `Unknown (-1)` is strictly preserved separately from `Licit (0)`.
- **Strict Isolation Guarantee:** Optional `class` labels **never enter** Stage 2 preprocessing, RankGauss transformation, BiDirGCN inference, or Stage 5C cluster/candidate scoring. They appear solely in optional post-hoc evaluation panels (`TP`/`FP`/`FN`/`TN`) and table badges.

---

## 3. Common Upload Errors & Troubleshooting

| Error Code | Cause | Resolution |
| :--- | :--- | :--- |
| `422 missing_transaction_columns` | One or more of `txId`, `Time step`, or the `182` features is absent or misspelled. | Verify exact header casing against `app/sample_data/transactions_sample.csv`. |
| `422 non_numeric_features` / `infinite_features` | A feature column contains text (`"abc"`) or `Inf`/`-Inf`. | Replace non-numeric tokens with valid floats or empty numeric blanks (`NaN`). |
| `422 duplicate_tx_id` | Repeated `txId` rows inside transactions CSV. | Deduplicate transaction rows so each `txId` appears once. |
| `422 dangling_edges` | An edge references a `txId` absent from `transactions.csv`. | Ensure `edges.csv` only links transactions present in `transactions.csv`. |
| `422 cross_timestep_edges` | An edge links two transactions in different `Time step` snapshots. | Partition edges strictly within identical `Time step` snapshots. |
| `413 file_too_large` | Uploaded CSV exceeds the `100 MB` limit. | Upload smaller chronological snapshot subsets (e.g., 1–7 steps per batch). |
| `409 snapshot_cache_expired` | On-demand IG explanation requested after the `50`-run / `30`-day tensor cache was evicted. | Re-upload the CSV bundle to generate fresh cached snapshot tensors. |
