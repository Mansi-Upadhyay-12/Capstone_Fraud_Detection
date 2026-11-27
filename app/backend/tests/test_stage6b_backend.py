"""Stage 6B backend API, SQLite persistence, XAI, validation, and Step 35 / Step 42 parity tests."""

import io
import json
import os
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import app.backend.config as config
from app.backend.database import db
from app.backend.main import app
from app.backend.ml.frozen_model import get_model_bundle
from src.models_stage4f import BiDirRelationalGCN, build_true_directed_gcn_adj


def _make_synthetic_csv_bytes(n_per_step=6, include_class=True, class_vals=None, nan_cell=False):
    feat_cols = config.load_feature_columns()
    rng = np.random.default_rng(123)
    rows = []
    # Two timesteps: 42 and 43
    tx_counter = 1000
    for step in (42, 43):
        for i in range(n_per_step):
            tx_id = tx_counter
            tx_counter += 1
            feats = rng.normal(0.0, 1.0, size=len(feat_cols)).tolist()
            row = {"txId": tx_id, "Time step": step}
            for cname, val in zip(feat_cols, feats):
                row[cname] = val
            if include_class:
                row["class"] = 3 if class_vals is None else class_vals[(step, i)]
            rows.append(row)

    df = pd.DataFrame(rows)
    if nan_cell:
        df.loc[0, feat_cols[0]] = np.nan

    # Edges strictly intra-timestep, including duplicate edge & chain/hub for clustering tests
    edge_rows = [
        {"txId1": 1000, "txId2": 1001},
        {"txId1": 1001, "txId2": 1002},
        {"txId1": 1000, "txId2": 1001},  # intentional duplicate
        {"txId1": 1000 + n_per_step, "txId2": 1001 + n_per_step},
    ]
    tx_bytes = df.to_csv(index=False).encode("utf-8")
    edge_bytes = pd.DataFrame(edge_rows).to_csv(index=False).encode("utf-8")
    return df, tx_bytes, edge_bytes


@pytest.fixture()
def isolated_client(tmp_path, monkeypatch):
    tmp_data = tmp_path / "data"
    tmp_db = tmp_data / "test_fraud_app.db"
    tmp_cache = tmp_data / "runs_cache"
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_data))
    monkeypatch.setattr(config, "DB_PATH", str(tmp_db))
    monkeypatch.setattr(config, "RUNS_CACHE_DIR", str(tmp_cache))
    monkeypatch.setattr(db, "DATA_DIR", str(tmp_data))
    monkeypatch.setattr(db, "DB_PATH", str(tmp_db))
    monkeypatch.setattr(db, "RUNS_CACHE_DIR", str(tmp_cache))
    with TestClient(app) as client:
        yield client, str(tmp_cache)


def test_health_endpoint(isolated_client):
    client, _ = isolated_client
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_id"] == "ExpA2_RankGauss182_BiDirGCN"
    assert body["threshold_used"] == pytest.approx(0.57, abs=1e-6)
    assert body["feature_count"] == 182
    assert body["stage2_artifact_ready"] is True
    assert body["rankgauss_ready"] is True
    assert body["streaming_supported"] is False


def test_validate_csv_valid_and_nan_imputation(isolated_client):
    client, _ = isolated_client
    _, tx_bytes, edge_bytes = _make_synthetic_csv_bytes(nan_cell=True)
    resp = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", tx_bytes, "text/csv"),
            "edges_file": ("edges.csv", edge_bytes, "text/csv"),
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["node_count"] == 12
    assert body["edge_count"] == 4
    assert body["unique_edges_after_dedup"] == 3
    assert body["duplicate_edges_removed"] == 1
    assert body["imputed_nan_cells"] == 1
    assert body["timesteps"] == [42, 43]
    assert body["has_optional_class_column"] is True


def test_validate_csv_missing_feature_column_422(isolated_client):
    client, _ = isolated_client
    df, _, edge_bytes = _make_synthetic_csv_bytes()
    df_bad = df.drop(columns=["Local_feature_18"])
    resp = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", df_bad.to_csv(index=False).encode("utf-8"), "text/csv"),
            "edges_file": ("edges.csv", edge_bytes, "text/csv"),
        },
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "missing_transaction_columns"
    assert "Local_feature_18" in detail["missing_sample"]


def test_validate_csv_non_numeric_and_inf_features_422(isolated_client):
    client, _ = isolated_client
    df, _, edge_bytes = _make_synthetic_csv_bytes()
    df_str = df.copy()
    df_str["Local_feature_1"] = df_str["Local_feature_1"].astype(object)
    df_str.loc[0, "Local_feature_1"] = "invalid_text"
    r1 = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", df_str.to_csv(index=False).encode("utf-8"), "text/csv"),
            "edges_file": ("edges.csv", edge_bytes, "text/csv"),
        },
    )
    assert r1.status_code == 422
    assert r1.json()["detail"]["error"] == "non_numeric_features"

    df_inf = df.copy()
    df_inf.loc[1, "Local_feature_2"] = np.inf
    r2 = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", df_inf.to_csv(index=False).encode("utf-8"), "text/csv"),
            "edges_file": ("edges.csv", edge_bytes, "text/csv"),
        },
    )
    assert r2.status_code == 422
    assert r2.json()["detail"]["error"] == "infinite_features"


def test_validate_csv_duplicate_tx_id_422(isolated_client):
    client, _ = isolated_client
    df, _, edge_bytes = _make_synthetic_csv_bytes()
    df.loc[1, "txId"] = df.loc[0, "txId"]
    resp = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", df.to_csv(index=False).encode("utf-8"), "text/csv"),
            "edges_file": ("edges.csv", edge_bytes, "text/csv"),
        },
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "duplicate_tx_id"


def test_validate_csv_dangling_and_cross_timestep_edges_422(isolated_client):
    client, _ = isolated_client
    _, tx_bytes, _ = _make_synthetic_csv_bytes()

    dangling_edges = pd.DataFrame([{"txId1": 1000, "txId2": 9999999}]).to_csv(index=False).encode("utf-8")
    r1 = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", tx_bytes, "text/csv"),
            "edges_file": ("edges.csv", dangling_edges, "text/csv"),
        },
    )
    assert r1.status_code == 422
    assert r1.json()["detail"]["error"] == "dangling_edges"
    assert r1.json()["detail"]["dangling_edge_count"] == 1

    # Edge spanning step 42 (txId 1000) and step 43 (txId 1006)
    cross_edges = pd.DataFrame([{"txId1": 1000, "txId2": 1006}]).to_csv(index=False).encode("utf-8")
    r2 = client.post(
        "/api/validate-csv",
        files={
            "transactions_file": ("tx.csv", tx_bytes, "text/csv"),
            "edges_file": ("edges.csv", cross_edges, "text/csv"),
        },
    )
    assert r2.status_code == 422
    assert r2.json()["detail"]["error"] == "cross_timestep_edges"
    assert r2.json()["detail"]["cross_timestep_edge_count"] == 1


def test_optional_class_isolation_from_inference(isolated_client):
    client, _ = isolated_client
    df_a, _, edge_bytes = _make_synthetic_csv_bytes(include_class=False)
    df_b = df_a.copy()
    df_b["class"] = 1  # flip all classes to Illicit

    ra = client.post(
        "/api/analyze",
        files={"transactions_file": ("a.csv", df_a.to_csv(index=False).encode("utf-8"), "text/csv"),
               "edges_file": ("e.csv", edge_bytes, "text/csv")},
    )
    rb = client.post(
        "/api/analyze",
        files={"transactions_file": ("b.csv", df_b.to_csv(index=False).encode("utf-8"), "text/csv"),
               "edges_file": ("e.csv", edge_bytes, "text/csv")},
    )
    assert ra.status_code == 200 and rb.status_code == 200
    run_a = ra.json()["run_id"]
    run_b = rb.json()["run_id"]

    tx_a = client.get(f"/api/runs/{run_a}/transactions").json()["items"]
    tx_b = client.get(f"/api/runs/{run_b}/transactions").json()["items"]
    assert [(t["tx_id"], t["pred_prob"], t["logit"]) for t in tx_a] == [
        (t["tx_id"], t["pred_prob"], t["logit"]) for t in tx_b
    ]


def test_end_to_end_run_queries_and_explain_cache(isolated_client):
    client, cache_dir = isolated_client
    # Use a small deterministic subset of real Step 42 data (first 300 nodes + internal edges)
    feat_cols = config.load_feature_columns()
    raw_tx = pd.read_csv(config.RAW_FEATURES_PATH)
    s42 = raw_tx[raw_tx["Time step"] == 42].sort_values(["Time step", "txId"]).head(300)
    keep_ids = set(s42["txId"].tolist())
    raw_edges = pd.read_csv(os.path.join(config.PROJECT_ROOT, "dataset", "txs_edgelist.txt"))
    e42 = raw_edges[raw_edges["txId1"].isin(keep_ids) & raw_edges["txId2"].isin(keep_ids)]

    res = client.post(
        "/api/analyze",
        data={"dataset_name": "step42_subset_300"},
        files={
            "transactions_file": ("s42_sub.csv", s42.to_csv(index=False).encode("utf-8"), "text/csv"),
            "edges_file": ("e42_sub.csv", e42.to_csv(index=False).encode("utf-8"), "text/csv"),
        },
    )
    assert res.status_code == 200
    run_id = res.json()["run_id"]

    runs = client.get("/api/runs").json()
    assert any(r["run_id"] == run_id and r["dataset_name"] == "step42_subset_300" for r in runs)

    run_detail = client.get(f"/api/runs/{run_id}").json()
    assert run_detail["node_count"] == 300
    assert run_detail["summary"]["model_id"] == "ExpA2_RankGauss182_BiDirGCN"

    tx_page = client.get(f"/api/runs/{run_id}/transactions?flagged_only=true&limit=10").json()
    assert tx_page["total"] == res.json()["flagged_count"]
    assert all(item["pred_prob"] >= 0.57 for item in tx_page["items"])

    top_tx_id = client.get(f"/api/runs/{run_id}/transactions?limit=1").json()["items"][0]["tx_id"]
    exp1 = client.post(f"/api/runs/{run_id}/transactions/{top_tx_id}/explain")
    assert exp1.status_code == 200
    b1 = exp1.json()
    assert b1["cache_hit"] is False
    assert b1["completeness"]["all_finite"] is True
    assert b1["completeness"]["abs_error"] < 0.05
    assert b1["completeness"]["rel_error_pct"] < 1.0

    exp2 = client.post(f"/api/runs/{run_id}/transactions/{top_tx_id}/explain")
    assert exp2.status_code == 200
    assert exp2.json()["cache_hit"] is True

    # Evict tensor cache -> requesting uncached tx returns clean 409
    shutil.rmtree(os.path.join(cache_dir, run_id))
    other_tx_id = client.get(f"/api/runs/{run_id}/transactions?limit=2").json()["items"][1]["tx_id"]
    exp_miss = client.post(f"/api/runs/{run_id}/transactions/{other_tx_id}/explain")
    assert exp_miss.status_code == 409
    assert exp_miss.json()["detail"]["error"] == "snapshot_cache_expired"


@pytest.mark.parametrize("target_step", [35, 42])
def test_numerical_parity_step35_and_step42_against_frozen_artifacts(isolated_client, target_step):
    """
    CRITICAL PARITY TEST:
    Uploads raw CSV slices from dataset/txs_features.txt + dataset/txs_edgelist.txt
    for full Step 35 (5,507 nodes) and Step 42 (7,140 nodes) through /api/analyze and verifies
    exact numerical agreement against processed/temporal_snapshots.pt, Stage 4H RankGauss
    transformer, and frozen ExpA2_RankGauss182_BiDirGCN checkpoint.
    """
    client, _ = isolated_client
    raw_tx = pd.read_csv(config.RAW_FEATURES_PATH)
    step_df = raw_tx[raw_tx["Time step"] == target_step].sort_values(["Time step", "txId"]).reset_index(drop=True)
    step_ids = set(step_df["txId"].tolist())

    raw_edges = pd.read_csv(os.path.join(config.PROJECT_ROOT, "dataset", "txs_edgelist.txt"))
    edge_df = raw_edges[raw_edges["txId1"].isin(step_ids) & raw_edges["txId2"].isin(step_ids)].reset_index(drop=True)

    # 1. Stage 2 feature parity vs processed/temporal_snapshots.pt[target_step - 1]
    snapshots = torch.load(os.path.join(config.PROJECT_ROOT, "processed", "temporal_snapshots.pt"), weights_only=False)
    ref_snap = snapshots[target_step - 1]
    assert int(ref_snap.timestep) == target_step
    assert int(ref_snap.num_nodes) == len(step_df)
    assert torch.equal(ref_snap.txId.cpu(), torch.tensor(step_df["txId"].to_numpy(dtype=np.int64)))

    feat_cols = config.load_feature_columns()
    bundle = get_model_bundle()
    prep = bundle.preprocess_raw_features(step_df[feat_cols].to_numpy(dtype=np.float64))
    max_s2_diff = float(np.abs(prep["x_stage2"] - ref_snap.x.cpu().numpy()).max())
    assert max_s2_diff < 1e-5, f"Step {target_step} Stage-2 diff {max_s2_diff}"

    # 2. RankGauss parity vs rankgauss_transformer_train_1_34.joblib applied to ref_snap.x
    rg_pipe = joblib.load(config.RANKGAUSS_PATH)
    ref_x_rg = rg_pipe.transform(ref_snap.x.cpu().numpy()).astype(np.float32)
    max_rg_diff = float(np.abs(prep["x_rg"] - ref_x_rg).max())
    assert max_rg_diff < 1e-5, f"Step {target_step} RankGauss diff {max_rg_diff}"

    # 3. Independent offline frozen BiDirGCN logits & probs on CPU
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    ref_model = BiDirRelationalGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3)
    ref_model.load_state_dict(ckpt["model_state_dict"])
    ref_model.eval()
    ei_cpu = ref_snap.edge_index.cpu()
    g_bundle = {
        "gcn_in": build_true_directed_gcn_adj(ei_cpu, ref_snap.num_nodes, direction="in", drop_edge_p=0.0),
        "gcn_out": build_true_directed_gcn_adj(ei_cpu, ref_snap.num_nodes, direction="out", drop_edge_p=0.0),
    }
    with torch.no_grad():
        ref_logits, _, _ = ref_model.forward_step(torch.from_numpy(ref_x_rg), g_bundle, None)
        ref_probs = torch.sigmoid(ref_logits).numpy()
        ref_logits_np = ref_logits.numpy()
    ref_preds = (ref_probs >= 0.57).astype(np.int64)

    # 4. Run full FastAPI /api/analyze end-to-end on the raw CSV upload
    res = client.post(
        "/api/analyze",
        data={"dataset_name": f"parity_step_{target_step}"},
        files={
            "transactions_file": (f"step_{target_step}_tx.csv", step_df.to_csv(index=False).encode("utf-8"), "text/csv"),
            "edges_file": (f"step_{target_step}_edges.csv", edge_df.to_csv(index=False).encode("utf-8"), "text/csv"),
        },
    )
    assert res.status_code == 200
    analyze_body = res.json()
    run_id = analyze_body["run_id"]
    assert analyze_body["node_count"] == int(ref_snap.num_nodes)
    assert analyze_body["flagged_count"] == int(ref_preds.sum())

    # Retrieve ALL transactions sorted by snapshot_node_idx via SQLite query helper
    with db.get_connection() as conn:
        rows = conn.execute(
            "SELECT tx_id, pred_prob, logit, pred_class FROM transactions WHERE run_id = ? ORDER BY snapshot_node_idx ASC",
            (run_id,),
        ).fetchall()

    api_probs = np.array([r["pred_prob"] for r in rows], dtype=np.float64)
    api_logits = np.array([r["logit"] for r in rows], dtype=np.float64)
    api_preds = np.array([r["pred_class"] for r in rows], dtype=np.int64)

    # Stored SQLite values are rounded to 6 decimal places (<= 1e-6), well below 1e-5 tolerance
    max_prob_diff = float(np.abs(api_probs - ref_probs).max())
    max_logit_diff = float(np.abs(api_logits - ref_logits_np).max())
    assert max_prob_diff < 1e-5, f"Step {target_step} max prob diff={max_prob_diff}"
    assert max_logit_diff < 1e-5, f"Step {target_step} max logit diff={max_logit_diff}"
    assert np.array_equal(api_preds, ref_preds)

    # 5. Verify Stage 5C clusters + cluster graph endpoint with internal & boundary edges
    clusters_resp = client.get(f"/api/runs/{run_id}/clusters")
    assert clusters_resp.status_code == 200
    clusters_data = clusters_resp.json()
    assert clusters_data["total"] == analyze_body["direct_clusters_count"]
    if clusters_data["clusters"]:
        top_cid = clusters_data["clusters"][0]["cluster_id"]
        g_resp = client.get(f"/api/runs/{run_id}/clusters/{top_cid}/graph")
        assert g_resp.status_code == 200
        g_body = g_resp.json()
        assert g_body["cluster"]["cluster_id"] == top_cid
        assert len(g_body["nodes"]) >= 2
        assert len(g_body["edges"]) >= 1

