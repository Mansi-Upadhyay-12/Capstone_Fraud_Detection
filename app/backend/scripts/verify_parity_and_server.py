"""
Final verification helper:
1. Measures exact numerical parity across full Step 35 & Step 42 vs frozen Stage 2 / RankGauss / BiDirGCN.
2. Starts live Uvicorn FastAPI server on 127.0.0.1:8010, probes /api/health, and terminates cleanly.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import app.backend.config as config
from app.backend.ml.frozen_model import get_model_bundle
from app.backend.services.subgraph_service import analyze_uploaded_snapshot_clusters
from src.models_stage4f import BiDirRelationalGCN, build_true_directed_gcn_adj


def print_parity_metrics() -> None:
    raw_tx = pd.read_csv(config.RAW_FEATURES_PATH)
    raw_edges = pd.read_csv(os.path.join(config.PROJECT_ROOT, "dataset", "txs_edgelist.txt"))
    snapshots = torch.load(os.path.join(config.PROJECT_ROOT, "processed", "temporal_snapshots.pt"), weights_only=False)
    rg_pipe = joblib.load(config.RANKGAUSS_PATH)
    ckpt = torch.load(config.CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    ref_model = BiDirRelationalGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3)
    ref_model.load_state_dict(ckpt["model_state_dict"])
    ref_model.eval()

    feat_cols = config.load_feature_columns()
    bundle = get_model_bundle()

    print("=== FULL SNAPSHOT PARITY METRICS ===")
    for step in (35, 42):
        step_df = raw_tx[raw_tx["Time step"] == step].sort_values(["Time step", "txId"]).reset_index(drop=True)
        step_ids = set(step_df["txId"].tolist())
        edge_df = raw_edges[raw_edges["txId1"].isin(step_ids) & raw_edges["txId2"].isin(step_ids)].reset_index(drop=True)

        ref_snap = snapshots[step - 1]
        prep = bundle.preprocess_raw_features(step_df[feat_cols].to_numpy(dtype=np.float64))
        s2_diff = float(np.abs(prep["x_stage2"] - ref_snap.x.cpu().numpy()).max())

        ref_x_rg = rg_pipe.transform(ref_snap.x.cpu().numpy()).astype(np.float32)
        rg_diff = float(np.abs(prep["x_rg"] - ref_x_rg).max())

        ei_cpu = ref_snap.edge_index.cpu()
        g_bundle = {
            "gcn_in": build_true_directed_gcn_adj(ei_cpu, ref_snap.num_nodes, direction="in", drop_edge_p=0.0),
            "gcn_out": build_true_directed_gcn_adj(ei_cpu, ref_snap.num_nodes, direction="out", drop_edge_p=0.0),
        }
        with torch.no_grad():
            ref_logits, _, _ = ref_model.forward_step(torch.from_numpy(ref_x_rg), g_bundle, None)
            ref_probs = torch.sigmoid(ref_logits).numpy()
            ref_logits_np = ref_logits.numpy()

        tx_to_local = {int(tid): i for i, tid in enumerate(step_df["txId"].tolist())}
        src_l = np.array([tx_to_local[int(s)] for s in edge_df["txId1"].tolist()], dtype=np.int64)
        dst_l = np.array([tx_to_local[int(d)] for d in edge_df["txId2"].tolist()], dtype=np.int64)
        e_tensor = torch.tensor(np.vstack([src_l, dst_l]), dtype=torch.long)

        pred_out = bundle.predict_snapshot(prep["x_rg"], e_tensor)
        prob_diff = float(np.abs(pred_out["probs"] - ref_probs).max())
        logit_diff = float(np.abs(pred_out["logits"] - ref_logits_np).max())
        pred_match = int(np.array_equal(pred_out["preds"], (ref_probs >= 0.57).astype(np.int64)))

        sub = analyze_uploaded_snapshot_clusters(
            timestep=step,
            tx_ids=step_df["txId"].to_numpy(dtype=np.int64),
            src_local=src_l,
            dst_local=dst_l,
            probs=pred_out["probs"],
            logits=pred_out["logits"],
            y_opt=ref_snap.y.cpu().numpy(),
            threshold=0.57,
        )
        print(
            f"Step {step}: nodes={len(step_df)}, edges={len(edge_df)}, "
            f"max_s2_diff={s2_diff:.2e}, max_rg_diff={rg_diff:.2e}, "
            f"max_logit_diff={logit_diff:.2e}, max_prob_diff={prob_diff:.2e}, "
            f"binary_match={bool(pred_match)}, flagged={int(pred_out['preds'].sum())}, "
            f"clusters={sub['direct_clusters_count']}, candidates={sub['potential_candidates_count']}, "
            f"bridges={sub['shared_bridges_count']}"
        )


def probe_live_uvicorn() -> None:
    print("=== LIVE UVICORN SERVER PROBE ===")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.backend.main:app", "--host", "127.0.0.1", "--port", "8010"],
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health_body = None
        for _ in range(30):
            try:
                with urllib.request.urlopen("http://127.0.0.1:8010/api/health", timeout=2) as resp:
                    health_body = json.loads(resp.read().decode("utf-8"))
                    break
            except Exception:
                time.sleep(0.5)
        if health_body is None:
            raise RuntimeError("Uvicorn /api/health failed to respond within 15s")
        print(json.dumps(health_body, indent=2))
        print(f"DB initialized at {config.DB_PATH}: exists={os.path.exists(config.DB_PATH)}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    print_parity_metrics()
    probe_live_uvicorn()
