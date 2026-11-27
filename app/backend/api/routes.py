"""REST API routes for Stage 6 fraud-ring detection & explainability backend."""

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
import torch

import app.backend.config as config
from app.backend.database import db
from app.backend.ml.frozen_model import assign_risk_category, get_model_bundle
from app.backend.schemas import AnalyzeResponse, HealthResponse, RunSummaryItem, ValidationResponse
from app.backend.services.analyze_service import run_full_analysis
from app.backend.services.csv_validator import validate_csv_uploads
from app.backend.services.explainability_service import explain_transaction
from src.stage5.stage5b_individual_explanations import gt_label_str

router = APIRouter(prefix="/api", tags=["fraud-detection"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    bundle = get_model_bundle()
    return HealthResponse(
        status="ok",
        model_id=bundle.exp_id,
        checkpoint_path=config.CHECKPOINT_PATH,
        threshold_used=float(bundle.threshold),
        preprocessing_order=[
            "Stage2 SimpleImputer(median) + StandardScaler (Steps 1-34)",
            "Stage4H QuantileTransformer(normal) + StandardScaler (Steps 1-34)",
        ],
        device=str(bundle.device),
        stage2_artifact_ready=os.path.exists(config.STAGE2_ARTIFACT_PATH),
        rankgauss_ready=os.path.exists(config.RANKGAUSS_PATH),
        feature_count=len(bundle.feature_columns),
        streaming_supported=False,
    )


@router.post("/validate-csv", response_model=ValidationResponse)
async def validate_csv_endpoint(
    transactions_file: UploadFile = File(...),
    edges_file: UploadFile = File(...),
    classes_file: Optional[UploadFile] = File(None),
) -> ValidationResponse:
    tx_bytes = await transactions_file.read()
    edge_bytes = await edges_file.read()
    class_bytes = await classes_file.read() if classes_file is not None else None
    v = validate_csv_uploads(tx_bytes, edge_bytes, class_bytes)
    return ValidationResponse(
        valid=True,
        node_count=v.node_count,
        edge_count=v.edge_count_raw,
        unique_edges_after_dedup=v.unique_edges_after_dedup,
        duplicate_edges_removed=v.duplicate_edges_removed,
        timesteps=v.timesteps,
        imputed_nan_cells=v.imputed_nan_cells,
        has_optional_class_column=v.has_optional_class_column,
        class_counts=v.class_counts,
        warnings=v.warnings,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_csv_endpoint(
    transactions_file: UploadFile = File(...),
    edges_file: UploadFile = File(...),
    classes_file: Optional[UploadFile] = File(None),
    dataset_name: Optional[str] = Form(None),
) -> AnalyzeResponse:
    tx_bytes = await transactions_file.read()
    edge_bytes = await edges_file.read()
    class_bytes = await classes_file.read() if classes_file is not None else None
    resolved_name = (dataset_name or transactions_file.filename or "uploaded_dataset.csv").strip()
    return run_full_analysis(tx_bytes, edge_bytes, class_bytes, dataset_name=resolved_name)


@router.get("/runs", response_model=List[RunSummaryItem])
def list_runs_endpoint() -> List[RunSummaryItem]:
    return [RunSummaryItem(**r) for r in db.list_runs()]


@router.get("/runs/{run_id}")
def get_run_endpoint(run_id: str) -> Dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "run_not_found", "message": f"Run '{run_id}' not found."})
    return run


@router.get("/runs/{run_id}/transactions")
def get_run_transactions_endpoint(
    run_id: str,
    timestep: Optional[int] = Query(None, ge=1),
    flagged_only: bool = Query(False),
    min_prob: float = Query(0.0, ge=0.0, le=1.0),
    search_tx_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "run_not_found", "message": f"Run '{run_id}' not found."})
    return db.query_transactions(
        run_id=run_id,
        timestep=timestep,
        flagged_only=flagged_only,
        min_prob=min_prob,
        search_tx_id=search_tx_id,
        limit=limit,
        offset=offset,
    )


@router.get("/runs/{run_id}/clusters")
def get_run_clusters_endpoint(
    run_id: str,
    candidates_only: bool = Query(False),
    timestep: Optional[int] = Query(None, ge=1),
) -> Dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "run_not_found", "message": f"Run '{run_id}' not found."})
    clusters = db.query_clusters(run_id=run_id, candidates_only=candidates_only, timestep=timestep)
    return {"run_id": run_id, "total": len(clusters), "clusters": clusters}


@router.get("/runs/{run_id}/clusters/{cluster_id}/graph")
def get_cluster_graph_endpoint(run_id: str, cluster_id: str) -> Dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"error": "run_not_found", "message": f"Run '{run_id}' not found."})
    cluster = db.get_cluster(run_id, cluster_id)
    if cluster is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "cluster_not_found", "message": f"Cluster '{cluster_id}' not found in run '{run_id}'."},
        )

    data = db.get_cluster_nodes_and_edges(run_id, cluster_id)
    nodes = [dict(n, is_external_context=False) for n in data["nodes"]]
    edges = [dict(e) for e in data["edges"]]
    cluster_tx_set = {int(n["tx_id"]) for n in nodes}

    # Add 1-hop external boundary nodes/edges if snapshot tensor cache exists
    t = int(cluster["timestep"])
    snap_path = os.path.join(config.RUNS_CACHE_DIR, run_id, f"step_{t}.pt")
    if os.path.exists(snap_path):
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        tx_ids = snap["tx_ids"].numpy()
        probs = snap["probs"].numpy()
        y_opt = snap["y_opt"].numpy()
        edge_index = snap["edge_index"].numpy()
        src_arr = edge_index[0]
        dst_arr = edge_index[1]
        in_deg = {}
        out_deg = {}
        for s_i, d_i in zip(src_arr.tolist(), dst_arr.tolist()):
            out_deg[s_i] = out_deg.get(s_i, 0) + 1
            in_deg[d_i] = in_deg.get(d_i, 0) + 1

        threshold = float(run["threshold_used"])
        ext_nodes_seen = set()
        for s_i, d_i in zip(src_arr.tolist(), dst_arr.tolist()):
            s_tx = int(tx_ids[s_i])
            d_tx = int(tx_ids[d_i])
            s_in = s_tx in cluster_tx_set
            d_in = d_tx in cluster_tx_set
            if s_in != d_in:
                edge_type = "external_boundary_out" if s_in else "external_boundary_in"
                edges.append(
                    {
                        "cluster_id": cluster_id,
                        "timestep": t,
                        "src_tx_id": s_tx,
                        "dst_tx_id": d_tx,
                        "edge_type": edge_type,
                        "src_pred_prob": round(float(probs[s_i]), 6),
                        "dst_pred_prob": round(float(probs[d_i]), 6),
                    }
                )
                ext_i, ext_tx = (d_i, d_tx) if s_in else (s_i, s_tx)
                if ext_tx not in ext_nodes_seen and len(ext_nodes_seen) < 40:
                    ext_nodes_seen.add(ext_tx)
                    ep = float(probs[ext_i])
                    nodes.append(
                        {
                            "tx_id": ext_tx,
                            "timestep": t,
                            "snapshot_node_idx": int(ext_i),
                            "pred_prob": round(ep, 6),
                            "pred_class": int(ep >= threshold),
                            "pred_label": "Illicit" if ep >= threshold else "Licit",
                            "risk_category": assign_risk_category(ep, threshold),
                            "in_deg": int(in_deg.get(ext_i, 0)),
                            "out_deg": int(out_deg.get(ext_i, 0)),
                            "cluster_id": "",
                            "optional_gt_code": int(y_opt[ext_i]),
                            "optional_gt_label": gt_label_str(int(y_opt[ext_i])),
                            "is_external_context": True,
                        }
                    )

    return {"run_id": run_id, "cluster": cluster, "nodes": nodes, "edges": edges}


@router.post("/runs/{run_id}/transactions/{tx_id}/explain")
def explain_transaction_endpoint(run_id: str, tx_id: int) -> Dict[str, Any]:
    return explain_transaction(run_id, tx_id)
