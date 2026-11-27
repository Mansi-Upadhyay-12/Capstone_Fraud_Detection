"""Pydantic schemas for Stage 6 FastAPI backend responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_id: str
    checkpoint_path: str
    threshold_used: float
    preprocessing_order: List[str]
    device: str
    stage2_artifact_ready: bool
    rankgauss_ready: bool
    feature_count: int
    streaming_supported: bool = False


class ValidationResponse(BaseModel):
    valid: bool
    node_count: int
    edge_count: int
    unique_edges_after_dedup: int
    duplicate_edges_removed: int
    timesteps: List[int]
    imputed_nan_cells: int
    has_optional_class_column: bool
    class_counts: Dict[str, int]
    warnings: List[str]


class AnalyzeResponse(BaseModel):
    run_id: str
    dataset_name: str
    status: str
    created_at: str
    inference_time_sec: float
    threshold_used: float
    node_count: int
    edge_count: int
    timesteps: List[int]
    flagged_count: int
    flagged_rate_pct: float
    direct_clusters_count: int
    potential_candidates_count: int
    shared_bridges_count: int
    warnings: List[str]
    disclaimer: str


class RunSummaryItem(BaseModel):
    run_id: str
    dataset_name: str
    created_at: str
    node_count: int
    edge_count: int
    flagged_count: int
    direct_clusters_count: int
    potential_candidates_count: int
    threshold_used: float
    inference_time_sec: float
    has_optional_class_column: bool
