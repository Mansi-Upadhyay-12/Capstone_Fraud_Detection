"""Central configuration and frozen model/artifact paths for Stage 6 backend."""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_ROOT = os.path.join(PROJECT_ROOT, "app")
BACKEND_ROOT = os.path.join(APP_ROOT, "backend")

# Frozen Stage 1-4H research artifacts (READ-ONLY)
RAW_FEATURES_PATH = os.path.join(PROJECT_ROOT, "dataset", "txs_features.txt")
METADATA_PATH = os.path.join(PROJECT_ROOT, "processed", "metadata.json")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "stage4h", "ExpA2_RankGauss182_BiDirGCN_seed42.pt")
RANKGAUSS_PATH = os.path.join(PROJECT_ROOT, "models", "stage4h", "rankgauss_transformer_train_1_34.joblib")

# Stage 6 exported read-only Stage-2 preprocessing artifact
ARTIFACTS_DIR = os.path.join(BACKEND_ROOT, "artifacts")
STAGE2_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "stage2_imputer_scaler_train_1_34.joblib")

# SQLite & Ephemeral Run Tensor Cache with bounded retention
DATA_DIR = os.path.join(APP_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "fraud_app.db")
RUNS_CACHE_DIR = os.path.join(DATA_DIR, "runs_cache")
MAX_CACHED_RUNS = 50          # Prunes oldest cached tensor bundles beyond 50 runs
RETENTION_DAYS = 30           # Raw CSVs are NEVER persisted; cached run tensors expire after 30 days

# Sample CSV bundles for quick demos/tests
SAMPLE_DATA_DIR = os.path.join(APP_ROOT, "sample_data")

# Security & CSV limits
MAX_CSV_BYTES = 100 * 1024 * 1024  # 100 MB per uploaded file
MAX_NODES_PER_UPLOAD = 250_000
MAX_EDGES_PER_UPLOAD = 500_000
ALLOWED_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

# Risk band labeling (distinct from criminal confirmation; threshold comes from frozen checkpoint)
FROZEN_THRESHOLD_EXPECTED = 0.57


def load_feature_columns():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cols = meta["feature_columns"]
    if len(cols) != 182:
        raise ValueError(f"Expected 182 feature columns in metadata.json, found {len(cols)}")
    return cols
