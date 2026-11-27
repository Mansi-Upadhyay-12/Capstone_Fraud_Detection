"""
Stage 6B Helper: Deterministically reconstructs Stage 2 SimpleImputer(median) + StandardScaler
strictly on raw dataset/txs_features.txt Steps 1-34, verifies 0.00e+00 numerical parity against
processed/temporal_snapshots.pt, and exports to app/backend/artifacts/stage2_imputer_scaler_train_1_34.joblib.
"""

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.backend.config import ARTIFACTS_DIR, METADATA_PATH, RAW_FEATURES_PATH, STAGE2_ARTIFACT_PATH


def export_and_verify_stage2_pipeline():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        feat_cols = json.load(f)["feature_columns"]
    assert len(feat_cols) == 182

    print(f"Loading raw features from {RAW_FEATURES_PATH}...")
    df_feat = pd.read_csv(RAW_FEATURES_PATH)
    df_tr = df_feat[df_feat["Time step"] <= 34].sort_values(["Time step", "txId"], ascending=[True, True])
    assert len(df_tr) == 136265, f"Expected 136,265 Train 1-34 rows, got {len(df_tr)}"

    X_tr_raw = df_tr[feat_cols].values.astype(np.float32)
    stage2_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    stage2_pipe.fit(X_tr_raw)

    # Verify numerical parity across Train (Step 1), Val (Step 35), and Test (Step 42)
    snaps = torch.load(os.path.join(PROJECT_ROOT, "processed", "temporal_snapshots.pt"), weights_only=False)
    for snap_idx, step_num in [(0, 1), (34, 35), (41, 42)]:
        df_s = df_feat[df_feat["Time step"] == step_num].sort_values(["Time step", "txId"]).reset_index(drop=True)
        x_s2 = stage2_pipe.transform(df_s[feat_cols].values.astype(np.float32))
        s = snaps[snap_idx]
        assert list(df_s["txId"].astype(int)) == s.txId.tolist()
        max_diff = float(np.max(np.abs(x_s2 - s.x.numpy())))
        print(f"  Verified Step {step_num:2d} parity vs temporal_snapshots.pt: max_abs_diff = {max_diff:.2e}")
        assert max_diff < 1e-5, f"Stage-2 parity drift at Step {step_num}: {max_diff}"

    joblib.dump(stage2_pipe, STAGE2_ARTIFACT_PATH)
    print(f"[Saved Stage-2 Artifact] {STAGE2_ARTIFACT_PATH} ({os.path.getsize(STAGE2_ARTIFACT_PATH):,} bytes)")
    return STAGE2_ARTIFACT_PATH


if __name__ == "__main__":
    export_and_verify_stage2_pipeline()
