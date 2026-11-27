"""
Frozen Stage 4H BiDirGCN & exact Stage-2 + RankGauss preprocessing singleton loader.
Strict guarantees:
  - Zero retraining / fine-tuning / recalibration.
  - Preprocessing order: Raw -> Stage 2 SimpleImputer(median) -> Stage 2 StandardScaler -> Stage 4H RankGauss Pipeline.
  - Reads threshold directly from saved checkpoint (`threshold_recal_01_99 = 0.57`).
"""

import os
import sys
import joblib
import numpy as np
import torch

from app.backend.config import (
    CHECKPOINT_PATH,
    FROZEN_THRESHOLD_EXPECTED,
    PROJECT_ROOT,
    RANKGAUSS_PATH,
    STAGE2_ARTIFACT_PATH,
    load_feature_columns,
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models_stage4f import BiDirRelationalGCN, build_true_directed_gcn_adj


def select_inference_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")


def assign_risk_category(prob: float, threshold: float) -> str:
    if prob >= 0.85:
        return "High Predicted Risk (P >= 0.85)"
    if prob >= threshold:
        return f"Elevated Risk (Flagged >= {threshold:.2f})"
    if prob >= 0.35:
        return "Moderate Monitor (0.35 - 0.57)"
    return "Low Predicted Risk (< 0.35)"


class FrozenFraudModelBundle:
    """Thread-safe read-only holder for frozen Stage 2, RankGauss, and BiDirGCN artifacts."""

    def __init__(self) -> None:
        if not os.path.exists(STAGE2_ARTIFACT_PATH):
            raise FileNotFoundError(
                f"Missing Stage-2 preprocessing artifact at {STAGE2_ARTIFACT_PATH}. "
                "Run app/backend/scripts/export_stage2_artifact.py first."
            )
        if not os.path.exists(RANKGAUSS_PATH):
            raise FileNotFoundError(f"Missing Stage 4H RankGauss artifact at {RANKGAUSS_PATH}")
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Missing frozen BiDirGCN checkpoint at {CHECKPOINT_PATH}")

        self.feature_columns = load_feature_columns()
        self.device = select_inference_device()
        self.stage2_pipe = joblib.load(STAGE2_ARTIFACT_PATH)
        self.rankgauss_pipe = joblib.load(RANKGAUSS_PATH)

        ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
        self.exp_id = str(ckpt["exp_id"])
        self.threshold = float(ckpt["threshold_recal_01_99"])
        if abs(self.threshold - FROZEN_THRESHOLD_EXPECTED) > 1e-9:
            raise ValueError(f"Checkpoint threshold mismatch: {self.threshold} vs {FROZEN_THRESHOLD_EXPECTED}")

        self.model = BiDirRelationalGCN(
            in_channels=182,
            hidden_channels=128,
            out_channels=64,
            dropout_rate=0.3,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def preprocess_raw_features(self, X_raw: np.ndarray) -> dict:
        """
        Applies exact training-order preprocessing WITHOUT fitting:
          Raw [N, 182] -> Stage 2 (SimpleImputer median + StandardScaler) -> Stage 4H RankGauss
        """
        x_in = np.asarray(X_raw, dtype=np.float32)
        if x_in.ndim != 2 or x_in.shape[1] != 182:
            raise ValueError(f"Expected feature shape [N, 182], got {x_in.shape}")

        x_stage2 = self.stage2_pipe.transform(x_in).astype(np.float32)
        if not np.isfinite(x_stage2).all():
            raise ValueError("Non-finite values produced by Stage 2 imputer/scaler")

        x_rg = self.rankgauss_pipe.transform(x_stage2).astype(np.float32)
        if not np.isfinite(x_rg).all():
            raise ValueError("Non-finite values produced by Stage 4H RankGauss pipeline")

        return {"x_stage2": x_stage2, "x_rg": x_rg}

    def build_adjacency_bundle(self, edge_index: torch.Tensor, num_nodes: int) -> dict:
        ei = edge_index.to(self.device)
        return {
            "gcn_in": build_true_directed_gcn_adj(ei, num_nodes, direction="in", device=self.device),
            "gcn_out": build_true_directed_gcn_adj(ei, num_nodes, direction="out", device=self.device),
        }

    def predict_snapshot(self, x_rg: np.ndarray, edge_index: torch.Tensor) -> dict:
        """Executes deterministic frozen inference on a single intra-timestep snapshot."""
        self.model.eval()
        num_nodes = int(x_rg.shape[0])
        x_tensor = torch.from_numpy(x_rg).to(self.device)
        bundle = self.build_adjacency_bundle(edge_index, num_nodes)
        with torch.no_grad():
            logits, _, _ = self.model.forward_step(x_tensor, bundle, None)
            probs = torch.sigmoid(logits)
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        logits_np = logits.detach().cpu().numpy().astype(np.float64)
        preds_np = (probs_np >= self.threshold).astype(np.int64)
        return {
            "logits": logits_np,
            "probs": probs_np,
            "preds": preds_np,
            "bundle_cpu": {
                "gcn_in": build_true_directed_gcn_adj(edge_index.cpu(), num_nodes, direction="in"),
                "gcn_out": build_true_directed_gcn_adj(edge_index.cpu(), num_nodes, direction="out"),
            },
        }


_BUNDLE_SINGLETON = None


def get_model_bundle() -> FrozenFraudModelBundle:
    global _BUNDLE_SINGLETON
    if _BUNDLE_SINGLETON is None:
        _BUNDLE_SINGLETON = FrozenFraudModelBundle()
    return _BUNDLE_SINGLETON
