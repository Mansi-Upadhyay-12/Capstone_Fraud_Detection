"""
Stage 4F: Graph Model Performance Optimization Pipeline
Strict Zero-Leakage Protocol:
  - Train: Steps 1-34 ONLY
  - Early Stopping, Config Selection, Threshold Tuning (0.01-0.99): Validation Steps 35-41 ONLY
  - Test: Steps 42-49 SEALED until final Stage 4F champion is selected
  - Does NOT modify any Stage 1-4E scripts, checkpoints, metrics, or reports.
"""

import json
import math
import os
import sys
import tempfile
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models_nextgen import BinaryFocalLoss, build_gcn_sparse_adj
from src.models_stage4f import (
    AnchorGCNAvgWrapper,
    BiDirGATv2Net,
    BiDirGraphSAGEMeanMax,
    BiDirRelationalGCN,
    CausalCtxBiDirGCN,
    build_directed_neighbor_mean_adj,
    build_true_directed_gcn_adj,
    compute_node_deg_features,
)
from src.train_nextgen import compute_metrics_and_calibration, select_thresholds_val_only


SEEDS = [42, 123, 456]


# =============================================================================
# 1. PRE-FLIGHT TOY GRAPH, SPLIT ISOLATION & SCALING VERIFICATION
# =============================================================================

def verify_toy_directed_adjacency_and_isolated_nodes():
    """
    Verifies edge 0 -> 1 + isolated node 2 (x = [10.0, 1.0, 5.0]) on all new Stage 4F builders
    and documents Stage 4D legacy direction behavior.
    """
    ei = torch.tensor([[0], [1]], dtype=torch.long)
    x = torch.tensor([[10.0], [1.0], [5.0]])

    adj_in_mean = build_directed_neighbor_mean_adj(ei, 3, direction="in")
    adj_out_mean = build_directed_neighbor_mean_adj(ei, 3, direction="out")
    y_in_mean = torch.sparse.mm(adj_in_mean, x).squeeze().tolist()
    y_out_mean = torch.sparse.mm(adj_out_mean, x).squeeze().tolist()
    assert abs(y_in_mean[1] - 10.0) < 1e-5 and abs(y_in_mean[0]) < 1e-5 and abs(y_in_mean[2]) < 1e-5
    assert abs(y_out_mean[0] - 1.0) < 1e-5 and abs(y_out_mean[1]) < 1e-5 and abs(y_out_mean[2]) < 1e-5

    adj_in_gcn = build_true_directed_gcn_adj(ei, 3, direction="in")
    adj_out_gcn = build_true_directed_gcn_adj(ei, 3, direction="out")
    adj_undir_gcn = build_true_directed_gcn_adj(ei, 3, direction="undirected")

    d_in = adj_in_gcn.to_dense()
    d_out = adj_out_gcn.to_dense()
    d_und = adj_undir_gcn.to_dense()

    # For 0 -> 1: incoming node 1 must receive from node 0 (d_in[1, 0] > 0) and isolated node 2 keeps self-loop 1.0
    assert d_in[1, 0].item() > 0.4 and abs(d_in[2, 2].item() - 1.0) < 1e-5
    # For outgoing: node 0 must receive from node 1 (d_out[0, 1] > 0) and isolated node 2 keeps self-loop 1.0
    assert d_out[0, 1].item() > 0.4 and abs(d_out[2, 2].item() - 1.0) < 1e-5
    # For undirected: symmetric, row sums == 1.0
    assert torch.allclose(d_und, d_und.t(), atol=1e-5)
    assert torch.allclose(d_und.sum(dim=1), torch.ones(3), atol=1e-5)
    for t in (d_in, d_out, d_und):
        assert torch.isfinite(t).all()

    return {
        "toy_mean_in_0to1_isolated2": [round(v, 4) for v in y_in_mean],
        "toy_mean_out_0to1_isolated2": [round(v, 4) for v in y_out_mean],
        "toy_gcn_in_0to1_isolated2": [round(v, 4) for v in torch.sparse.mm(adj_in_gcn, x).squeeze().tolist()],
        "toy_gcn_out_0to1_isolated2": [round(v, 4) for v in torch.sparse.mm(adj_out_gcn, x).squeeze().tolist()],
        "toy_gcn_undir_0to1_isolated2": [round(v, 4) for v in torch.sparse.mm(adj_undir_gcn, x).squeeze().tolist()],
    }


def verify_chronological_splits_and_scaling(snapshots, metadata):
    """Confirms strict Train(1-34) / Val(35-41) / Test(42-49) split ordering and Train-only scaling."""
    assert len(snapshots) == 49
    train_ts = [int(s.timestep) for s in snapshots[:34]]
    val_ts = [int(s.timestep) for s in snapshots[34:41]]
    test_ts = [int(s.timestep) for s in snapshots[41:49]]
    assert train_ts == list(range(1, 35))
    assert val_ts == list(range(35, 42))
    assert test_ts == list(range(42, 50))
    assert all(str(s.split) == "train" for s in snapshots[:34])
    assert all(str(s.split) == "val" for s in snapshots[34:41])
    assert all(str(s.split) == "test" for s in snapshots[41:49])

    cols = metadata["feature_columns"]
    assert len(cols) == 182
    assert all(c.startswith("Local_feature_") for c in cols[:93])

    local_idx = list(range(0, 93))
    loc_dom_idx = list(range(0, 93)) + list(range(165, 182))
    checks = {}
    for name, idx in [("local_93", local_idx), ("local93_domain17", loc_dom_idx)]:
        x_tr = torch.cat([s.x[:, idx] for s in snapshots[:34]], dim=0).numpy()
        m_max = float(np.abs(x_tr.mean(axis=0)).max())
        s_min, s_max = float(x_tr.std(axis=0).min()), float(x_tr.std(axis=0).max())
        assert m_max < 1e-2 and 0.98 < s_min and s_max < 1.02
        checks[name] = {"train_mean_abs_max": round(m_max, 6), "train_std_range": [round(s_min, 6), round(s_max, 6)]}
    return checks


def build_feature_slices():
    return {
        "local_93": list(range(0, 93)),
        "local93_domain17": list(range(0, 93)) + list(range(165, 182)),
    }


def build_snapshot_bundle(s, drop_edge_p=0.0, device=None):
    ei = s.edge_index
    if drop_edge_p > 0.0 and ei.shape[1] > 0:
        keep = torch.rand(ei.shape[1], device=ei.device) >= drop_edge_p
        if keep.any():
            ei = ei[:, keep]
    return {
        "edge_index": ei.to(device) if device is not None else ei,
        "edge_index_rev": ei.flip(0).to(device) if device is not None else ei.flip(0),
        "legacy_dir": build_gcn_sparse_adj(ei, s.num_nodes, use_undirected=False, device=device),
        "legacy_undir": build_gcn_sparse_adj(ei, s.num_nodes, use_undirected=True, device=device),
        "gcn_in": build_true_directed_gcn_adj(ei, s.num_nodes, direction="in", device=device),
        "gcn_out": build_true_directed_gcn_adj(ei, s.num_nodes, direction="out", device=device),
        "mean_in": build_directed_neighbor_mean_adj(ei, s.num_nodes, direction="in", device=device),
        "mean_out": build_directed_neighbor_mean_adj(ei, s.num_nodes, direction="out", device=device),
        "deg_feats": compute_node_deg_features(s.edge_index, s.num_nodes, device=device),
    }


# =============================================================================
# 2. EXPERIMENT CATALOG & MODEL/LOSS FACTORIES
# =============================================================================

def build_stage4f_catalog():
    """Defines 12 controlled Phase A experiments anchored on F4F_00_Anchor_GCNAvg."""
    base_bce = {"kind": "weighted_bce"}
    return [
        {
            "exp_id": "F4F_00_Anchor_GCNAvg",
            "group": "Anchor",
            "description": "Stage 4D/4E champion anchor (Local-93, SeparatedGraphNet gcn_avg, legacy dir)",
            "arch": "AnchorGCNAvg",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"adj_key": "legacy_dir"},
        },
        {
            "exp_id": "F4F_01_Undir_GCNAvg",
            "group": "Direction_Fix",
            "description": "Same GCNAvg architecture with symmetric undirected adjacency (use_undirected=True)",
            "arch": "AnchorGCNAvg",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"adj_key": "legacy_undir"},
        },
        {
            "exp_id": "F4F_02_BiDir_InOutGCN",
            "group": "Direction_Fix",
            "description": "Decoupled [Self || True-In-GCN || True-Out-GCN] with branch LayerNorm",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {},
        },
        {
            "exp_id": "F4F_03_BiDir_SAGE_MeanMax",
            "group": "Aggregation",
            "description": "Decoupled [Self || InMean || OutMean || InMax || OutMax] + per-node channel gate",
            "arch": "BiDirGraphSAGEMeanMax",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {},
        },
        {
            "exp_id": "F4F_04_BiDir_GATv2_Res",
            "group": "Attention",
            "description": "Bidirectional Multi-Head GATv2 + Dual-LN Residual + EdgeDrop(0.1)",
            "arch": "BiDirGATv2Net",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"edge_dropout": 0.1},
        },
        {
            "exp_id": "F4F_05_BiDir_ResSwiGLU",
            "group": "Tabular_Stem",
            "description": "BiDirRelationalGCN + Pre-LN SwiGLU stem + structural log-degree features",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"stem_mode": "swiglu", "use_deg_feats": True},
        },
        {
            "exp_id": "F4F_06_Asinh_BiDirGCN",
            "group": "Tail_Robustness",
            "description": "BiDirRelationalGCN + monotonic asinh(x) tail compression on Local-93",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"use_tail_asinh": True},
        },
        {
            "exp_id": "F4F_07_Asinh_110_BiDirGCN",
            "group": "Tail_Robustness",
            "description": "BiDirRelationalGCN + asinh(x) on Local-93 + Domain-17 (110 features)",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local93_domain17",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"use_tail_asinh": True},
        },
        {
            "exp_id": "F4F_08_ModBCE_pw276_BiDir",
            "group": "Loss_Calibration",
            "description": "BiDirRelationalGCN with moderated weighted BCE (pos_weight=sqrt(7.63)=2.76)",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": {"kind": "weighted_bce", "pos_weight": 2.7631},
            "model_kwargs": {},
        },
        {
            "exp_id": "F4F_09_Focal_a050_g20_BiDir",
            "group": "Loss_Calibration",
            "description": "BiDirRelationalGCN with BinaryFocalLoss (alpha=0.50, gamma=2.0, pos=1)",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": {"kind": "focal", "alpha": 0.50, "gamma": 2.0},
            "model_kwargs": {},
        },
        {
            "exp_id": "F4F_10_Reg_DropEdge_BiDir",
            "group": "Regularization",
            "description": "BiDirRelationalGCN + DropEdge(0.15) + FeatDropout(0.10) + wd=3e-3",
            "arch": "BiDirRelationalGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.15,
            "lr": 0.002,
            "weight_decay": 3e-3,
            "loss": base_bce,
            "model_kwargs": {"feat_dropout": 0.10},
        },
        {
            "exp_id": "F4F_11_CausalCtx_BiDir",
            "group": "Temporal_Context",
            "description": "BiDirRelationalGCN + strictly causal snapshot attention (W=6, < t) & FiLM",
            "arch": "CausalCtxBiDirGCN",
            "feat_key": "local_93",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"window_size": 6, "num_heads": 4},
        },
    ]


def build_model_from_spec(spec, in_channels, device):
    arch = spec["arch"]
    kw = spec.get("model_kwargs", {})
    if arch == "AnchorGCNAvg":
        return AnchorGCNAvgWrapper(in_channels=in_channels, **kw).to(device)
    elif arch == "BiDirRelationalGCN":
        return BiDirRelationalGCN(in_channels=in_channels, **kw).to(device)
    elif arch == "BiDirGraphSAGEMeanMax":
        return BiDirGraphSAGEMeanMax(in_channels=in_channels, **kw).to(device)
    elif arch == "BiDirGATv2Net":
        return BiDirGATv2Net(in_channels=in_channels, **kw).to(device)
    elif arch == "CausalCtxBiDirGCN":
        return CausalCtxBiDirGCN(in_channels=in_channels, **kw).to(device)
    raise ValueError(f"Unknown arch: {arch}")


def build_loss_from_spec(loss_spec, default_pos_weight, device):
    if loss_spec["kind"] == "weighted_bce":
        pw = float(loss_spec.get("pos_weight", default_pos_weight))
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
    elif loss_spec["kind"] == "focal":
        return BinaryFocalLoss(alpha=loss_spec["alpha"], gamma=loss_spec["gamma"])
    raise ValueError(f"Unknown loss: {loss_spec}")


def verify_anchor_checkpoint_reproduction(base_dir, snapshots_dev, eval_bundles, y_val_all, feat_slices, device):
    """Verifies existing Stage 4D/4E anchor checkpoint reproduces Val PR-AUC 0.9051."""
    ckpt_path = os.path.join(base_dir, "models", "nextgen_gnn_best.pt")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = AnchorGCNAvgWrapper(in_channels=93, adj_key="legacy_dir").to(device)
    model.core.load_state_dict(ckpt["model_state_dict"])
    idx_tensor = torch.tensor(feat_slices["local_93"], dtype=torch.long, device=device)
    val_probs = rollout_val_only(model, snapshots_dev, eval_bundles, idx_tensor)
    pr = float(round(average_precision_score(y_val_all, val_probs), 4))
    m_val = compute_metrics_and_calibration(y_val_all, val_probs, threshold=ckpt["threshold_recal_01_99"])
    assert abs(pr - 0.9051) < 1e-4, f"Expected anchor Val PR-AUC 0.9051, got {pr}"
    assert abs(m_val["illicit_f1"] - 0.8654) < 1e-4, f"Expected anchor Val F1 0.8654, got {m_val['illicit_f1']}"
    return {"anchor_ckpt": ckpt_path, "val_pr_auc": pr, "val_f1": m_val["illicit_f1"], "verified": True}


def run_preflight_model_and_ckpt_checks(catalog, snapshots_dev, eval_bundles, feat_slices, pos_weight_val, device):
    """
    Verifies for ALL configs: tensor dimensions, finite forward/backward passes,
    non-zero gradients, and temporary-dir checkpoint save/load exact reproduction.
    """
    s0 = snapshots_dev[0]
    b0 = eval_bundles[0]
    results = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for spec in catalog:
            idx_t = torch.tensor(feat_slices[spec["feat_key"]], dtype=torch.long, device=device)
            in_ch = len(feat_slices[spec["feat_key"]])
            model = build_model_from_spec(spec, in_ch, device)
            crit = build_loss_from_spec(spec["loss"], pos_weight_val, device)

            model.train()
            model.zero_grad()
            state = model.init_state()
            logits, _, _ = model.forward_step(s0.x.index_select(1, idx_t), b0, state)
            assert logits.shape == (s0.num_nodes,), f"Shape mismatch for {spec['exp_id']}: {logits.shape}"
            assert torch.isfinite(logits).all(), f"Non-finite logits for {spec['exp_id']}"

            loss = crit(logits[s0.labeled_mask], s0.y[s0.labeled_mask].float())
            assert torch.isfinite(loss).item(), f"Non-finite loss for {spec['exp_id']}"
            loss.backward()
            grad_norm = float(sum(p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None))
            assert math.isfinite(grad_norm) and grad_norm > 0.0, f"Zero/invalid grad for {spec['exp_id']}"

            ckpt_file = os.path.join(tmpdir, f"{spec['exp_id']}_preflight.pt")
            torch.save({"state_dict": model.state_dict(), "spec": spec}, ckpt_file)
            loaded = torch.load(ckpt_file, weights_only=False, map_location=device)
            model2 = build_model_from_spec(spec, in_ch, device)
            model2.load_state_dict(loaded["state_dict"])
            model.eval(); model2.eval()
            with torch.no_grad():
                l1, _, _ = model.forward_step(s0.x.index_select(1, idx_t), b0, model.init_state())
                l2, _, _ = model2.forward_step(s0.x.index_select(1, idx_t), b0, model2.init_state())
            assert torch.allclose(l1, l2, atol=1e-6), f"Reload mismatch for {spec['exp_id']}"
            results.append({"exp_id": spec["exp_id"], "params": sum(p.numel() for p in model.parameters()), "loss": round(loss.item(), 4), "grad_sum": round(grad_norm, 4)})
    return results


# =============================================================================
# 3. ISOLATED ROLLOUTS & TRAINING ENGINE
# =============================================================================

def rollout_val_only(model, snapshots_dev, eval_bundles, idx_tensor):
    """Chronological warm-up (Steps 1-34) + Validation evaluation (Steps 35-41) ONLY."""
    model.eval()
    val_probs = []
    with torch.no_grad():
        state = model.init_state()
        for t in range(34):
            s = snapshots_dev[t]
            _, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), eval_bundles[t], state)
        for t in range(34, 41):
            s = snapshots_dev[t]
            logits, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), eval_bundles[t], state)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_probs.extend(probs[s.labeled_mask.cpu().numpy()])
    return np.array(val_probs)


def rollout_test_post_selection(model, snapshots_dev, eval_bundles, idx_tensor):
    """Strictly post-selection rollout on Test Steps 42-49 AFTER final model selection."""
    model.eval()
    test_probs = []
    step_outputs = {}
    with torch.no_grad():
        state = model.init_state()
        for t in range(41):
            s = snapshots_dev[t]
            _, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), eval_bundles[t], state)
        t0 = time.perf_counter()
        for t in range(41, 49):
            s = snapshots_dev[t]
            logits, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), eval_bundles[t], state)
            probs = torch.sigmoid(logits).cpu().numpy()
            lm = s.labeled_mask.cpu().numpy()
            y_np = s.y.cpu().numpy()
            lab_p, lab_y = probs[lm], y_np[lm]
            test_probs.extend(lab_p)
            step_outputs[int(s.timestep)] = {"probs": lab_p, "y": lab_y}
        inf_time = time.perf_counter() - t0
    return np.array(test_probs), step_outputs, inf_time


def train_single_stage4f(
    spec,
    seed,
    snapshots_dev,
    eval_bundles,
    feat_slices,
    y_val_all,
    pos_weight_val,
    device,
    max_epochs=50,
    patience=15,
    smoke_test=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    idx_list = feat_slices[spec["feat_key"]]
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    in_ch = len(idx_list)
    drop_edge_p = spec.get("drop_edge_p", 0.0)

    model = build_model_from_spec(spec, in_ch, device)
    criterion = build_loss_from_spec(spec["loss"], pos_weight_val, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_prauc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    max_grad_norm = 0.0
    total_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        state = model.init_state()
        total_loss = 0.0
        loss_steps = 0

        for t in range(34):
            s = snapshots_dev[t]
            bundle_t = (
                build_snapshot_bundle(s, drop_edge_p=drop_edge_p, device=device)
                if drop_edge_p > 0.0
                else eval_bundles[t]
            )
            logits_t, state, _ = model.forward_step(s.x.index_select(1, idx_tensor), bundle_t, state)
            if s.labeled_mask.sum() > 0:
                loss_t = criterion(logits_t[s.labeled_mask], s.y[s.labeled_mask].float())
                total_loss = total_loss + loss_t
                loss_steps += 1

        avg_loss = total_loss / max(loss_steps, 1)
        assert torch.isfinite(avg_loss).item(), f"Non-finite training loss at ep {epoch}"
        avg_loss.backward()
        pre_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0))
        max_grad_norm = max(max_grad_norm, pre_clip)
        optimizer.step()

        y_val_prob = rollout_val_only(model, snapshots_dev, eval_bundles, idx_tensor)
        val_prauc = float(average_precision_score(y_val_all, y_val_prob))
        scheduler.step(val_prauc)

        if val_prauc > best_val_prauc or best_state is None:
            best_val_prauc = val_prauc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == max_epochs:
            print(
                f"    [{spec['exp_id']} | seed={seed}] Ep {epoch:02d}/{max_epochs} | "
                f"Loss={avg_loss.item():.4f} | Val PR-AUC={val_prauc:.4f} | Grad={pre_clip:.2f}"
            )

        if patience_counter >= patience and not smoke_test:
            print(f"    Early stop at epoch {epoch} (Best Ep={best_epoch}, Val PR-AUC={best_val_prauc:.4f})")
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    y_val_best = rollout_val_only(model, snapshots_dev, eval_bundles, idx_tensor)

    tau_orig, _, _ = select_thresholds_val_only(y_val_all, y_val_best, 0.10, 0.90, 81, 0.30)
    tau_rec, _, tau_rec30 = select_thresholds_val_only(y_val_all, y_val_best, 0.01, 0.99, 99, 0.30)
    val_rec = compute_metrics_and_calibration(y_val_all, y_val_best, threshold=tau_rec)
    val_orig = compute_metrics_and_calibration(y_val_all, y_val_best, threshold=tau_orig)

    return {
        "exp_id": spec["exp_id"],
        "group": spec["group"],
        "description": spec["description"],
        "seed": int(seed),
        "spec": spec,
        "total_parameters": total_params,
        "best_epoch": int(best_epoch),
        "training_time_seconds": round(train_time, 3),
        "max_pre_clip_grad_norm": round(max_grad_norm, 3),
        "threshold_legacy_01_90": tau_orig,
        "threshold_recal_01_99": tau_rec,
        "threshold_rec30_recal": tau_rec30,
        "validation_metrics_legacy": val_orig,
        "validation_metrics_recal": val_rec,
        "best_state": best_state,
    }


def evaluate_stage4f_post_selection(run_rec, snapshots_dev, eval_bundles, feat_slices, y_test_all, device):
    """Evaluates finalized checkpoint on Test Steps 42-49 using pre-finalized Val threshold."""
    spec = run_rec["spec"]
    idx_list = feat_slices[spec["feat_key"]]
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    model = build_model_from_spec(spec, len(idx_list), device)
    model.load_state_dict(run_rec["best_state"])

    test_probs, step_outputs, inf_time = rollout_test_post_selection(model, snapshots_dev, eval_bundles, idx_tensor)
    m_rec = compute_metrics_and_calibration(y_test_all, test_probs, threshold=run_rec["threshold_recal_01_99"])
    m_orig = compute_metrics_and_calibration(y_test_all, test_probs, threshold=run_rec["threshold_legacy_01_90"])

    per_step = {}
    for step_num, info in step_outputs.items():
        ms = compute_metrics_and_calibration(info["y"], info["probs"], threshold=run_rec["threshold_recal_01_99"])
        per_step[str(step_num)] = {
            "pr_auc": ms["pr_auc"],
            "roc_auc": ms["roc_auc"],
            "illicit_f1": ms["illicit_f1"],
            "illicit_precision": ms["illicit_precision"],
            "illicit_recall": ms["illicit_recall"],
            "tp": ms["confusion_matrix"]["true_positives_caught_fraud"],
            "fp": ms["confusion_matrix"]["false_positives_false_alarm"],
            "fn": ms["confusion_matrix"]["false_negatives_missed_fraud"],
            "tn": ms["confusion_matrix"]["true_negatives_licit"],
        }

    s42_n = len(step_outputs[42]["y"])
    m_step42 = compute_metrics_and_calibration(y_test_all[:s42_n], test_probs[:s42_n], threshold=run_rec["threshold_recal_01_99"])
    m_steps43_49 = compute_metrics_and_calibration(y_test_all[s42_n:], test_probs[s42_n:], threshold=run_rec["threshold_recal_01_99"])

    return {
        "inference_time_seconds": round(inf_time, 4),
        "test_metrics_legacy_01_90": m_orig,
        "test_metrics_recal_01_99": m_rec,
        "test_step42_only": m_step42,
        "test_steps43_49_only": m_steps43_49,
        "per_snapshot_test_metrics": per_step,
    }


def summarize_multi_seed(runs, key="validation_metrics_recal"):
    def st(vals):
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
    ms = [r[key] for r in runs]
    return {
        "pr_auc": st([m["pr_auc"] for m in ms]),
        "roc_auc": st([m["roc_auc"] for m in ms]),
        "illicit_f1": st([m["illicit_f1"] for m in ms]),
        "illicit_precision": st([m["illicit_precision"] for m in ms]),
        "illicit_recall": st([m["illicit_recall"] for m in ms]),
        "brier_score": st([m["brier_score"] for m in ms]),
        "ece_10bin": st([m["ece_10bin"] for m in ms]),
        "best_epoch": st([r["best_epoch"] for r in runs]),
    }


def save_stage4f_plots(screening_runs, shortlist_summary, champion_res, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ranked = sorted(screening_runs, key=lambda r: r["validation_metrics_recal"]["pr_auc"], reverse=True)
    names = [r["exp_id"] for r in ranked][::-1]
    vals = [r["validation_metrics_recal"]["pr_auc"] for r in ranked][::-1]
    colors = ["#4a5568" if "Anchor" in n else "#2b6cb0" for n in names]
    axes[0].barh(names, vals, color=colors, edgecolor="black", alpha=0.9)
    axes[0].axvline(0.9051, color="#e53e3e", linestyle="--", linewidth=1.5, label="Stage 4D Anchor (0.9051)")
    axes[0].set_xlim(0.70, 0.96)
    axes[0].set_xlabel("Validation PR-AUC (Steps 35-41)")
    axes[0].set_title("Stage 4F Screening: Validation PR-AUC")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend()

    s_names = [s["exp_id"] for s in shortlist_summary]
    s_means = [s["val_summary"]["pr_auc"]["mean"] for s in shortlist_summary]
    s_stds = [s["val_summary"]["pr_auc"]["std"] for s in shortlist_summary]
    axes[1].bar(s_names, s_means, yerr=s_stds, capsize=5, color="#38a169", edgecolor="black", alpha=0.9)
    axes[1].axhline(0.9040, color="#e53e3e", linestyle="--", linewidth=1.5, label="Anchor Mean (0.9040)")
    axes[1].set_ylim(0.80, 0.96)
    axes[1].set_ylabel("Mean Val PR-AUC (+/- Std across 3 seeds)")
    axes[1].set_title(f"Multi-Seed Confirmation | Champ: {champion_res['exp_id']}")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(results_dir, "stage4f_optimization_comparison.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  [Saved Plot]: {out_path}")


def generate_stage4f_report(payload):
    champ = payload["stage4f_champion"]
    tm = champ["test_metrics_recal_01_99"]
    cm = tm["confusion_matrix"]
    md = f"""# Stage 4F: Graph Model Performance Optimization Report

**Generated On:** {payload['timestamp']}  
**Selection Protocol:** Train `Steps 1–34` $\\to$ Validate/Select `Steps 35–41` (PR-AUC) $\\to$ Post-Selection Test `Steps 42–49`  
**Selected Stage 4F Champion:** `{champ['exp_id']}` (`seed={champ['seed']}`, `params={champ['total_parameters']:,}`, `best_epoch={champ['best_epoch']}`)  
**Post-Selection Test Metrics (`0.01–0.99`, $\\tau^*={champ['threshold_recal_01_99']:.2f}$):** PR-AUC=`{tm['pr_auc']:.4f}`, ROC-AUC=`{tm['roc_auc']:.4f}`, F1=`{tm['illicit_f1']:.4f}`, Prec=`{tm['illicit_precision']*100:.2f}%`, Rec=`{tm['illicit_recall']*100:.2f}%`, Brier=`{tm['brier_score']:.4f}`, ECE=`{tm['ece_10bin']:.4f}`, CM=`TP={cm['true_positives_caught_fraud']}/FP={cm['false_positives_false_alarm']}/TN={cm['true_negatives_licit']}/FN={cm['false_negatives_missed_fraud']}`

---

## 1. Phase A Screening Results (`seed=42`, Validation Steps `35–41`)

| Exp ID | Group | Params | Best Ep | Val PR-AUC | Val ROC-AUC | Val F1 | Val Prec | Val Rec | Brier | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in sorted(payload["screening_results"], key=lambda x: x["validation_metrics_recal"]["pr_auc"], reverse=True):
        vm = r["validation_metrics_recal"]
        md += (
            f"| `{r['exp_id']}` | `{r['group']}` | `{r['total_parameters']:,}` | {r['best_epoch']} | "
            f"**{vm['pr_auc']:.4f}** | {vm['roc_auc']:.4f} | {vm['illicit_f1']:.4f} | "
            f"{vm['illicit_precision']*100:.2f}% | {vm['illicit_recall']*100:.2f}% | "
            f"{vm['brier_score']:.4f} | {vm['ece_10bin']:.4f} |\n"
        )

    md += "\n---\n\n## 2. Phase B Multi-Seed Shortlist Confirmation (Validation Steps `35–41`, Mean ± Std)\n\n"
    md += "| Exp ID | Seeds | Params | Best Ep | Val PR-AUC | Val ROC-AUC | Val F1 | Brier | ECE |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for s in payload["shortlist_summary"]:
        vs = s["val_summary"]
        md += (
            f"| `{s['exp_id']}` | `{s['seeds']}` | `{s['total_parameters']:,}` | "
            f"{vs['best_epoch']['mean']:.1f} ± {vs['best_epoch']['std']:.1f} | "
            f"**{vs['pr_auc']['mean']:.4f} ± {vs['pr_auc']['std']:.4f}** | "
            f"{vs['roc_auc']['mean']:.4f} ± {vs['roc_auc']['std']:.4f} | "
            f"{vs['illicit_f1']['mean']:.4f} ± {vs['illicit_f1']['std']:.4f} | "
            f"{vs['brier_score']['mean']:.4f} | {vs['ece_10bin']['mean']:.4f} |\n"
        )
    return md


# =============================================================================
# 4. MAIN PIPELINE ENTRYPOINT
# =============================================================================

def run_stage4f(smoke_test=False):
    print("=" * 88)
    mode_str = " (SMOKE TEST MODE)" if smoke_test else ""
    print(f"STAGE 4F: GRAPH MODEL PERFORMANCE OPTIMIZATION{mode_str}")
    print("=" * 88)
    t_start = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "processed")
    models_dir = os.path.join(base_dir, "models", "stage4f")
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")

    # 1. Pre-flight toy directed graph check
    toy_res = verify_toy_directed_adjacency_and_isolated_nodes()
    print(f"[Pre-Flight PASS] Toy 3-node (0->1, isolated 2) adjacency check: {toy_res}")

    with open(os.path.join(processed_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    snapshots = torch.load(os.path.join(processed_dir, "temporal_snapshots.pt"), weights_only=False)
    split_checks = verify_chronological_splits_and_scaling(snapshots, metadata)
    print(f"[Pre-Flight PASS] Split isolation & Train-only scaler check: {split_checks}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshots_dev = [s.to(device) for s in snapshots]
    eval_bundles = [build_snapshot_bundle(s, drop_edge_p=0.0, device=device) for s in snapshots]
    feat_slices = build_feature_slices()

    train_labels = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34)])
    pos_weight_val = int((train_labels == 0).sum()) / max(int((train_labels == 1).sum()), 1)
    y_val_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34, 41)])
    print(f"Train (1-34) pos_weight={pos_weight_val:.4f} | Val (35-41) N={len(y_val_all):,} | Test (42-49) SEALED.")

    # 2. Verify existing Stage 4D/4E anchor checkpoint reproduces Val PR-AUC = 0.9051
    anchor_check = verify_anchor_checkpoint_reproduction(base_dir, snapshots_dev, eval_bundles, y_val_all, feat_slices, device)
    print(f"[Pre-Flight PASS] Anchor checkpoint reproduction: {anchor_check}")

    # 3. Verify all 12 configs tensor dims, forward/backward, finite loss/grads, ckpt save/load
    catalog = build_stage4f_catalog()
    preflight_models = run_preflight_model_and_ckpt_checks(catalog, snapshots_dev, eval_bundles, feat_slices, pos_weight_val, device)
    print(f"[Pre-Flight PASS] Verified {len(preflight_models)}/{len(catalog)} configs (shapes, finite loss/grads, ckpt save/load).")

    max_epochs = 2 if smoke_test else 50
    patience = 15

    # Phase A: Controlled single-seed screening (Steps 1-34 train, Steps 35-41 validation ONLY)
    print(f"\n[Phase A] Screening {len(catalog)} configs (seed=42, max_epochs={max_epochs})...")
    screening_runs = []
    for spec in catalog:
        res = train_single_stage4f(
            spec=spec,
            seed=42,
            snapshots_dev=snapshots_dev,
            eval_bundles=eval_bundles,
            feat_slices=feat_slices,
            y_val_all=y_val_all,
            pos_weight_val=pos_weight_val,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            smoke_test=smoke_test,
        )
        screening_runs.append(res)

    # Validation-driven composed candidate check (e.g. combining top topology + moderated BCE/asinh if non-anchor wins)
    ranked_phase_a = sorted(screening_runs, key=lambda r: r["validation_metrics_recal"]["pr_auc"], reverse=True)
    top_topo = next((r for r in ranked_phase_a if r["group"] in ("Direction_Fix", "Aggregation", "Attention", "Tabular_Stem")), None)
    if top_topo is not None:
        composed_spec = {
            "exp_id": "F4F_12_Composed_TopTopo_ModBCE_Asinh",
            "group": "Validation_Composed",
            "description": f"Validation-composed {top_topo['exp_id']} + asinh + pos_weight=2.76",
            "arch": top_topo["spec"]["arch"],
            "feat_key": top_topo["spec"]["feat_key"],
            "drop_edge_p": 0.10,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": {"kind": "weighted_bce", "pos_weight": 2.7631},
            "model_kwargs": dict(top_topo["spec"]["model_kwargs"]),
        }
        if composed_spec["arch"] == "BiDirRelationalGCN":
            composed_spec["model_kwargs"]["use_tail_asinh"] = True
        print(f"\n[Phase A Composed Candidate] {composed_spec['exp_id']}")
        res_comp = train_single_stage4f(
            spec=composed_spec,
            seed=42,
            snapshots_dev=snapshots_dev,
            eval_bundles=eval_bundles,
            feat_slices=feat_slices,
            y_val_all=y_val_all,
            pos_weight_val=pos_weight_val,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            smoke_test=smoke_test,
        )
        screening_runs.append(res_comp)

    # Phase B: Multi-seed confirmation on top-4 Validation PR-AUC configs + Anchor control
    ranked_all = sorted(screening_runs, key=lambda r: r["validation_metrics_recal"]["pr_auc"], reverse=True)
    shortlist_base = ranked_all[: (2 if smoke_test else 4)]
    if all(r["exp_id"] != "F4F_00_Anchor_GCNAvg" for r in shortlist_base):
        anchor_base = next(r for r in screening_runs if r["exp_id"] == "F4F_00_Anchor_GCNAvg")
        shortlist_base.append(anchor_base)

    confirm_seeds = [42, 123] if smoke_test else SEEDS
    print(f"\n[Phase B] Multi-seed confirmation (seeds={confirm_seeds}) on {[r['exp_id'] for r in shortlist_base]}...")
    multi_seed_runs = {}
    shortlist_summary = []
    all_runs = list(screening_runs)

    for base_r in shortlist_base:
        eid = base_r["exp_id"]
        runs_eid = [base_r]
        for s_val in confirm_seeds:
            if s_val == 42:
                continue
            r_s = train_single_stage4f(
                spec=base_r["spec"],
                seed=s_val,
                snapshots_dev=snapshots_dev,
                eval_bundles=eval_bundles,
                feat_slices=feat_slices,
                y_val_all=y_val_all,
                pos_weight_val=pos_weight_val,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                smoke_test=smoke_test,
            )
            runs_eid.append(r_s)
            all_runs.append(r_s)

        multi_seed_runs[eid] = runs_eid
        shortlist_summary.append({
            "exp_id": eid,
            "seeds": confirm_seeds,
            "total_parameters": base_r["total_parameters"],
            "val_summary": summarize_multi_seed(runs_eid, "validation_metrics_recal"),
        })

    shortlist_summary.sort(key=lambda x: (x["val_summary"]["pr_auc"]["mean"], x["val_summary"]["illicit_f1"]["mean"]), reverse=True)
    champ_exp_id = shortlist_summary[0]["exp_id"]
    champ_best_run = max(multi_seed_runs[champ_exp_id], key=lambda r: r["validation_metrics_recal"]["pr_auc"])

    print("\n" + "=" * 88)
    print(
        f"[Validation-Only Champion Selected] {champ_exp_id} "
        f"(Mean Val PR-AUC={shortlist_summary[0]['val_summary']['pr_auc']['mean']:.4f}, BestSeed={champ_best_run['seed']})"
    )
    print("=" * 88)

    if smoke_test:
        print("\n[Smoke Test] Test snapshots 42-49 kept SEALED; skipping all production disk writes to protect Stage 1-4E outputs.")
        payload = {
            "stage": "Stage 4F Smoke Test",
            "smoke_test": True,
            "toy_check": toy_res,
            "split_checks": split_checks,
            "anchor_check": anchor_check,
            "preflight_models": preflight_models,
            "screening_count": len(screening_runs),
            "validation_selected_smoke_winner": champ_exp_id,
        }
    else:
        # Phase C: Strictly Post-Selection Test Evaluation on Steps 42-49
        y_test_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(41, 49)])
        champ_test = evaluate_stage4f_post_selection(champ_best_run, snapshots_dev, eval_bundles, feat_slices, y_test_all, device)
        champ_payload = {k: v for k, v in champ_best_run.items() if k != "best_state"}
        champ_payload.update(champ_test)

        # Also evaluate multi-seed shortlisted runs post-selection for mean+-std reporting
        for s_entry in shortlist_summary:
            eid = s_entry["exp_id"]
            for r in multi_seed_runs[eid]:
                r.update(evaluate_stage4f_post_selection(r, snapshots_dev, eval_bundles, feat_slices, y_test_all, device))
            s_entry["test_summary"] = summarize_multi_seed(multi_seed_runs[eid], "test_metrics_recal_01_99")

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        for r in all_runs:
            ckpt_p = os.path.join(models_dir, f"{r['exp_id']}_seed{r['seed']}.pt")
            torch.save({
                "exp_id": r["exp_id"],
                "seed": r["seed"],
                "spec": r["spec"],
                "best_epoch": r["best_epoch"],
                "total_parameters": r["total_parameters"],
                "model_state_dict": r["best_state"],
                "threshold_recal_01_99": r["threshold_recal_01_99"],
                "validation_metrics_recal": r["validation_metrics_recal"],
            }, ckpt_p)

        champ_ckpt = os.path.join(models_dir, "stage4f_champion.pt")
        torch.save({
            "role": "stage4f_champion",
            "exp_id": champ_payload["exp_id"],
            "seed": champ_payload["seed"],
            "spec": champ_payload["spec"],
            "best_epoch": champ_payload["best_epoch"],
            "total_parameters": champ_payload["total_parameters"],
            "model_state_dict": champ_best_run["best_state"],
            "threshold_recal_01_99": champ_payload["threshold_recal_01_99"],
            "threshold_legacy_01_90": champ_payload["threshold_legacy_01_90"],
            "validation_metrics_recal": champ_payload["validation_metrics_recal"],
            "test_metrics_recal_01_99": champ_payload["test_metrics_recal_01_99"],
        }, champ_ckpt)

        clean_screen = [{k: v for k, v in r.items() if k != "best_state"} for r in screening_runs]
        clean_ms = {eid: [{k: v for k, v in r.items() if k != "best_state"} for r in runs] for eid, runs in multi_seed_runs.items()}

        payload = {
            "stage": "Stage 4F - Graph Model Performance Optimization",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "smoke_test": False,
            "toy_check": toy_res,
            "split_checks": split_checks,
            "anchor_check": anchor_check,
            "preflight_models": preflight_models,
            "screening_results": clean_screen,
            "shortlist_summary": shortlist_summary,
            "multi_seed_runs": clean_ms,
            "stage4f_champion": champ_payload,
        }

        save_stage4f_plots(clean_screen, shortlist_summary, champ_payload, results_dir)
        with open(os.path.join(reports_dir, "stage4f_optimization_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        with open(os.path.join(reports_dir, "stage4f_optimization_report.md"), "w", encoding="utf-8") as f:
            f.write(generate_stage4f_report(payload))

    print(f"\nStage 4F completed in {time.time() - t_start:.2f}s.")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
    run_stage4f(smoke_test=is_smoke)
