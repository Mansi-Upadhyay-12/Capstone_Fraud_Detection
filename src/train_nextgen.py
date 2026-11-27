"""
Stage 4D: Next-Generation Graph Fraud Detection Experimental Suite
Implements controlled experiments 1-6 under strict chronological zero-leakage rules:
  - Train: Steps 1-34 ONLY
  - Validation (Selection, Early Stopping, Thresholds, Shortlist): Steps 35-41 ONLY
  - Test (Sealed until final champion selected): Steps 42-49 ONLY

Does NOT modify any Stage 1-4C scripts, checkpoints, or reports.
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models_nextgen import (
    build_gcn_sparse_adj,
    build_neighbor_mean_sparse_adj,
    BinaryFocalLoss,
    StabilizedEvolveGCN,
    CausalTemporalAttnGCN,
    CausalAttnEvolveGCN,
    SeparatedGraphNet,
)


# =============================================================================
# 1. VERIFICATION, CALIBRATION & METRIC UTILITIES
# =============================================================================

def verify_focal_loss_positive_class():
    """
    Unit-verifies that BinaryFocalLoss treats class y=1 (Illicit) as the positive class
    and applies higher weight alpha (for alpha > 0.5) to Illicit errors than Licit errors.
    """
    fl = BinaryFocalLoss(alpha=0.75, gamma=2.0)

    # Same-magnitude confident error: Illicit (y=1, logit=-3.0) vs Licit (y=0, logit=+3.0)
    loss_illicit_err = fl(torch.tensor([-3.0]), torch.tensor([1.0])).item()
    loss_licit_err = fl(torch.tensor([3.0]), torch.tensor([0.0])).item()
    ratio = loss_illicit_err / max(loss_licit_err, 1e-12)
    assert abs(ratio - 3.0) < 1e-4, f"Expected 3.0x illicit weighting for alpha=0.75, got {ratio:.4f}"

    # Easy illicit (logit=+4.0) must be down-weighted vs hard illicit (logit=-1.0)
    loss_easy = fl(torch.tensor([4.0]), torch.tensor([1.0])).item()
    loss_hard = fl(torch.tensor([-1.0]), torch.tensor([1.0])).item()
    assert loss_easy < loss_hard * 0.01, "Focal factor failed to down-weight easy positive!"
    return True


def compute_ece(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE) over n_bins uniform probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            acc_bin = float(y_true[mask].mean())
            conf_bin = float(y_prob[mask].mean())
            ece += (cnt / n) * abs(acc_bin - conf_bin)
    return float(round(ece, 4))


def compute_metrics_and_calibration(y_true, y_prob, threshold=0.5):
    """Computes classification, ranking, and calibration metrics (Brier, ECE, mean probs)."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    illicit_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    illicit_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    illicit_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    roc_auc = float(round(roc_auc_score(y_true, y_prob), 4)) if len(np.unique(y_true)) > 1 else None
    pr_auc = float(round(average_precision_score(y_true, y_prob), 4)) if (y_true == 1).sum() > 0 else None

    brier = float(round(brier_score_loss(y_true, y_prob), 4))
    ece = compute_ece(y_true, y_prob, n_bins=10)

    ill_mask = (y_true == 1)
    lic_mask = (y_true == 0)

    return {
        "threshold_used": float(round(threshold, 4)),
        "accuracy": float(round(acc, 4)),
        "illicit_precision": float(round(illicit_prec, 4)),
        "illicit_recall": float(round(illicit_rec, 4)),
        "illicit_f1": float(round(illicit_f1, 4)),
        "macro_f1": float(round(macro_f1, 4)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "ece_10bin": ece,
        "mean_prob_all_labeled": float(round(float(y_prob.mean()), 4)),
        "mean_prob_illicit": float(round(float(y_prob[ill_mask].mean()), 4)) if ill_mask.any() else None,
        "mean_prob_licit": float(round(float(y_prob[lic_mask].mean()), 4)) if lic_mask.any() else None,
        "confusion_matrix": {
            "true_negatives_licit": int(tn),
            "false_positives_false_alarm": int(fp),
            "false_negatives_missed_fraud": int(fn),
            "true_positives_caught_fraud": int(tp),
        },
        "total_samples": int(len(y_true)),
        "illicit_count": int(ill_mask.sum()),
        "licit_count": int(lic_mask.sum()),
    }


def select_thresholds_val_only(y_val, y_val_prob, lo=0.01, hi=0.99, steps=99, min_precision=0.30):
    """Selects F1-optimal and recall-constrained thresholds strictly on Validation."""
    grid = np.linspace(lo, hi, steps)
    best_f1_t, best_f1 = 0.50, -1.0
    best_rec_t, best_rec = None, -1.0

    for t in grid:
        tr = float(round(float(t), 2))
        preds = (y_val_prob >= tr).astype(int)
        prec = precision_score(y_val, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_val, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = tr
        if prec >= min_precision and rec > best_rec:
            best_rec = rec
            best_rec_t = tr

    if best_rec_t is None:
        best_rec_t = best_f1_t
    return best_f1_t, float(round(best_f1, 4)), best_rec_t


# =============================================================================
# 2. EXPERIMENT 2: REGIME-SHIFT DIAGNOSTICS (TRAIN/VAL FIRST; TEST POST-SELECTION)
# =============================================================================

def profile_snapshot_statistics(snapshots, start_idx, end_idx):
    """Profiles per-snapshot feature norms, degrees, and label ratios without touching models."""
    stats = {}
    for idx in range(start_idx, end_idx):
        s = snapshots[idx]
        x = s.x
        y = s.y.numpy()
        lm = s.labeled_mask.numpy()
        lic = (y == 0)
        ill = (y == 1)
        unl = (y == -1)

        ei = s.edge_index.numpy()
        in_deg = np.bincount(ei[1], minlength=s.num_nodes) if ei.shape[1] > 0 else np.zeros(s.num_nodes)
        out_deg = np.bincount(ei[0], minlength=s.num_nodes) if ei.shape[1] > 0 else np.zeros(s.num_nodes)

        stats[str(int(s.timestep))] = {
            "split": str(s.split),
            "num_nodes": int(s.num_nodes),
            "num_edges": int(s.edge_index.shape[1]),
            "labeled_count": int(lm.sum()),
            "illicit_count": int(ill.sum()),
            "unlabeled_pct": round(float(unl.mean() * 100.0), 2),
            "x_norm_mean_all": round(float(x.norm(dim=1).mean().item()), 3),
            "x_norm_mean_licit": round(float(x[lic].norm(dim=1).mean().item()), 3) if lic.any() else None,
            "x_norm_mean_unlabeled": round(float(x[unl].norm(dim=1).mean().item()), 3) if unl.any() else None,
            "licit_in_deg_mean": round(float(in_deg[lic].mean()), 3) if lic.any() else None,
            "licit_out_deg_mean": round(float(out_deg[lic].mean()), 3) if lic.any() else None,
        }
    return stats


# =============================================================================
# 3. MODEL & LOSS FACTORIES + ISOLATED ROLLOUTS
# =============================================================================

def build_model_from_spec(spec, in_channels, device):
    arch = spec["arch"]
    params = spec.get("model_kwargs", {})
    if arch == "StabilizedEvolveGCN":
        model = StabilizedEvolveGCN(in_channels=in_channels, **params)
    elif arch == "CausalTemporalAttnGCN":
        model = CausalTemporalAttnGCN(in_channels=in_channels, **params)
    elif arch == "CausalAttnEvolveGCN":
        model = CausalAttnEvolveGCN(in_channels=in_channels, **params)
    elif arch == "SeparatedGraphNet":
        model = SeparatedGraphNet(in_channels=in_channels, **params)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model.to(device)


def build_loss_from_spec(loss_spec, default_pos_weight, device):
    kind = loss_spec["kind"]
    if kind == "weighted_bce":
        pw = float(loss_spec.get("pos_weight", default_pos_weight))
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
    elif kind == "focal":
        return BinaryFocalLoss(alpha=loss_spec["alpha"], gamma=loss_spec["gamma"])
    else:
        raise ValueError(f"Unknown loss kind: {kind}")


def get_slice_x(x, feat_mode):
    return x[:, :93] if feat_mode == "local_93" else x


def rollout_val_only(model, snapshots_dev, adjs_dev, feat_mode="full_182"):
    """
    Runs deterministic chronological rollout strictly on Steps 1-34 (warm-up)
    and Steps 35-41 (Validation). NEVER touches Steps 42-49.
    """
    model.eval()
    val_probs = []
    val_step_telemetry = {}
    unlabeled_ge90_list = []

    with torch.no_grad():
        state = model.init_state()
        for t in range(34):
            s = snapshots_dev[t]
            _, state, _ = model.forward_step(get_slice_x(s.x, feat_mode), adjs_dev[t], state)

        for t in range(34, 41):
            s = snapshots_dev[t]
            logits, state, telem = model.forward_step(get_slice_x(s.x, feat_mode), adjs_dev[t], state)
            all_p = torch.sigmoid(logits).cpu().numpy()
            lm = s.labeled_mask.cpu().numpy()
            y_np = s.y.cpu().numpy()

            lab_p = all_p[lm]
            lab_y = y_np[lm]
            unl_p = all_p[y_np == -1]
            unl_ge90 = float((unl_p >= 0.90).mean()) if len(unl_p) > 0 else 0.0
            unlabeled_ge90_list.append(unl_ge90)

            step_prauc = float(round(average_precision_score(lab_y, lab_p), 4)) if (lab_y == 1).sum() > 0 else None
            val_probs.extend(lab_p)
            val_step_telemetry[str(int(s.timestep))] = {
                "val_step_pr_auc": step_prauc,
                "unlabeled_pct_ge_0.90": round(unl_ge90 * 100.0, 2),
                "labeled_median_prob": round(float(np.median(lab_p)), 4),
                **telem,
            }

    return np.array(val_probs), val_step_telemetry, float(round(float(np.mean(unlabeled_ge90_list)) * 100.0, 2))


def rollout_test_post_selection(model, snapshots_dev, adjs_dev, feat_mode="full_182"):
    """
    Strictly post-selection deterministic rollout on Test Steps 42-49.
    Called ONLY for the final selected champion after Validation Steps 35-41 selection.
    """
    model.eval()
    test_probs = []
    step_details = {}

    with torch.no_grad():
        state = model.init_state()
        for t in range(41):
            s = snapshots_dev[t]
            _, state, _ = model.forward_step(get_slice_x(s.x, feat_mode), adjs_dev[t], state)

        t_inf0 = time.perf_counter()
        for t in range(41, 49):
            s = snapshots_dev[t]
            logits, state, telem = model.forward_step(get_slice_x(s.x, feat_mode), adjs_dev[t], state)
            all_p = torch.sigmoid(logits).cpu().numpy()
            lm = s.labeled_mask.cpu().numpy()
            y_np = s.y.cpu().numpy()

            lab_p = all_p[lm]
            lab_y = y_np[lm]
            unl_p = all_p[y_np == -1]

            test_probs.extend(lab_p)
            step_details[int(s.timestep)] = {
                "probs": lab_p,
                "y": lab_y,
                "all_probs": all_p,
                "y_all": y_np,
                "labeled_mean_prob": round(float(lab_p.mean()), 4),
                "labeled_median_prob": round(float(np.median(lab_p)), 4),
                "unlabeled_pct_ge_0.90": round(float((unl_p >= 0.90).mean() * 100.0), 2) if len(unl_p) > 0 else 0.0,
                **telem,
            }
        inf_time = time.perf_counter() - t_inf0

    return np.array(test_probs), step_details, inf_time


# =============================================================================
# 4. SINGLE-RUN TRAIN + VALIDATION ENGINE (ZERO TEST ACCESS)
# =============================================================================

def train_single_experiment(
    spec,
    seed,
    snapshots_dev,
    adj_cache,
    y_val_all,
    default_pos_weight,
    device,
    max_epochs=60,
    patience=15,
    smoke_test=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    feat_mode = spec.get("feat_mode", "full_182")
    adj_kind = spec.get("adj_kind", "gcn_directed")
    drop_edge_p = spec.get("drop_edge_p", 0.0)
    lr = spec.get("lr", 0.002)
    weight_decay = spec.get("weight_decay", 1e-4)
    grad_clip = spec.get("grad_clip", 2.0)

    in_channels = 93 if feat_mode == "local_93" else 182
    model = build_model_from_spec(spec, in_channels=in_channels, device=device)
    criterion = build_loss_from_spec(spec["loss"], default_pos_weight, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    eval_adjs = adj_cache[adj_kind]

    best_val_prauc = -1.0
    best_epoch = 0
    best_state = None
    best_telem_snap = None
    patience_counter = 0
    max_grad_norm_seen = 0.0

    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()

        state = model.init_state()
        total_loss = 0.0
        loss_steps = 0
        last_telem = None

        for t in range(34):
            s = snapshots_dev[t]
            if drop_edge_p > 0.0:
                use_undir = "undirected" in adj_kind
                if adj_kind.startswith("sage"):
                    adj_t = build_neighbor_mean_sparse_adj(s.edge_index, s.num_nodes, use_undirected=use_undir, drop_edge_p=drop_edge_p, device=device)
                else:
                    adj_t = build_gcn_sparse_adj(s.edge_index, s.num_nodes, use_undirected=use_undir, drop_edge_p=drop_edge_p, device=device)
            else:
                adj_t = eval_adjs[t]

            logits_t, state, telem = model.forward_step(get_slice_x(s.x, feat_mode), adj_t, state)
            last_telem = telem
            if s.labeled_mask.sum() > 0:
                loss_t = criterion(logits_t[s.labeled_mask], s.y[s.labeled_mask].float())
                total_loss = total_loss + loss_t
                loss_steps += 1

        avg_loss = total_loss / max(loss_steps, 1)
        avg_loss.backward()
        pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip))
        max_grad_norm_seen = max(max_grad_norm_seen, pre_clip_norm)
        optimizer.step()

        y_val_prob_ep, _, _ = rollout_val_only(model, snapshots_dev, eval_adjs, feat_mode=feat_mode)
        val_prauc = float(average_precision_score(y_val_all, y_val_prob_ep)) if (y_val_all == 1).sum() > 0 else 0.0
        scheduler.step(val_prauc)

        if val_prauc > best_val_prauc or best_state is None:
            best_val_prauc = val_prauc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_telem_snap = last_telem
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
            print(
                f"    [{spec['exp_id']} | seed={seed}] Ep {epoch:02d}/{max_epochs} | Loss={avg_loss.item():.4f} | "
                f"Val PR-AUC={val_prauc:.4f} | Grad={pre_clip_norm:.2f} | W1={last_telem['w1_norm']:.2f}"
            )

        if patience_counter >= patience and not smoke_test:
            print(f"    Early stop at epoch {epoch} (Best Epoch: {best_epoch}, Best Val PR-AUC: {best_val_prauc:.4f})")
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    y_val_prob_best, val_step_telem, unl_ge90_pct = rollout_val_only(
        model, snapshots_dev, eval_adjs, feat_mode=feat_mode
    )

    tau_01_90, _, tau_rec30_legacy = select_thresholds_val_only(y_val_all, y_val_prob_best, 0.10, 0.90, 81, 0.30)
    tau_01_99, _, tau_rec30_recal = select_thresholds_val_only(y_val_all, y_val_prob_best, 0.01, 0.99, 99, 0.30)

    val_metrics_legacy = compute_metrics_and_calibration(y_val_all, y_val_prob_best, threshold=tau_01_90)
    val_metrics_recal = compute_metrics_and_calibration(y_val_all, y_val_prob_best, threshold=tau_01_99)

    step_praucs = [v["val_step_pr_auc"] for v in val_step_telem.values() if v["val_step_pr_auc"] is not None]

    return {
        "exp_id": spec["exp_id"],
        "group": spec["group"],
        "description": spec["description"],
        "seed": int(seed),
        "spec": spec,
        "total_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "best_epoch": int(best_epoch),
        "training_time_seconds": round(train_time, 3),
        "max_pre_clip_grad_norm": round(max_grad_norm_seen, 3),
        "train_end_telemetry": best_telem_snap,
        "val_unlabeled_pct_ge_0.90": unl_ge90_pct,
        "val_step_prauc_min": min(step_praucs) if step_praucs else None,
        "val_step_prauc_std": round(float(np.std(step_praucs)), 4) if step_praucs else None,
        "val_step_telemetry": val_step_telem,
        "threshold_legacy_01_90": tau_01_90,
        "threshold_recal_01_99": tau_01_99,
        "threshold_rec30_recal": tau_rec30_recal,
        "validation_metrics_legacy": val_metrics_legacy,
        "validation_metrics_recal": val_metrics_recal,
        "best_state": best_state,
    }


# =============================================================================
# 5. EXPERIMENT CATALOG (EXPERIMENTS 1, 3, 4, 5, 6)
# =============================================================================

def build_experiment_catalog():
    """Defines controlled single-variable experiments aligned with the research plan."""
    base_bce = {"kind": "weighted_bce"}
    return [
        # --- Experiment 1: Stabilizing EvolveGCN-H ---
        {
            "exp_id": "E1_Stab_EvolveGCNH_DualLN_Bound",
            "group": "Exp1_Stabilize_EvolveGCNH",
            "description": "Stabilized EvolveGCN-H (Dual-Branch LN, Gated Skip, Frobenius<=15, anchor=0.75)",
            "arch": "StabilizedEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"variant": "H", "residual_mode": "gated", "max_fro_norm": 15.0, "anchor_alpha": 0.75},
        },
        {
            "exp_id": "E1_Stab_EvolveGCNH_DropEdge",
            "group": "Exp1_Stabilize_EvolveGCNH",
            "description": "Stabilized EvolveGCN-H + DropEdge(0.15) + weight_decay=5e-3",
            "arch": "StabilizedEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.15,
            "lr": 0.002,
            "weight_decay": 5e-3,
            "loss": base_bce,
            "model_kwargs": {"variant": "H", "residual_mode": "gated", "max_fro_norm": 12.0, "anchor_alpha": 0.50},
        },
        # --- Experiment 3: EvolveGCN-O ---
        {
            "exp_id": "E3_Stab_EvolveGCNO",
            "group": "Exp3_EvolveGCNO",
            "description": "Stabilized EvolveGCN-O (Weight-only GRU evolution, independent of H_t)",
            "arch": "StabilizedEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"variant": "O", "residual_mode": "gated", "max_fro_norm": 15.0, "anchor_alpha": 0.75},
        },
        # --- Experiment 4: Strictly Causal Temporal Attention ---
        {
            "exp_id": "E4_CausalTemporalAttnGCN",
            "group": "Exp4_Temporal_Attention",
            "description": "Causal Snapshot-Context Multi-Head Attention (W=8) + FiLM Modulation",
            "arch": "CausalTemporalAttnGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"window_size": 8, "num_heads": 4, "residual_mode": "gated"},
        },
        {
            "exp_id": "E4_CausalAttnEvolveGCN",
            "group": "Exp4_Temporal_Attention",
            "description": "Causal Attentive Weight-Bank EvolveGCN-H (W=6)",
            "arch": "CausalAttnEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"window_size": 6, "max_fro_norm": 15.0, "residual_mode": "gated"},
        },
        # --- Experiment 5: Weighted BCE vs. Focal Loss (on Stabilized EvolveGCN-H) ---
        {
            "exp_id": "E5_Loss_ModeratedBCE_pw2.76",
            "group": "Exp5_Loss_Comparison",
            "description": "Stabilized EvolveGCN-H with Moderated Weighted BCE (pos_weight=sqrt(7.63)=2.76)",
            "arch": "StabilizedEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": {"kind": "weighted_bce", "pos_weight": 2.7631},
            "model_kwargs": {"variant": "H", "residual_mode": "gated", "max_fro_norm": 15.0, "anchor_alpha": 0.75},
        },
        {
            "exp_id": "E5_Loss_Focal_a0.75_g2.0",
            "group": "Exp5_Loss_Comparison",
            "description": "Stabilized EvolveGCN-H with Binary Focal Loss (alpha=0.75, gamma=2.0, pos=1)",
            "arch": "StabilizedEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": {"kind": "focal", "alpha": 0.75, "gamma": 2.0},
            "model_kwargs": {"variant": "H", "residual_mode": "gated", "max_fro_norm": 15.0, "anchor_alpha": 0.75},
        },
        {
            "exp_id": "E5_Loss_Focal_a0.50_g2.0",
            "group": "Exp5_Loss_Comparison",
            "description": "Stabilized EvolveGCN-H with Balanced Focal Loss (alpha=0.50, gamma=2.0, pos=1)",
            "arch": "StabilizedEvolveGCN",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": {"kind": "focal", "alpha": 0.50, "gamma": 2.0},
            "model_kwargs": {"variant": "H", "residual_mode": "gated", "max_fro_norm": 15.0, "anchor_alpha": 0.75},
        },
        # --- Experiment 6: Value-Added of Graph Message Passing (Local-93 vs Full-182) ---
        {
            "exp_id": "E6_Local93_NoGraph",
            "group": "Exp6_Graph_Value_Test",
            "description": "Local-93 features ONLY, No Graph (A=I MLP equivalent)",
            "arch": "SeparatedGraphNet",
            "feat_mode": "local_93",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"graph_mode": "no_graph"},
        },
        {
            "exp_id": "E6_Local93_GCNAvg",
            "group": "Exp6_Graph_Value_Test",
            "description": "Local-93 features + Directed GCN Averaging",
            "arch": "SeparatedGraphNet",
            "feat_mode": "local_93",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"graph_mode": "gcn_avg"},
        },
        {
            "exp_id": "E6_Local93_SAGEConcat",
            "group": "Exp6_Graph_Value_Test",
            "description": "Local-93 features + Separated Self||Neighbor GraphSAGE Concat",
            "arch": "SeparatedGraphNet",
            "feat_mode": "local_93",
            "adj_kind": "sage_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"graph_mode": "sage_concat"},
        },
        {
            "exp_id": "E6_Full182_NoGraph",
            "group": "Exp6_Graph_Value_Test",
            "description": "Full-182 features, No Graph (A=I MLP equivalent)",
            "arch": "SeparatedGraphNet",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"graph_mode": "no_graph"},
        },
        {
            "exp_id": "E6_Full182_GCNAvg",
            "group": "Exp6_Graph_Value_Test",
            "description": "Full-182 features + Directed GCN Averaging",
            "arch": "SeparatedGraphNet",
            "feat_mode": "full_182",
            "adj_kind": "gcn_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"graph_mode": "gcn_avg"},
        },
        {
            "exp_id": "E6_Full182_SAGEConcat",
            "group": "Exp6_Graph_Value_Test",
            "description": "Full-182 features + Separated Self||Neighbor GraphSAGE Concat",
            "arch": "SeparatedGraphNet",
            "feat_mode": "full_182",
            "adj_kind": "sage_directed",
            "drop_edge_p": 0.0,
            "lr": 0.002,
            "weight_decay": 1e-3,
            "loss": base_bce,
            "model_kwargs": {"graph_mode": "sage_concat"},
        },
    ]


# =============================================================================
# 6. REPORTING & PLOTTING HELPERS
# =============================================================================

def is_nograph_spec(spec):
    return spec.get("model_kwargs", {}).get("graph_mode") == "no_graph"


def evaluate_champion_on_test(best_seed_run, role_label, snapshots_dev, adj_cache, y_test_all, device):
    """Post-selection deterministic Test evaluation (Steps 42-49) shared identically across champions."""
    spec = best_seed_run["spec"]
    feat_mode = spec.get("feat_mode", "full_182")
    in_dim = 93 if feat_mode == "local_93" else 182
    model = build_model_from_spec(spec, in_channels=in_dim, device=device)
    model.load_state_dict(best_seed_run["best_state"])

    test_probs, test_step_details, inf_time = rollout_test_post_selection(
        model, snapshots_dev, adj_cache[spec.get("adj_kind", "gcn_directed")], feat_mode=feat_mode
    )

    test_metrics_legacy = compute_metrics_and_calibration(y_test_all, test_probs, threshold=best_seed_run["threshold_legacy_01_90"])
    test_metrics_recal = compute_metrics_and_calibration(y_test_all, test_probs, threshold=best_seed_run["threshold_recal_01_99"])
    test_metrics_rec30 = compute_metrics_and_calibration(y_test_all, test_probs, threshold=best_seed_run["threshold_rec30_recal"])

    per_step_test = {}
    for step, info in test_step_details.items():
        ms = compute_metrics_and_calibration(info["y"], info["probs"], threshold=best_seed_run["threshold_recal_01_99"])
        per_step_test[str(step)] = {
            "labeled_mean_prob": info["labeled_mean_prob"],
            "labeled_median_prob": info["labeled_median_prob"],
            "unlabeled_pct_ge_0.90": info["unlabeled_pct_ge_0.90"],
            "w1_norm": info["w1_norm"],
            "w2_norm": info["w2_norm"],
            "h1_conv_norm": info["h1_conv_norm"],
            "h1_skip_norm": info["h1_skip_norm"],
            "illicit_precision": ms["illicit_precision"],
            "illicit_recall": ms["illicit_recall"],
            "illicit_f1": ms["illicit_f1"],
            "pr_auc": ms["pr_auc"],
            "roc_auc": ms["roc_auc"],
            "tp": ms["confusion_matrix"]["true_positives_caught_fraud"],
            "fp": ms["confusion_matrix"]["false_positives_false_alarm"],
            "fn": ms["confusion_matrix"]["false_negatives_missed_fraud"],
            "tn": ms["confusion_matrix"]["true_negatives_licit"],
        }

    print(
        f"[Post-Selection Test | {role_label}] {best_seed_run['exp_id']} (seed={best_seed_run['seed']}): "
        f"PR-AUC={test_metrics_recal['pr_auc']:.4f}, ROC-AUC={test_metrics_recal['roc_auc']:.4f} | "
        f"F1(0.99 tau*={best_seed_run['threshold_recal_01_99']:.2f})={test_metrics_recal['illicit_f1']:.4f} "
        f"(Prec={test_metrics_recal['illicit_precision']:.4f}, Rec={test_metrics_recal['illicit_recall']:.4f}, "
        f"Brier={test_metrics_recal['brier_score']:.4f}, ECE={test_metrics_recal['ece_10bin']:.4f}, "
        f"TP={test_metrics_recal['confusion_matrix']['true_positives_caught_fraud']}, "
        f"FP={test_metrics_recal['confusion_matrix']['false_positives_false_alarm']})"
    )

    return {
        "role": role_label,
        "exp_id": best_seed_run["exp_id"],
        "is_graph_model": not is_nograph_spec(spec),
        "best_seed": int(best_seed_run["seed"]),
        "spec": spec,
        "inference_time_seconds": round(inf_time, 4),
        "threshold_legacy_01_90": best_seed_run["threshold_legacy_01_90"],
        "threshold_recal_01_99": best_seed_run["threshold_recal_01_99"],
        "threshold_rec30_recal": best_seed_run["threshold_rec30_recal"],
        "validation_metrics_legacy": best_seed_run["validation_metrics_legacy"],
        "validation_metrics_recal": best_seed_run["validation_metrics_recal"],
        "test_metrics_legacy_01_90": test_metrics_legacy,
        "test_metrics_recal_01_99": test_metrics_recal,
        "test_metrics_rec30": test_metrics_rec30,
        "per_snapshot_test_metrics": per_step_test,
    }


def compute_graph_vs_nograph_delta(graph_champ, nograph_champ):
    """Quantifies exact test-set differences (Graph Champion - No-Graph Control)."""
    gm = graph_champ["test_metrics_recal_01_99"]
    nm = nograph_champ["test_metrics_recal_01_99"]
    gcm = gm["confusion_matrix"]
    ncm = nm["confusion_matrix"]
    return {
        "graph_exp_id": graph_champ["exp_id"],
        "nograph_exp_id": nograph_champ["exp_id"],
        "delta_pr_auc": round((gm["pr_auc"] or 0.0) - (nm["pr_auc"] or 0.0), 4),
        "delta_roc_auc": round((gm["roc_auc"] or 0.0) - (nm["roc_auc"] or 0.0), 4),
        "delta_f1_01_99": round(gm["illicit_f1"] - nm["illicit_f1"], 4),
        "delta_precision_01_99": round(gm["illicit_precision"] - nm["illicit_precision"], 4),
        "delta_recall_01_99": round(gm["illicit_recall"] - nm["illicit_recall"], 4),
        "delta_brier": round(gm["brier_score"] - nm["brier_score"], 4),
        "delta_ece": round(gm["ece_10bin"] - nm["ece_10bin"], 4),
        "delta_tp": int(gcm["true_positives_caught_fraud"] - ncm["true_positives_caught_fraud"]),
        "delta_fp": int(gcm["false_positives_false_alarm"] - ncm["false_positives_false_alarm"]),
        "delta_tn": int(gcm["true_negatives_licit"] - ncm["true_negatives_licit"]),
        "delta_fn": int(gcm["false_negatives_missed_fraud"] - ncm["false_negatives_missed_fraud"]),
        "graph_outperforms_on_pr_auc": bool((gm["pr_auc"] or 0.0) > (nm["pr_auc"] or 0.0)),
        "graph_outperforms_on_f1": bool(gm["illicit_f1"] > nm["illicit_f1"]),
    }


def save_nextgen_plots(screening_results, shortlist_summary, graph_champ, nograph_champ, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # 1. Validation PR-AUC across all screening experiments (colored by Graph vs No-Graph)
    ax0 = axes[0]
    sorted_res = sorted(screening_results, key=lambda r: r["validation_metrics_recal"]["pr_auc"] or 0.0)
    names = [r["exp_id"] for r in sorted_res]
    val_pr = [r["validation_metrics_recal"]["pr_auc"] or 0.0 for r in sorted_res]
    bar_cols = ["#7f7f7f" if is_nograph_spec(r["spec"]) else "#1f77b4" for r in sorted_res]
    ax0.barh(names, val_pr, color=bar_cols, alpha=0.88, edgecolor="black")
    ax0.axvline(x=0.6387, color="#d62728", linestyle="--", lw=1.6, label="Stage 4C Baseline (Val PR-AUC=0.6387)")
    ax0.set_title("Validation PR-AUC: Graph Models (Blue) vs No-Graph Controls (Gray)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("Validation PR-AUC (Steps 35-41)")
    ax0.legend(loc="lower right", fontsize=8.5)
    ax0.grid(axis="x", alpha=0.3)

    # 2. Multi-seed shortlist comparison (Graph vs No-Graph Control)
    ax1 = axes[1]
    s_names = [s["exp_id"] for s in shortlist_summary]
    s_means = [s["val_prauc_mean"] for s in shortlist_summary]
    s_stds = [s["val_prauc_std"] for s in shortlist_summary]
    s_cols = ["#7f7f7f" if s["is_nograph"] else "#2ca02c" for s in shortlist_summary]
    ax1.bar(s_names, s_means, yerr=s_stds, capsize=6, color=s_cols, alpha=0.88, edgecolor="black")
    ax1.axhline(y=0.6387, color="#d62728", linestyle="--", lw=1.5, label="Stage 4C Baseline (0.6387)")
    ax1.set_title(
        f"Multi-Seed Val PR-AUC\nGraph Champ: {graph_champ['exp_id']} | Control: {nograph_champ['exp_id']}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_ylabel("Mean Validation PR-AUC (+/- Std)")
    ax1.tick_params(axis="x", rotation=20)
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "stage4d_nextgen_comparison.png")
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved Stage 4D Plot]: {out_path}")


def generate_nextgen_report(payload):
    g_champ = payload["graph_champion"]
    n_champ = payload["nograph_control_champion"]
    overall = payload["overall_champion"]
    delta = payload["graph_vs_nograph_delta"]

    md = f"""# Stage 4D: Next-Generation Graph Fraud Detection Experimental Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** {payload['timestamp']}  
**Selection Protocol:** Train Steps `1–34` $\\to$ Validate/Select Steps `35–41` (PR-AUC) $\\to$ Post-Selection Test Steps `42–49`  
**Overall Champion (Across Both Groups, by Multi-Seed Val PR-AUC):** `{overall['exp_id']}` (`{overall['category']}`, Mean Val PR-AUC = `{overall['val_prauc_mean']:.4f}`)  
**Best Graph-Based Champion:** `{g_champ['exp_id']}` (Best Seed `{g_champ['best_seed']}`)  
**Best No-Graph Control:** `{n_champ['exp_id']}` (Best Seed `{n_champ['best_seed']}`)  

---

## 1. Screening Results Across Controlled Experiments (Validation Steps `35–41`)

| Exp ID | Type | Group | Best Ep | Val PR-AUC | Val F1 (`0.10–0.90`) | Val F1 (`0.01–0.99`) | Val Prec | Val Rec | Brier | ECE | Unl $\\ge 0.90$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in sorted(payload["screening_results"], key=lambda x: x["validation_metrics_recal"]["pr_auc"] or 0.0, reverse=True):
        vr = r["validation_metrics_recal"]
        vl = r["validation_metrics_legacy"]
        kind_tag = "No-Graph" if is_nograph_spec(r["spec"]) else "Graph"
        md += (
            f"| `{r['exp_id']}` | {kind_tag} | `{r['group']}` | {r['best_epoch']} | **{vr['pr_auc']:.4f}** | "
            f"{vl['illicit_f1']:.4f} | {vr['illicit_f1']:.4f} | {vr['illicit_precision']*100:.2f}% | "
            f"{vr['illicit_recall']*100:.2f}% | {vr['brier_score']:.4f} | {vr['ece_10bin']:.4f} | {r['val_unlabeled_pct_ge_0.90']:.1f}% |\n"
        )

    md += "\n---\n\n## 2. Multi-Seed Shortlist Evaluation (Validation Steps `35–41`)\n\n"
    md += "| Shortlisted Exp ID | Category | Seeds | Val PR-AUC (Mean $\\pm$ Std) | Val F1 `0.01–0.99` (Mean $\\pm$ Std) | Brier Mean | ECE Mean |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for s in payload["shortlist_summary"]:
        cat = "No-Graph Control" if s["is_nograph"] else "Graph"
        md += (
            f"| `{s['exp_id']}` | {cat} | `{s['seeds']}` | **{s['val_prauc_mean']:.4f} $\\pm$ {s['val_prauc_std']:.4f}** | "
            f"{s['val_f1_recal_mean']:.4f} $\\pm$ {s['val_f1_recal_std']:.4f} | {s['brier_mean']:.4f} | {s['ece_mean']:.4f} |\n"
        )

    md += "\n### Per-Seed Validation Breakdown\n\n"
    md += "| Exp ID | Seed | Best Ep | Val PR-AUC | Val F1 (`0.01–0.99`) | Val Prec | Val Rec | Brier | ECE |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for eid, runs in payload.get("multi_seed_runs", {}).items():
        for r in runs:
            vm = r["validation_metrics_recal"]
            md += (
                f"| `{eid}` | {r['seed']} | {r['best_epoch']} | {vm['pr_auc']:.4f} | "
                f"{vm['illicit_f1']:.4f} | {vm['illicit_precision']*100:.2f}% | "
                f"{vm['illicit_recall']*100:.2f}% | {vm['brier_score']:.4f} | {vm['ece_10bin']:.4f} |\n"
            )

    md += "\n---\n\n## 3. Post-Selection Test Evaluation: Graph Champion vs. No-Graph Control (Steps `42–49`, `8,841` Nodes)\n\n"
    md += "| Champion Role | Exp ID | Grid | $\\tau^*$ | Precision | Recall | Illicit F1 | PR-AUC | ROC-AUC | Brier | ECE | TP | FP | TN | FN |\n"
    md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for role_name, ch in [("Graph Champion", g_champ), ("No-Graph Control", n_champ)]:
        for g_lbl, th_key, m_key in [
            ("0.10–0.90", "threshold_legacy_01_90", "test_metrics_legacy_01_90"),
            ("0.01–0.99", "threshold_recal_01_99", "test_metrics_recal_01_99"),
        ]:
            m = ch[m_key]
            cm = m["confusion_matrix"]
            md += (
                f"| **{role_name}** | `{ch['exp_id']}` | `{g_lbl}` | `{ch[th_key]:.2f}` | "
                f"{m['illicit_precision']*100:.2f}% | {m['illicit_recall']*100:.2f}% | **{m['illicit_f1']:.4f}** | "
                f"**{m['pr_auc']:.4f}** | {m['roc_auc']:.4f} | {m['brier_score']:.4f} | {m['ece_10bin']:.4f} | "
                f"{cm['true_positives_caught_fraud']} | {cm['false_positives_false_alarm']:,} | "
                f"{cm['true_negatives_licit']:,} | {cm['false_negatives_missed_fraud']} |\n"
            )

    md += f"""
### Quantified Graph Message Passing Impact (`Graph Champion` minus `No-Graph Control`, `0.01–0.99` Grid)
- **$\\Delta$ Test PR-AUC:** `{delta['delta_pr_auc']:+.4f}` (`{'Graph higher' if delta['graph_outperforms_on_pr_auc'] else 'No-Graph higher/equal'}`)
- **$\\Delta$ Test ROC-AUC:** `{delta['delta_roc_auc']:+.4f}`
- **$\\Delta$ Test Illicit F1:** `{delta['delta_f1_01_99']:+.4f}` (`{'Graph higher' if delta['graph_outperforms_on_f1'] else 'No-Graph higher/equal'}`)
- **$\\Delta$ Precision / Recall:** `{delta['delta_precision_01_99']*100:+.2f} pp` / `{delta['delta_recall_01_99']*100:+.2f} pp`
- **$\\Delta$ Calibration (Brier / ECE):** `{delta['delta_brier']:+.4f}` / `{delta['delta_ece']:+.4f}`
- **$\\Delta$ Confusion Counts (TP / FP / TN / FN):** `{delta['delta_tp']:+d}` / `{delta['delta_fp']:+,d}` / `{delta['delta_tn']:+,d}` / `{delta['delta_fn']:+d}`
"""
    return md


# =============================================================================
# 7. MAIN PIPELINE ENTRYPOINT
# =============================================================================

def run_stage4d(smoke_test=False):
    print("=" * 88)
    mode_str = " (SMOKE TEST MODE)" if smoke_test else ""
    print(f"STAGE 4D: NEXT-GENERATION GRAPH FRAUD DETECTION EXPERIMENTS{mode_str}")
    print("=" * 88)
    t_start = time.time()

    # Verify FocalLoss positive-class semantics before running any experiments
    assert verify_focal_loss_positive_class(), "FocalLoss positive-class check failed!"
    print("[Unit Check PASSED] BinaryFocalLoss verified: y=1 (Illicit) is positive class.")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "processed")
    models_dir = os.path.join(base_dir, "models", "stage4d")
    reports_dir = os.path.join(base_dir, "reports")
    results_dir = os.path.join(base_dir, "results")

    snaps_path = os.path.join(processed_dir, "temporal_snapshots.pt")
    if not os.path.exists(snaps_path):
        raise FileNotFoundError(f"Missing {snaps_path}. Run Stage 2 preprocessing first!")

    snapshots = torch.load(snaps_path, weights_only=False)
    assert len(snapshots) == 49
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded 49 snapshots on device={device}.")

    # Strict training-only pos_weight (Steps 1-34)
    train_labels = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34)])
    n_lic = int((train_labels == 0).sum())
    n_ill = int((train_labels == 1).sum())
    pos_weight_val = n_lic / max(n_ill, 1)
    y_val_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34, 41)])
    print(f"Train (1-34) pos_weight={pos_weight_val:.4f} | Val (35-41) N={len(y_val_all):,} | Test (42-49) SEALED.")

    # Experiment 2 Phase A: Train & Val Regime-Shift Profile (Steps 1-41 ONLY)
    train_val_profile = profile_snapshot_statistics(snapshots, 0, 41)
    val_x_norms = [v["x_norm_mean_licit"] for v in train_val_profile.values() if v["split"] == "val"]
    print(f"[Exp 2 Phase A] Train/Val snapshot profile computed (Val licit |x| range: {min(val_x_norms):.2f}-{max(val_x_norms):.2f}).")

    snapshots_dev = [s.to(device) for s in snapshots]
    adj_cache = {
        "gcn_directed": [build_gcn_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device) for s in snapshots],
        "sage_directed": [build_neighbor_mean_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device) for s in snapshots],
    }

    catalog = build_experiment_catalog()
    max_epochs = 2 if smoke_test else 50
    patience = 15

    # Phase 1: Single-seed (seed=42) screening strictly on Train (1-34) / Val (35-41)
    print(f"\n[Phase 1] Running controlled screening across {len(catalog)} configurations (max_epochs={max_epochs})...")
    screening_runs = []
    for spec in catalog:
        res = train_single_experiment(
            spec=spec,
            seed=42,
            snapshots_dev=snapshots_dev,
            adj_cache=adj_cache,
            y_val_all=y_val_all,
            default_pos_weight=pos_weight_val,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            smoke_test=smoke_test,
        )
        screening_runs.append(res)

    # Phase 2: Separate Graph shortlist & No-Graph Control shortlist, then multi-seed eval
    shortlist_k_graph = 2 if smoke_test else 3
    shortlist_k_nograph = 1 if smoke_test else 2
    shortlist_seeds = [42, 123] if smoke_test else [42, 123, 456]

    ranked_screening = sorted(
        screening_runs,
        key=lambda r: (r["validation_metrics_recal"]["pr_auc"] or 0.0, r["validation_metrics_recal"]["illicit_f1"]),
        reverse=True,
    )
    graph_shortlist = [r for r in ranked_screening if not is_nograph_spec(r["spec"])][:shortlist_k_graph]
    nograph_shortlist = [r for r in ranked_screening if is_nograph_spec(r["spec"])][:shortlist_k_nograph]
    shortlisted = graph_shortlist + nograph_shortlist

    print(
        f"\n[Phase 2] Multi-seed evaluation (seeds={shortlist_seeds}) on "
        f"top-{len(graph_shortlist)} Graph models + top-{len(nograph_shortlist)} No-Graph controls..."
    )

    shortlist_summary = []
    all_seed_runs = {}
    all_individual_runs = list(screening_runs)

    for base_run in shortlisted:
        eid = base_run["exp_id"]
        runs_for_exp = [base_run]
        for s_val in shortlist_seeds:
            if s_val == 42:
                continue
            r_s = train_single_experiment(
                spec=base_run["spec"],
                seed=s_val,
                snapshots_dev=snapshots_dev,
                adj_cache=adj_cache,
                y_val_all=y_val_all,
                default_pos_weight=pos_weight_val,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                smoke_test=smoke_test,
            )
            runs_for_exp.append(r_s)
            all_individual_runs.append(r_s)

        all_seed_runs[eid] = runs_for_exp
        praucs = [r["validation_metrics_recal"]["pr_auc"] or 0.0 for r in runs_for_exp]
        f1s = [r["validation_metrics_recal"]["illicit_f1"] for r in runs_for_exp]
        briers = [r["validation_metrics_recal"]["brier_score"] for r in runs_for_exp]
        eces = [r["validation_metrics_recal"]["ece_10bin"] for r in runs_for_exp]
        shortlist_summary.append({
            "exp_id": eid,
            "is_nograph": is_nograph_spec(base_run["spec"]),
            "seeds": shortlist_seeds,
            "val_prauc_mean": round(float(np.mean(praucs)), 4),
            "val_prauc_std": round(float(np.std(praucs)), 4),
            "val_f1_recal_mean": round(float(np.mean(f1s)), 4),
            "val_f1_recal_std": round(float(np.std(f1s)), 4),
            "brier_mean": round(float(np.mean(briers)), 4),
            "ece_mean": round(float(np.mean(eces)), 4),
        })

    shortlist_summary.sort(key=lambda x: (x["val_prauc_mean"], x["val_f1_recal_mean"]), reverse=True)

    graph_summary = [s for s in shortlist_summary if not s["is_nograph"]]
    nograph_summary = [s for s in shortlist_summary if s["is_nograph"]]

    graph_champ_id = graph_summary[0]["exp_id"]
    graph_best_seed_run = max(all_seed_runs[graph_champ_id], key=lambda r: r["validation_metrics_recal"]["pr_auc"] or 0.0)
    assert not is_nograph_spec(graph_best_seed_run["spec"]), "Graph champion must exclude no_graph!"

    nograph_champ_id = nograph_summary[0]["exp_id"]
    nograph_best_seed_run = max(all_seed_runs[nograph_champ_id], key=lambda r: r["validation_metrics_recal"]["pr_auc"] or 0.0)
    assert is_nograph_spec(nograph_best_seed_run["spec"]), "No-Graph champion must be a no_graph config!"

    overall_top = shortlist_summary[0]
    overall_champion_summary = {
        "exp_id": overall_top["exp_id"],
        "category": "no_graph" if overall_top["is_nograph"] else "graph",
        "val_prauc_mean": overall_top["val_prauc_mean"],
        "val_prauc_std": overall_top["val_prauc_std"],
        "val_f1_recal_mean": overall_top["val_f1_recal_mean"],
    }

    print("\n" + "=" * 88)
    print(f"SELECTED GRAPH CHAMPION (Val PR-AUC):    {graph_champ_id} (Mean={graph_summary[0]['val_prauc_mean']:.4f}, BestSeed={graph_best_seed_run['seed']})")
    print(f"SELECTED NO-GRAPH CONTROL (Val PR-AUC):  {nograph_champ_id} (Mean={nograph_summary[0]['val_prauc_mean']:.4f}, BestSeed={nograph_best_seed_run['seed']})")
    print(f"SELECTED OVERALL CHAMPION (Val PR-AUC):  {overall_champion_summary['exp_id']} ({overall_champion_summary['category']}, Mean={overall_champion_summary['val_prauc_mean']:.4f})")
    print("=" * 88)

    # Phase 3: Post-selection Test Evaluation (Steps 42-49) ONLY AFTER all selections & thresholds finalized
    y_test_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(41, 49)])
    graph_champion_res = evaluate_champion_on_test(graph_best_seed_run, "Graph Champion", snapshots_dev, adj_cache, y_test_all, device)
    nograph_champion_res = evaluate_champion_on_test(nograph_best_seed_run, "No-Graph Control", snapshots_dev, adj_cache, y_test_all, device)
    graph_vs_nograph_delta = compute_graph_vs_nograph_delta(graph_champion_res, nograph_champion_res)

    print(
        f"[Graph vs No-Graph Test Delta (0.01-0.99)] "
        f"dPR-AUC={graph_vs_nograph_delta['delta_pr_auc']:+.4f}, "
        f"dROC-AUC={graph_vs_nograph_delta['delta_roc_auc']:+.4f}, "
        f"dF1={graph_vs_nograph_delta['delta_f1_01_99']:+.4f}, "
        f"dFP={graph_vs_nograph_delta['delta_fp']:+d}, dTP={graph_vs_nograph_delta['delta_tp']:+d}"
    )

    clean_screening = [
        {k: v for k, v in r.items() if k != "best_state"} for r in screening_runs
    ]
    clean_multi_seed = {
        eid: [{k: v for k, v in r.items() if k != "best_state"} for r in runs]
        for eid, runs in all_seed_runs.items()
    }

    payload = {
        "stage": "Stage 4D - Next-Generation Graph Fraud Detection",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "smoke_test": bool(smoke_test),
        "train_val_regime_profile": train_val_profile,
        "screening_results": clean_screening,
        "shortlist_summary": shortlist_summary,
        "multi_seed_runs": clean_multi_seed,
        "graph_champion": graph_champion_res,
        "nograph_control_champion": nograph_champion_res,
        "overall_champion": overall_champion_summary,
        "graph_vs_nograph_delta": graph_vs_nograph_delta,
    }

    if smoke_test:
        print("\n[Smoke Test] Skipping all checkpoint, metrics, report, and plot disk writes to protect production outputs.")
    else:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Preserve individual experiment checkpoints (all screening + multi-seed runs) in models/stage4d/
        for r in all_individual_runs:
            exp_ckpt = os.path.join(models_dir, f"{r['exp_id']}_seed{r['seed']}.pt")
            torch.save({
                "exp_id": r["exp_id"],
                "seed": r["seed"],
                "spec": r["spec"],
                "model_state_dict": r["best_state"],
                "validation_metrics_legacy": r["validation_metrics_legacy"],
                "validation_metrics_recal": r["validation_metrics_recal"],
            }, exp_ckpt)

        # Save Graph Champion separately at models/nextgen_gnn_best.pt
        graph_ckpt_path = os.path.join(base_dir, "models", "nextgen_gnn_best.pt")
        torch.save({
            "role": "graph_champion",
            "exp_id": graph_champ_id,
            "seed": graph_best_seed_run["seed"],
            "spec": graph_best_seed_run["spec"],
            "model_state_dict": graph_best_seed_run["best_state"],
            "threshold_legacy_01_90": graph_best_seed_run["threshold_legacy_01_90"],
            "threshold_recal_01_99": graph_best_seed_run["threshold_recal_01_99"],
            "threshold_rec30_recal": graph_best_seed_run["threshold_rec30_recal"],
            "test_metrics_legacy_01_90": graph_champion_res["test_metrics_legacy_01_90"],
            "test_metrics_recal_01_99": graph_champion_res["test_metrics_recal_01_99"],
        }, graph_ckpt_path)
        print(f"  [Saved Graph Champion Checkpoint]: {graph_ckpt_path}")

        # Save No-Graph Control Champion separately inside models/stage4d/
        nograph_ckpt_path = os.path.join(models_dir, "nograph_control_best.pt")
        torch.save({
            "role": "nograph_control_champion",
            "exp_id": nograph_champ_id,
            "seed": nograph_best_seed_run["seed"],
            "spec": nograph_best_seed_run["spec"],
            "model_state_dict": nograph_best_seed_run["best_state"],
            "threshold_legacy_01_90": nograph_best_seed_run["threshold_legacy_01_90"],
            "threshold_recal_01_99": nograph_best_seed_run["threshold_recal_01_99"],
            "threshold_rec30_recal": nograph_best_seed_run["threshold_rec30_recal"],
            "test_metrics_legacy_01_90": nograph_champion_res["test_metrics_legacy_01_90"],
            "test_metrics_recal_01_99": nograph_champion_res["test_metrics_recal_01_99"],
        }, nograph_ckpt_path)
        print(f"  [Saved No-Graph Control Checkpoint]: {nograph_ckpt_path}")

        save_nextgen_plots(clean_screening, shortlist_summary, graph_champion_res, nograph_champion_res, results_dir)

        json_path = os.path.join(reports_dir, "stage4d_nextgen_experiments_metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"  [Saved Metrics JSON]: {json_path}")

        md_path = os.path.join(reports_dir, "stage4d_nextgen_experiments_report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(generate_nextgen_report(payload))
        print(f"  [Saved Markdown Report]: {md_path}")

    print(f"\nStage 4D completed in {time.time() - t_start:.2f}s.")
    print("=" * 88)
    return payload


if __name__ == "__main__":
    is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
    run_stage4d(smoke_test=is_smoke)
