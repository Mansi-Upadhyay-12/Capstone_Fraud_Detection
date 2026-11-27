"""
Stage 4H: Distribution-Robust Feature Learning (RankGauss) and Hybrid Model Investigation

Experiments:
  - Experiment A (RankGauss): QuantileTransformer(output_distribution='normal')
    fitted strictly on Train Steps 1-34 (all 182 features), transformed unchanged
    onto Val (35-41) and Test (42-49). Evaluated across GCNAvg, BiDirGCN, NoGraph, and HGB.
  - Experiment B (Hybrid GNN + HistGradientBoosting):
    5-fold contiguous snapshot OOF GNN embeddings strictly within Train Steps 1-34
    (zero target/temporal leakage) combined with original 182 features -> HGB
    trained ONLY on Steps 1-34.

Strict Controls:
  - Preserves all Stage 1-4G artifacts untouched.
  - Multi-seed evaluation [42, 123, 456], mean +- std reporting.
  - Validation Steps 35-41 ONLY for architecture/hyperparameter selection & 0.01-0.99 thresholding.
  - Test Steps 42-49 evaluated ONLY after all selections are frozen.
"""

import json
import os
import sys
import tempfile
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models_nextgen import SeparatedGraphNet, build_gcn_sparse_adj
from src.models_stage4f import BiDirRelationalGCN, build_true_directed_gcn_adj
from src.train_nextgen import compute_metrics_and_calibration, select_thresholds_val_only


SEEDS = [42, 123, 456]


# =============================================================================
# 1. STRICT TRAIN-ONLY RANKGAUSS TRANSFORMATION
# =============================================================================

def build_rankgauss_snapshots(snapshots):
    """
    Fits QuantileTransformer(output_distribution='normal') + post-StandardScaler
    strictly on Train Steps 1-34 across all 182 features (re-centering zero-inflated
    boundary quantile spikes), then transforms Steps 1-49 unchanged.
    """
    X_train_fit = torch.cat([s.x for s in snapshots[:34]], dim=0).numpy()
    qt_pipe = Pipeline([
        ("qt", QuantileTransformer(
            n_quantiles=1000,
            output_distribution="normal",
            subsample=200000,
            random_state=42,
        )),
        ("post_scaler", StandardScaler()),
    ])
    qt_pipe.fit(X_train_fit)

    rg_snapshots = []
    for s in snapshots:
        s_clone = s.clone()
        x_rg = qt_pipe.transform(s.x.numpy()).astype(np.float32)
        assert np.isfinite(x_rg).all(), f"Non-finite RankGauss values at step {s.timestep}"
        s_clone.x = torch.from_numpy(x_rg)
        rg_snapshots.append(s_clone)

    x_tr_rg = torch.cat([s.x for s in rg_snapshots[:34]], dim=0).numpy()
    x_43_rg = torch.cat([s.x for s in rg_snapshots[42:49]], dim=0).numpy()
    diagnostics = {
        "n_quantiles": int(qt_pipe.named_steps["qt"].n_quantiles_),
        "fit_rows": int(len(X_train_fit)),
        "train_mean_abs_max": round(float(np.abs(x_tr_rg.mean(axis=0)).max()), 4),
        "train_std_min": round(float(x_tr_rg.std(axis=0).min()), 4),
        "train_std_max": round(float(x_tr_rg.std(axis=0).max()), 4),
        "train_abs_max": round(float(np.abs(x_tr_rg).max()), 4),
        "steps43_49_abs_max": round(float(np.abs(x_43_rg).max()), 4),
        "steps43_49_frac_gt5_pct": round(float((np.abs(x_43_rg) > 5.0).mean() * 100.0), 4),
    }
    return qt_pipe, rg_snapshots, diagnostics


# =============================================================================
# 2. EXPERIMENT A: NEURAL GNN / NO-GRAPH MODELS ON RANKGAUSS-182
# =============================================================================

def build_expa_model(arch_name, device):
    if arch_name == "GCNAvg_182":
        return SeparatedGraphNet(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3, graph_mode="gcn_avg").to(device)
    elif arch_name == "NoGraph_182":
        return SeparatedGraphNet(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3, graph_mode="no_graph").to(device)
    elif arch_name == "BiDirGCN_182":
        return BiDirRelationalGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3).to(device)
    raise ValueError(f"Unknown arch_name: {arch_name}")


def forward_expa_step(model, arch_name, x, bundle, state):
    if arch_name == "BiDirGCN_182":
        return model.forward_step(x, bundle, state)
    return model.forward_step(x, bundle["legacy_dir"], state)


def rollout_expa(model, arch_name, snapshots_dev, bundles_dev, start_idx, end_idx):
    model.eval()
    probs_out = []
    step_out = {}
    with torch.no_grad():
        state = model.init_state()
        for t in range(start_idx):
            s = snapshots_dev[t]
            _, state, _ = forward_expa_step(model, arch_name, s.x, bundles_dev[t], state)
        for t in range(start_idx, end_idx):
            s = snapshots_dev[t]
            logits, state, _ = forward_expa_step(model, arch_name, s.x, bundles_dev[t], state)
            p = torch.sigmoid(logits).cpu().numpy()
            lm = s.labeled_mask.cpu().numpy()
            y_l = s.y.cpu().numpy()[lm]
            p_l = p[lm]
            probs_out.extend(p_l)
            step_out[int(s.timestep)] = {"probs": p_l, "y": y_l, "all_probs": p}
    return np.array(probs_out), step_out


def train_single_expa_neural(
    exp_id,
    arch_name,
    seed,
    snapshots_dev,
    bundles_dev,
    y_val_all,
    pos_weight_val,
    device,
    max_epochs=50,
    patience=15,
    smoke_test=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_expa_model(arch_name, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_prauc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
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
            logits_t, state, _ = forward_expa_step(model, arch_name, s.x, bundles_dev[t], state)
            if s.labeled_mask.sum() > 0:
                total_loss = total_loss + criterion(logits_t[s.labeled_mask], s.y[s.labeled_mask].float())
                loss_steps += 1
        avg_loss = total_loss / max(loss_steps, 1)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        y_val_prob, _ = rollout_expa(model, arch_name, snapshots_dev, bundles_dev, 34, 41)
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
            print(f"    [{exp_id} | seed={seed}] Ep {epoch:02d}/{max_epochs} | Loss={avg_loss.item():.4f} | Val PR-AUC={val_prauc:.4f}")
        if patience_counter >= patience and not smoke_test:
            print(f"    Early stop at epoch {epoch} (Best Ep={best_epoch}, Val PR-AUC={best_val_prauc:.4f})")
            break

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    y_val_best, _ = rollout_expa(model, arch_name, snapshots_dev, bundles_dev, 34, 41)
    tau_rec, _, tau_rec30 = select_thresholds_val_only(y_val_all, y_val_best, 0.01, 0.99, 99, 0.30)
    val_rec = compute_metrics_and_calibration(y_val_all, y_val_best, threshold=tau_rec)

    return {
        "exp_id": exp_id,
        "experiment_group": "ExpA_RankGauss",
        "arch_name": arch_name,
        "seed": int(seed),
        "total_parameters": total_params,
        "best_epoch": int(best_epoch),
        "training_time_seconds": round(train_time, 3),
        "threshold_recal_01_99": tau_rec,
        "threshold_rec30_recal": tau_rec30,
        "validation_metrics_recal": val_rec,
        "best_state": best_state,
    }


# =============================================================================
# 3. EXPERIMENT B: LEAKAGE-FREE 5-FOLD OOF GNN EMBEDDINGS + HISTGRADIENTBOOSTING
# =============================================================================

def extract_model_embeddings_for_indices(model, encoder_type, snapshots_dev, bundles_dev, step_indices, device):
    """
    Extracts 64-d hidden embeddings (before classification head) + 1-d raw logit
    for the requested snapshot indices [0..48]. Zero changes to existing model classes.
    """
    model.eval()
    captured = {}

    def hook_fn(mod, inp, out):
        captured["emb"] = inp[0].detach().cpu().numpy()

    handle = model.head.register_forward_hook(hook_fn)
    out_by_step = {}
    with torch.no_grad():
        for t in step_indices:
            s = snapshots_dev[t]
            x_in = s.x[:, :93] if encoder_type == "local93_gcnavg" else s.x
            if encoder_type == "local93_gcnavg":
                logits, _, _ = model.forward_step(x_in, bundles_dev[t]["legacy_dir"], None)
            else:
                logits, _, _ = model.forward_step(x_in, bundles_dev[t], None)
            emb = captured["emb"].copy()
            lg = logits.detach().cpu().numpy().reshape(-1, 1)
            out_by_step[t] = np.concatenate([emb, lg], axis=1).astype(np.float32)  # [N_t, 65]
    handle.remove()
    return out_by_step


def train_encoder_for_indices(encoder_type, seed, snapshots_dev, bundles_dev, train_indices, pos_weight_val, device, epochs=38):
    """Trains GNN encoder strictly on `train_indices` (never seeing held-out fold/Val/Test)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if encoder_type == "local93_gcnavg":
        model = SeparatedGraphNet(in_channels=93, hidden_channels=128, out_channels=64, dropout_rate=0.3, graph_mode="gcn_avg").to(device)
    elif encoder_type == "rg182_bidir":
        model = BiDirRelationalGCN(in_channels=182, hidden_channels=128, out_channels=64, dropout_rate=0.3).to(device)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    # Compute pos_weight strictly from train_indices labels (zero held-out info)
    tr_y = np.concatenate([snapshots_dev[t].y[snapshots_dev[t].labeled_mask].cpu().numpy() for t in train_indices])
    pw = int((tr_y == 0).sum()) / max(int((tr_y == 1).sum()), 1)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        steps = 0
        for t in train_indices:
            s = snapshots_dev[t]
            x_in = s.x[:, :93] if encoder_type == "local93_gcnavg" else s.x
            if encoder_type == "local93_gcnavg":
                logits_t, _, _ = model.forward_step(x_in, bundles_dev[t]["legacy_dir"], None)
            else:
                logits_t, _, _ = model.forward_step(x_in, bundles_dev[t], None)
            if s.labeled_mask.sum() > 0:
                total_loss = total_loss + criterion(logits_t[s.labeled_mask], s.y[s.labeled_mask].float())
                steps += 1
        avg_loss = total_loss / max(steps, 1)
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

    return model


def generate_leakage_free_oof_embeddings(
    encoder_type,
    seed,
    snapshots_dev,
    bundles_dev,
    pos_weight_val,
    device,
    fold_epochs=38,
    final_epochs=40,
):
    """
    Generates leakage-free embeddings for Train (Steps 1-34 via 5 contiguous snapshot blocks),
    and fits the final GNN encoder strictly on Steps 1-34 for Val (35-41) & Test (42-49).
    Zero Val/Test labels touch GNN fitting; zero held-out fold labels touch OOF embeddings.
    """
    # 5 contiguous chronological blocks within Steps 1-34 (indices 0..33)
    blocks = [
        list(range(0, 7)),    # Steps 1-7
        list(range(7, 14)),   # Steps 8-14
        list(range(14, 21)),  # Steps 15-21
        list(range(21, 28)),  # Steps 22-28
        list(range(28, 34)),  # Steps 29-34
    ]
    assert sum(len(b) for b in blocks) == 34

    oof_emb_by_step = {}
    for fold_idx, val_block in enumerate(blocks):
        tr_indices = [t for i, b in enumerate(blocks) if i != fold_idx for t in b]
        assert set(tr_indices).isdisjoint(set(val_block))
        assert max(tr_indices + val_block) <= 33
        fold_model = train_encoder_for_indices(
            encoder_type, seed + fold_idx + 1, snapshots_dev, bundles_dev, tr_indices, pos_weight_val, device, epochs=fold_epochs
        )
        fold_embs = extract_model_embeddings_for_indices(fold_model, encoder_type, snapshots_dev, bundles_dev, val_block, device)
        oof_emb_by_step.update(fold_embs)

    # Train final encoder strictly on Steps 1-34 (indices 0..33) for Val (34..40) and Test (41..48)
    full_train_indices = list(range(34))
    final_encoder = train_encoder_for_indices(
        encoder_type, seed, snapshots_dev, bundles_dev, full_train_indices, pos_weight_val, device, epochs=final_epochs
    )
    val_test_embs = extract_model_embeddings_for_indices(
        final_encoder, encoder_type, snapshots_dev, bundles_dev, list(range(34, 49)), device
    )
    oof_emb_by_step.update(val_test_embs)
    assert len(oof_emb_by_step) == 49
    return oof_emb_by_step, final_encoder


def train_hybrid_hgb_on_oof(
    exp_id,
    seed,
    raw_snapshots,
    emb_by_step,
    hgb_kwargs,
    feature_mode="orig182_plus_emb64",
):
    """Trains downstream HistGradientBoosting strictly on Train Steps 1-34 and evaluates on Val 35-41."""
    def collect_xy(start_idx, end_idx):
        Xs, ys = [], []
        step_x, step_y = {}, {}
        for t in range(start_idx, end_idx):
            s = raw_snapshots[t]
            lm = s.labeled_mask.numpy()
            x_orig = s.x.numpy()[lm]
            emb_65 = emb_by_step[t][lm]
            if feature_mode == "orig182_plus_emb64":
                x_comb = np.concatenate([x_orig, emb_65[:, :64]], axis=1)
            elif feature_mode == "orig182_plus_logit1":
                x_comb = np.concatenate([x_orig, emb_65[:, 64:65]], axis=1)
            elif feature_mode == "orig182_only":
                x_comb = x_orig
            else:
                raise ValueError(f"Unknown feature_mode: {feature_mode}")
            y_l = s.y.numpy()[lm]
            Xs.append(x_comb)
            ys.append(y_l)
            step_x[int(s.timestep)] = x_comb
            step_y[int(s.timestep)] = y_l
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), step_x, step_y

    X_tr, y_tr, _, _ = collect_xy(0, 34)
    X_va, y_va, step_x_va, step_y_va = collect_xy(34, 41)

    t0 = time.time()
    clf = HistGradientBoostingClassifier(**hgb_kwargs, random_state=seed)
    clf.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    val_prob = clf.predict_proba(X_va)[:, 1]
    tau_rec, _, tau_rec30 = select_thresholds_val_only(y_va, val_prob, 0.01, 0.99, 99, 0.30)
    val_metrics = compute_metrics_and_calibration(y_va, val_prob, threshold=tau_rec)

    return {
        "exp_id": exp_id,
        "experiment_group": "ExpB_Hybrid_HGB",
        "seed": int(seed),
        "feature_dim": int(X_tr.shape[1]),
        "hgb_kwargs": hgb_kwargs,
        "feature_mode": feature_mode,
        "training_time_seconds": round(fit_time, 3),
        "threshold_recal_01_99": tau_rec,
        "threshold_rec30_recal": tau_rec30,
        "validation_metrics_recal": val_rec if False else val_metrics,
        "clf": clf,
        "collect_xy_fn": collect_xy,
    }


# =============================================================================
# 4. POST-SELECTION EVALUATION, SUMMARIZATION & VISUALIZATIONS
# =============================================================================

def eval_slices_from_step_outputs(step_probs, tau):
    def agg(steps):
        y = np.concatenate([step_probs[t]["y"] for t in steps])
        p = np.concatenate([step_probs[t]["probs"] for t in steps])
        return compute_metrics_and_calibration(y, p, threshold=tau)

    per_step = {}
    for t in range(42, 50):
        ms = compute_metrics_and_calibration(step_probs[t]["y"], step_probs[t]["probs"], threshold=tau)
        per_step[str(t)] = {
            "pr_auc": ms["pr_auc"],
            "roc_auc": ms["roc_auc"],
            "illicit_f1": ms["illicit_f1"],
            "illicit_precision": ms["illicit_precision"],
            "illicit_recall": ms["illicit_recall"],
            "brier_score": ms["brier_score"],
            "ece_10bin": ms["ece_10bin"],
            "tp": ms["confusion_matrix"]["true_positives_caught_fraud"],
            "fp": ms["confusion_matrix"]["false_positives_false_alarm"],
            "fn": ms["confusion_matrix"]["false_negatives_missed_fraud"],
            "tn": ms["confusion_matrix"]["true_negatives_licit"],
        }

    return {
        "test_all_42_49": agg(range(42, 50)),
        "test_step42_only": agg([42]),
        "test_steps43_49_only": agg(range(43, 50)),
        "per_snapshot_test_metrics": per_step,
    }


def summarize_runs_mean_std(runs):
    def st(vals):
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    val_m = [r["validation_metrics_recal"] for r in runs]
    te_all = [r["test_all_42_49"] for r in runs]
    te_42 = [r["test_step42_only"] for r in runs]
    te_43 = [r["test_steps43_49_only"] for r in runs]

    def grp_stats(ms):
        return {
            "pr_auc": st([m["pr_auc"] for m in ms]),
            "roc_auc": st([m["roc_auc"] for m in ms]),
            "illicit_f1": st([m["illicit_f1"] for m in ms]),
            "illicit_precision": st([m["illicit_precision"] for m in ms]),
            "illicit_recall": st([m["illicit_recall"] for m in ms]),
            "brier_score": st([m["brier_score"] for m in ms]),
            "ece_10bin": st([m["ece_10bin"] for m in ms]),
        }

    return {
        "val_35_41": grp_stats(val_m),
        "test_all_42_49": grp_stats(te_all),
        "test_step42_only": grp_stats(te_42),
        "test_steps43_49_only": grp_stats(te_43),
    }


def save_stage4h_plots(out_dir, summaries, best_expa_step, best_expb_step, hgb_step, gcn_anchor_step):
    # Plot 1: Val & Test PR-AUC + Steps 43-49 ROC-AUC across all Stage 4H configs
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    names = [s["exp_id"] for s in summaries]
    x = np.arange(len(names))
    w = 0.35

    v_pr = [s["summary"]["val_35_41"]["pr_auc"]["mean"] for s in summaries]
    v_sd = [s["summary"]["val_35_41"]["pr_auc"]["std"] for s in summaries]
    t_pr = [s["summary"]["test_all_42_49"]["pr_auc"]["mean"] for s in summaries]
    t_sd = [s["summary"]["test_all_42_49"]["pr_auc"]["std"] for s in summaries]

    axes[0].bar(x - w / 2, v_pr, w, yerr=v_sd, capsize=4, label="Val PR-AUC (35-41)", color="#2b6cb0", edgecolor="black")
    axes[0].bar(x + w / 2, t_pr, w, yerr=t_sd, capsize=4, label="Test PR-AUC (42-49)", color="#38a169", edgecolor="black")
    axes[0].axhline(0.5646, color="#e53e3e", linestyle="--", label="HGB Base Test PR (0.5646)")
    axes[0].axhline(0.5268, color="#4a5568", linestyle=":", label="GCNAvg Base Test PR (0.5268)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("PR-AUC (Mean +/- Std)")
    axes[0].set_title("Stage 4H Ranking: Validation vs. Test All (42-49)")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(axis="y", alpha=0.25)

    roc43 = [s["summary"]["test_steps43_49_only"]["roc_auc"]["mean"] for s in summaries]
    roc43_sd = [s["summary"]["test_steps43_49_only"]["roc_auc"]["std"] for s in summaries]
    axes[1].bar(x, roc43, 0.5, yerr=roc43_sd, capsize=4, color="#805ad5", edgecolor="black")
    axes[1].axhline(0.7698, color="#e53e3e", linestyle="--", label="HGB Base 43-49 ROC (0.7698)")
    axes[1].axhline(0.6247, color="#4a5568", linestyle=":", label="GCNAvg Base 43-49 ROC (0.6247)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    axes[1].set_ylim(0.45, 0.90)
    axes[1].set_ylabel("ROC-AUC (Steps 43-49)")
    axes[1].set_title("Post-Shutdown Generalization (Steps 43-49 ROC-AUC)")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(axis="y", alpha=0.25)

    brier = [s["summary"]["test_all_42_49"]["brier_score"]["mean"] for s in summaries]
    ece = [s["summary"]["test_all_42_49"]["ece_10bin"]["mean"] for s in summaries]
    axes[2].bar(x - w / 2, brier, w, label="Test Brier", color="#dd6b20", edgecolor="black")
    axes[2].bar(x + w / 2, ece, w, label="Test ECE", color="#319795", edgecolor="black")
    axes[2].axhline(0.0362, color="#e53e3e", linestyle="--", label="HGB Base Brier (0.0362)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    axes[2].set_ylabel("Calibration Error (Lower Better)")
    axes[2].set_title("Test Calibration (Brier & ECE)")
    axes[2].legend(fontsize=7.5)
    axes[2].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage4h_comparison.png"), dpi=200)
    plt.close(fig)

    # Plot 2: Per-snapshot Steps 42-49 ROC-AUC & PR-AUC comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    steps = list(range(42, 50))
    series = [
        ("HGB Baseline (182)", hgb_step, "#e53e3e", "-o"),
        ("GCNAvg Anchor (Local-93)", gcn_anchor_step, "#4a5568", "--s"),
        ("Best Exp A (RankGauss)", best_expa_step, "#2b6cb0", "-^"),
        ("Best Exp B (Hybrid HGB)", best_expb_step, "#38a169", "-D"),
    ]
    for lbl, st_dict, col, fmt in series:
        axes[0].plot(steps, [st_dict[str(t)]["pr_auc"] for t in steps], fmt, color=col, label=lbl, ms=5)
        axes[1].plot(steps, [st_dict[str(t)]["roc_auc"] for t in steps], fmt, color=col, label=lbl, ms=5)
    axes[0].set_title("Per-Snapshot Test PR-AUC (Steps 42-49)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("PR-AUC")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Per-Snapshot Test ROC-AUC (Steps 42-49)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stage4h_per_snapshot.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# 5. MAIN STAGE 4H RUNNER
# =============================================================================

def run_stage4h(smoke_test=False):
    print("=" * 88)
    mode_str = " (SMOKE TEST MODE)" if smoke_test else ""
    print(f"STAGE 4H: RANKGAUSS & HYBRID GNN+HGB INVESTIGATION{mode_str}")
    print("=" * 88)
    t_start = time.time()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "reports", "stage4h")
    models_dir = os.path.join(base_dir, "models", "stage4h")

    snapshots = torch.load(os.path.join(base_dir, "processed", "temporal_snapshots.pt"), weights_only=False)
    assert len(snapshots) == 49
    device = torch.device("cpu")

    # Strict train-only pos_weight
    train_y = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34)])
    pos_weight_val = int((train_y == 0).sum()) / max(int((train_y == 1).sum()), 1)
    y_val_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(34, 41)])
    y_test_all = np.concatenate([snapshots[t].y[snapshots[t].labeled_mask].numpy() for t in range(41, 49)])

    # Experiment A preprocessing: Train-only RankGauss (Steps 1-34 fit only)
    qt, rg_snapshots, rg_diag = build_rankgauss_snapshots(snapshots)
    print(f"[Train-Only RankGauss PASS] {rg_diag}")

    raw_snaps_dev = [s.to(device) for s in snapshots]
    rg_snaps_dev = [s.to(device) for s in rg_snapshots]

    raw_bundles = [
        {
            "legacy_dir": build_gcn_sparse_adj(s.edge_index, s.num_nodes, use_undirected=False, device=device),
            "gcn_in": build_true_directed_gcn_adj(s.edge_index, s.num_nodes, direction="in", device=device),
            "gcn_out": build_true_directed_gcn_adj(s.edge_index, s.num_nodes, direction="out", device=device),
        }
        for s in snapshots
    ]
    rg_bundles = raw_bundles  # Topology identical; only features transformed

    seeds_used = [42, 123] if smoke_test else SEEDS
    max_epochs = 2 if smoke_test else 50
    patience = 15
    fold_epochs = 2 if smoke_test else 38
    final_epochs = 2 if smoke_test else 40

    # --- Experiment A: RankGauss across neural & tabular configs (Train 1-34, Val 35-41) ---
    print(f"\n[Experiment A] Training RankGauss-182 models (seeds={seeds_used}, max_epochs={max_epochs})...")
    expa_specs = [
        ("ExpA1_RankGauss182_GCNAvg", "GCNAvg_182"),
        ("ExpA2_RankGauss182_BiDirGCN", "BiDirGCN_182"),
        ("ExpA3_RankGauss182_NoGraph", "NoGraph_182"),
    ]
    expa_runs_by_id = {}
    for eid, arch_name in expa_specs:
        runs = []
        for s_val in seeds_used:
            r = train_single_expa_neural(
                eid, arch_name, s_val, rg_snaps_dev, rg_bundles, y_val_all, pos_weight_val,
                device, max_epochs=max_epochs, patience=patience, smoke_test=smoke_test
            )
            runs.append(r)
        expa_runs_by_id[eid] = runs

    # ExpA0: HistGradientBoosting on RankGauss-182 (sanity baseline)
    X_rg_tr = np.concatenate([rg_snapshots[t].x.numpy()[rg_snapshots[t].labeled_mask.numpy()] for t in range(34)], axis=0)
    X_rg_va = np.concatenate([rg_snapshots[t].x.numpy()[rg_snapshots[t].labeled_mask.numpy()] for t in range(34, 41)], axis=0)
    expa0_runs = []
    for s_val in seeds_used:
        clf_rg = HistGradientBoostingClassifier(max_iter=100, max_depth=8, learning_rate=0.1, class_weight="balanced", random_state=s_val)
        t0_c = time.time()
        clf_rg.fit(X_rg_tr, train_y)
        ft = time.time() - t0_c
        vp = clf_rg.predict_proba(X_rg_va)[:, 1]
        tau_rec, _, tau_r30 = select_thresholds_val_only(y_val_all, vp, 0.01, 0.99, 99, 0.30)
        vm = compute_metrics_and_calibration(y_val_all, vp, threshold=tau_rec)
        expa0_runs.append({
            "exp_id": "ExpA0_RankGauss182_HGB",
            "experiment_group": "ExpA_RankGauss",
            "seed": int(s_val),
            "feature_dim": 182,
            "training_time_seconds": round(ft, 3),
            "threshold_recal_01_99": tau_rec,
            "threshold_rec30_recal": tau_r30,
            "validation_metrics_recal": vm,
            "clf": clf_rg,
        })
    expa_runs_by_id["ExpA0_RankGauss182_HGB"] = expa0_runs

    # --- Experiment B: Leakage-Free OOF GNN Embeddings + Orig 182 Features -> HistGradientBoosting ---
    print(f"\n[Experiment B] Generating 5-fold OOF GNN embeddings (Train 1-34 only) & training Hybrid HGB...")
    oof_cache_local93 = {}
    oof_cache_rg182 = {}
    for s_val in seeds_used:
        print(f"  Generating OOF embeddings [local93_gcnavg | seed={s_val}]...")
        emb_l93, enc_l93 = generate_leakage_free_oof_embeddings(
            "local93_gcnavg", s_val, raw_snaps_dev, raw_bundles, pos_weight_val, device, fold_epochs=fold_epochs, final_epochs=final_epochs
        )
        oof_cache_local93[s_val] = (emb_l93, enc_l93)

        print(f"  Generating OOF embeddings [rg182_bidir | seed={s_val}]...")
        emb_rg, enc_rg = generate_leakage_free_oof_embeddings(
            "rg182_bidir", s_val, rg_snaps_dev, rg_bundles, pos_weight_val, device, fold_epochs=fold_epochs, final_epochs=final_epochs
        )
        oof_cache_rg182[s_val] = (emb_rg, enc_rg)

    expb_configs = [
        ("ExpB1_Hybrid_GCNAvg93Emb_HGB", oof_cache_local93, "orig182_plus_emb64",
         {"max_iter": 100, "max_depth": 8, "learning_rate": 0.1, "class_weight": "balanced"}),
        ("ExpB2_Hybrid_GCNAvg93Emb_RegHGB", oof_cache_local93, "orig182_plus_emb64",
         {"max_iter": 150, "max_depth": 6, "min_samples_leaf": 40, "l2_regularization": 2.0, "max_features": 0.6, "learning_rate": 0.08, "class_weight": "balanced"}),
        ("ExpB3_Hybrid_RG182BiDirEmb_HGB", oof_cache_rg182, "orig182_plus_emb64",
         {"max_iter": 100, "max_depth": 8, "learning_rate": 0.1, "class_weight": "balanced"}),
        ("ExpB4_Hybrid_RG182BiDirEmb_RegHGB", oof_cache_rg182, "orig182_plus_emb64",
         {"max_iter": 150, "max_depth": 6, "min_samples_leaf": 40, "l2_regularization": 2.0, "max_features": 0.6, "learning_rate": 0.08, "class_weight": "balanced"}),
        ("ExpB5_Hybrid_GCNAvg93Logit1_HGB", oof_cache_local93, "orig182_plus_logit1",
         {"max_iter": 100, "max_depth": 8, "learning_rate": 0.1, "class_weight": "balanced"}),
    ]

    expb_runs_by_id = {}
    for eid, cache_map, fmode, hkwargs in expb_configs:
        runs = []
        for s_val in seeds_used:
            emb_dict, final_enc = cache_map[s_val]
            r = train_hybrid_hgb_on_oof(eid, s_val, snapshots, emb_dict, hkwargs, feature_mode=fmode)
            r["emb_by_step"] = emb_dict
            r["final_encoder"] = final_enc
            runs.append(r)
            print(f"    [{eid} | seed={s_val}] Val PR-AUC={r['validation_metrics_recal']['pr_auc']:.4f} | Val F1={r['validation_metrics_recal']['illicit_f1']:.4f} | Brier={r['validation_metrics_recal']['brier_score']:.4f}")
        expb_runs_by_id[eid] = runs

    # --- VALIDATION-ONLY MODEL SELECTION (Test 42-49 completely sealed until here) ---
    all_groups = {**expa_runs_by_id, **expb_runs_by_id}
    val_ranking = sorted(
        [(eid, float(np.mean([r["validation_metrics_recal"]["pr_auc"] for r in runs]))) for eid, runs in all_groups.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    best_expa_id = max(expa_runs_by_id.keys(), key=lambda k: np.mean([r["validation_metrics_recal"]["pr_auc"] for r in expa_runs_by_id[k]]))
    best_expb_id = max(expb_runs_by_id.keys(), key=lambda k: np.mean([r["validation_metrics_recal"]["pr_auc"] for r in expb_runs_by_id[k]]))
    print("\n" + "=" * 88)
    print(f"[Validation-Only Winner ExpA] {best_expa_id} | [ExpB] {best_expb_id} | [Overall] {val_ranking[0][0]}")
    print("=" * 88)

    if smoke_test:
        # Verify temp checkpoint save/load & clean exit without touching Test 42-49 or disk
        with tempfile.TemporaryDirectory() as tmpdir:
            r0 = expa_runs_by_id["ExpA1_RankGauss182_GCNAvg"][0]
            p_tmp = os.path.join(tmpdir, "test_ckpt.pt")
            torch.save(r0["best_state"], p_tmp)
            sd2 = torch.load(p_tmp, weights_only=True)
            assert all(torch.equal(r0["best_state"][k], sd2[k]) for k in sd2)
        print("[Smoke Test PASS] All RankGauss + 5-fold OOF Hybrid checks passed. Zero production writes.")
        return {"smoke_test": True, "val_ranking": val_ranking}

    # --- PHASE C: POST-SELECTION TEST EVALUATION (Steps 42-49) ---
    # 1. Evaluate Experiment A runs on Test
    for eid, runs in expa_runs_by_id.items():
        for r in runs:
            if eid == "ExpA0_RankGauss182_HGB":
                sp_test = {}
                for t in range(41, 49):
                    s = rg_snapshots[t]
                    lm = s.labeled_mask.numpy()
                    sp_test[int(s.timestep)] = {"y": s.y.numpy()[lm], "probs": r["clf"].predict_proba(s.x.numpy()[lm])[:, 1]}
            else:
                m = build_expa_model(r["arch_name"], device)
                m.load_state_dict(r["best_state"])
                _, sp_test = rollout_expa(m, r["arch_name"], rg_snaps_dev, rg_bundles, 41, 49)
            r.update(eval_slices_from_step_outputs(sp_test, r["threshold_recal_01_99"]))

    # 2. Evaluate Experiment B runs on Test
    for eid, runs in expb_runs_by_id.items():
        for r in runs:
            _, _, step_x_te, step_y_te = r["collect_xy_fn"](41, 49)
            sp_test = {
                st: {"y": step_y_te[st], "probs": r["clf"].predict_proba(step_x_te[st])[:, 1]}
                for st in range(42, 50)
            }
            r.update(eval_slices_from_step_outputs(sp_test, r["threshold_recal_01_99"]))

    # 3. Baselines (Existing frozen HistGradientBoosting & GCNAvg Stage4F champion)
    hgb_base = joblib.load(os.path.join(base_dir, "models", "hist_gradient_boosting.joblib"))
    s3_meta = json.loads(open(os.path.join(base_dir, "reports", "stage3_baselines_metrics.json"), "r", encoding="utf-8").read())
    hgb_tau = float(s3_meta["Gradient_Boosting"]["optimal_threshold"])
    hgb_step_test = {
        int(snapshots[t].timestep): {
            "y": snapshots[t].y.numpy()[snapshots[t].labeled_mask.numpy()],
            "probs": hgb_base.predict_proba(snapshots[t].x.numpy()[snapshots[t].labeled_mask.numpy()])[:, 1],
        }
        for t in range(41, 49)
    }
    hgb_base_res = eval_slices_from_step_outputs(hgb_step_test, hgb_tau)

    gcn_ckpt = torch.load(os.path.join(base_dir, "models", "stage4f", "stage4f_champion.pt"), weights_only=False, map_location=device)
    gcn_model = SeparatedGraphNet(in_channels=93, hidden_channels=128, out_channels=64, dropout_rate=0.3, graph_mode="gcn_avg").to(device)
    gcn_model.load_state_dict({k.replace("core.", ""): v for k, v in gcn_ckpt["model_state_dict"].items()})
    gcn_model.eval()
    gcn_tau = float(gcn_ckpt["threshold_recal_01_99"])
    gcn_step_test = {}
    with torch.no_grad():
        for t in range(41, 49):
            s = raw_snaps_dev[t]
            logits, _, _ = gcn_model.forward_step(s.x[:, :93], raw_bundles[t]["legacy_dir"], None)
            lm = s.labeled_mask.numpy()
            gcn_step_test[int(s.timestep)] = {"y": s.y.numpy()[lm], "probs": torch.sigmoid(logits).numpy()[lm]}
    gcn_base_res = eval_slices_from_step_outputs(gcn_step_test, gcn_tau)

    # Summarize all configs
    summaries = []
    for eid, runs in all_groups.items():
        summaries.append({
            "exp_id": eid,
            "experiment_group": runs[0]["experiment_group"],
            "seeds": seeds_used,
            "summary": summarize_runs_mean_std(runs),
            "best_val_seed": int(max(runs, key=lambda x: x["validation_metrics_recal"]["pr_auc"])["seed"]),
        })

    # Save checkpoints and outputs strictly in Stage 4H paths
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(qt, os.path.join(models_dir, "rankgauss_transformer_train_1_34.joblib"))
    for eid, runs in expa_runs_by_id.items():
        for r in runs:
            if eid == "ExpA0_RankGauss182_HGB":
                joblib.dump({"clf": r["clf"], "tau": r["threshold_recal_01_99"], "seed": r["seed"]},
                            os.path.join(models_dir, f"{eid}_seed{r['seed']}.joblib"))
            else:
                torch.save({
                    "exp_id": eid, "arch_name": r["arch_name"], "seed": r["seed"], "best_epoch": r["best_epoch"],
                    "total_parameters": r["total_parameters"], "model_state_dict": r["best_state"],
                    "threshold_recal_01_99": r["threshold_recal_01_99"], "validation_metrics_recal": r["validation_metrics_recal"],
                    "test_all_42_49": r["test_all_42_49"],
                }, os.path.join(models_dir, f"{eid}_seed{r['seed']}.pt"))

    for eid, runs in expb_runs_by_id.items():
        for r in runs:
            joblib.dump({
                "exp_id": eid, "seed": r["seed"], "feature_mode": r["feature_mode"],
                "hgb_kwargs": r["hgb_kwargs"], "clf": r["clf"], "tau": r["threshold_recal_01_99"],
                "validation_metrics_recal": r["validation_metrics_recal"], "test_all_42_49": r["test_all_42_49"],
            }, os.path.join(models_dir, f"{eid}_seed{r['seed']}.joblib"))
    for s_val in seeds_used:
        torch.save({"model_state_dict": oof_cache_local93[s_val][1].state_dict(), "encoder": "local93_gcnavg", "seed": s_val},
                   os.path.join(models_dir, f"oof_final_encoder_local93_seed{s_val}.pt"))
        torch.save({"model_state_dict": oof_cache_rg182[s_val][1].state_dict(), "encoder": "rg182_bidir", "seed": s_val},
                   os.path.join(models_dir, f"oof_final_encoder_rg182_seed{s_val}.pt"))

    best_a_run = max(expa_runs_by_id[best_expa_id], key=lambda x: x["validation_metrics_recal"]["pr_auc"])
    best_b_run = max(expb_runs_by_id[best_expb_id], key=lambda x: x["validation_metrics_recal"]["pr_auc"])

    save_stage4h_plots(
        out_dir, summaries,
        best_a_run["per_snapshot_test_metrics"],
        best_b_run["per_snapshot_test_metrics"],
        hgb_base_res["per_snapshot_test_metrics"],
        gcn_base_res["per_snapshot_test_metrics"],
    )

    clean_runs = {}
    for eid, runs in all_groups.items():
        clean_runs[eid] = [
            {k: v for k, v in r.items() if k not in ("best_state", "clf", "emb_by_step", "final_encoder", "collect_xy_fn")}
            for r in runs
        ]

    payload = {
        "stage": "Stage 4H - Distribution-Robust Feature Learning & Hybrid Model Investigation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds_used,
        "rankgauss_diagnostics": rg_diag,
        "validation_selected_winners": {
            "best_exp_a": best_expa_id,
            "best_exp_b": best_expb_id,
            "overall_validation_winner": val_ranking[0][0],
        },
        "baselines_post_selection": {
            "HistGradientBoosting": {"threshold": hgb_tau, **hgb_base_res},
            "GCNAvg_Stage4F_Champion": {"threshold": gcn_tau, **gcn_base_res},
        },
        "summaries": summaries,
        "individual_runs": clean_runs,
    }

    with open(os.path.join(out_dir, "stage4h_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(out_dir, "stage4h_report.md"), "w", encoding="utf-8") as f:
        f.write(generate_stage4h_report(payload, best_a_run, best_b_run))

    print(f"\nSaved all Stage 4H artifacts in {out_dir} and {models_dir}.")
    print(f"Stage 4H completed in {time.time() - t_start:.2f}s.")
    print("=" * 88)
    return payload


def generate_stage4h_report(payload, best_a_run, best_b_run):
    hb = payload["baselines_post_selection"]["HistGradientBoosting"]
    gb = payload["baselines_post_selection"]["GCNAvg_Stage4F_Champion"]
    w = payload["validation_selected_winners"]
    rg = payload["rankgauss_diagnostics"]

    md = f"""# Stage 4H: Distribution-Robust Feature Learning (RankGauss) & Hybrid GNN+HGB Investigation

**Generated On:** {payload['timestamp']}  
**Seeds:** `{payload['seeds']}` | **Protocol:** Train `1–34` (incl. RankGauss fit & 5-fold snapshot OOF GNN embeddings) $\\to$ Validate/Select & Calibrate (`0.01–0.99`) on `Steps 35–41` $\\to$ Post-Selection Test on `Steps 42–49`  
**Validation-Selected Winners:** Exp A = `{w['best_exp_a']}`, Exp B = `{w['best_exp_b']}`, Overall = `{w['overall_validation_winner']}`  
**RankGauss Tail Compression Verified:** `Steps 43–49` `abs_max = {rg['steps43_49_abs_max']}`, `frac(|z|>5) = {rg['steps43_49_frac_gt5_pct']}%`

---

## 1. Multi-Seed Validation & Post-Selection Test Summary (Mean ± Std)

| Config | Val PR-AUC (`35–41`) | Test All (`42–49`) PR-AUC / ROC | Test F1 / Prec / Rec | Test Brier / ECE | Step 42 PR / ROC | Steps `43–49` PR / ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| *HistGradientBoosting (Base)* | *0.9292* | *0.5646 / 0.8848* | *0.6166 / 88.53% / 47.30%* | *0.0362 / 0.0396* | *0.8913 / 0.9653* | *0.0552 / 0.7698* |
| *GCNAvg Champion (Base)* | *0.9051* | *0.5268 / 0.8188* | *0.5965 / 85.39% / 45.83%* | *0.0740 / 0.1145* | *0.8829 / 0.9544* | *0.0316 / 0.6247* |
"""
    for s in payload["summaries"]:
        sm = s["summary"]
        v = sm["val_35_41"]
        ta = sm["test_all_42_49"]
        s42 = sm["test_step42_only"]
        s43 = sm["test_steps43_49_only"]
        md += (
            f"| `{s['exp_id']}` | `{v['pr_auc']['mean']:.4f} ± {v['pr_auc']['std']:.4f}` | "
            f"`{ta['pr_auc']['mean']:.4f} ± {ta['pr_auc']['std']:.4f}` / `{ta['roc_auc']['mean']:.4f}` | "
            f"`{ta['illicit_f1']['mean']:.4f}` / `{ta['illicit_precision']['mean']*100:.2f}%` / `{ta['illicit_recall']['mean']*100:.2f}%` | "
            f"`{ta['brier_score']['mean']:.4f}` / `{ta['ece_10bin']['mean']:.4f}` | "
            f"`{s42['pr_auc']['mean']:.4f}` / `{s42['roc_auc']['mean']:.4f}` | "
            f"`{s43['pr_auc']['mean']:.4f}` / `{s43['roc_auc']['mean']:.4f}` |\n"
        )

    md += "\n---\n\n## 2. Confusion Matrices & Per-Snapshot Test Breakdown (`Steps 42–49`)\n\n"
    for label, run_obj in [("Best Exp A (`seed` by Val)", best_a_run), ("Best Exp B (`seed` by Val)", best_b_run)]:
        ta = run_obj["test_all_42_49"]
        cm = ta["confusion_matrix"]
        md += (
            f"- **{label} — `{run_obj['exp_id']}` (`seed={run_obj['seed']}`, $\\tau^*={run_obj['threshold_recal_01_99']:.2f}$):** "
            f"PR=`{ta['pr_auc']:.4f}`, ROC=`{ta['roc_auc']:.4f}`, F1=`{ta['illicit_f1']:.4f}`, "
            f"CM: `TP={cm['true_positives_caught_fraud']}, FP={cm['false_positives_false_alarm']}, TN={cm['true_negatives_licit']}, FN={cm['false_negatives_missed_fraud']}`\n"
        )

    md += "\n| Step | HGB PR / ROC | GCNAvg PR / ROC | Best Exp A PR / ROC | Best Exp B PR / ROC | Best Exp A TP/FP/FN | Best Exp B TP/FP/FN |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for st in range(42, 50):
        k = str(st)
        hs = hb["per_snapshot_test_metrics"][k]
        gs = gb["per_snapshot_test_metrics"][k]
        sa = best_a_run["per_snapshot_test_metrics"][k]
        sb = best_b_run["per_snapshot_test_metrics"][k]
        md += (
            f"| `{st}` | `{hs['pr_auc']:.4f}` / `{hs['roc_auc']:.4f}` | `{gs['pr_auc']:.4f}` / `{gs['roc_auc']:.4f}` | "
            f"`{sa['pr_auc']:.4f}` / `{sa['roc_auc']:.4f}` | `{sb['pr_auc']:.4f}` / `{sb['roc_auc']:.4f}` | "
            f"`{sa['tp']}/{sa['fp']}/{sa['fn']}` | `{sb['tp']}/{sb['fp']}/{sb['fn']}` |\n"
        )

    return md


if __name__ == "__main__":
    is_smoke = "--smoke-test" in sys.argv or "--quick" in sys.argv
    run_stage4h(smoke_test=is_smoke)
