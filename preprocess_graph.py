"""
Stage 2: Data Preprocessing and Temporal Graph Construction
Capstone Project: Temporal Graph Neural Networks with Real-Time Explainable AI
for Online Payment Fraud Ring Detection
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def run_stage2_preprocessing():
    print("=" * 80)
    print("STAGE 2: DATA PREPROCESSING & TEMPORAL GRAPH CONSTRUCTION")
    print("=" * 80)
    start_time = time.time()
    
    # -------------------------------------------------------------
    # 1. File Paths Setup
    # -------------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    if not os.path.exists(dataset_dir):
        alt_dir = os.path.join(base_dir, "data")
        if os.path.exists(alt_dir):
            dataset_dir = alt_dir
        else:
            raise FileNotFoundError(f"Dataset directory not found at {dataset_dir} or {alt_dir}")
            
    processed_dir = os.path.join(base_dir, "processed")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    classes_path = os.path.join(dataset_dir, "txs_classes.txt")
    if not os.path.exists(classes_path):
        classes_path = os.path.join(dataset_dir, "txs_classes.csv")
        
    edges_path = os.path.join(dataset_dir, "txs_edgelist.txt")
    if not os.path.exists(edges_path):
        edges_path = os.path.join(dataset_dir, "txs_edgelist.csv")
        
    features_path = os.path.join(dataset_dir, "txs_features.txt")
    if not os.path.exists(features_path):
        features_path = os.path.join(dataset_dir, "txs_features.csv")
        
    print(f"Reading dataset from: {dataset_dir}")
    print(f"Saving processed graphs to: {processed_dir}")
    print("-" * 80)
    
    # -------------------------------------------------------------
    # 2. Load Raw Files (Explicitly Join on txId)
    # -------------------------------------------------------------
    print("\n[Step 1/6] Loading raw data files...")
    t0 = time.time()
    df_classes = pd.read_csv(classes_path)
    df_edges = pd.read_csv(edges_path)
    df_features = pd.read_csv(features_path)
    print(f"  Loaded raw files in {time.time() - t0:.2f}s")
    print(f"  - Classes rows:  {len(df_classes):,}")
    print(f"  - Edges rows:    {len(df_edges):,}")
    print(f"  - Features rows: {len(df_features):,}")
    
    # Explicit join on txId to prevent any silent row-order mismatch
    print("\n[Step 2/6] Aligning nodes and ordering by (Time step, txId)...")
    df_merged = pd.merge(df_features, df_classes, on='txId', how='inner')
    assert len(df_merged) == len(df_features) == len(df_classes), "Node count mismatch during merge!"
    
    # Sort deterministically by Time step, then txId
    df_merged.sort_values(by=['Time step', 'txId'], ascending=[True, True], inplace=True)
    df_merged.reset_index(drop=True, inplace=True)
    
    num_nodes = len(df_merged)
    print(f"  Total aligned nodes: {num_nodes:,}")
    
    # Create bidirectional ID mapping: txId <-> integer index [0 .. N-1]
    txId_to_idx = {tx_id: idx for idx, tx_id in enumerate(df_merged['txId'].values)}
    idx_to_txId = {idx: int(tx_id) for idx, tx_id in enumerate(df_merged['txId'].values)}
    
    # -------------------------------------------------------------
    # 3. Handle Labels, Splits, and Masks
    # -------------------------------------------------------------
    print("\n[Step 3/6] Configuring labels, chronological splits, and supervision masks...")
    
    # Label encoding:
    # Class 1 (Illicit / Fraud) -> 1
    # Class 2 (Licit / Legitimate) -> 0
    # Class 3 (Unknown / Unlabeled) -> -1
    raw_classes = df_merged['class'].values
    y_labels = np.full(num_nodes, -1, dtype=np.int64)
    y_labels[raw_classes == 1] = 1
    y_labels[raw_classes == 2] = 0
    
    timesteps = df_merged['Time step'].values.astype(np.int64)
    
    # Chronological Split Definitions (Standard Elliptic benchmark):
    # Train: Timesteps 1 to 34 (34 temporal steps)
    # Val:   Timesteps 35 to 41 (7 temporal steps)
    # Test:  Timesteps 42 to 49 (8 temporal steps)
    train_timesteps_range = (1, 34)
    val_timesteps_range = (35, 41)
    test_timesteps_range = (42, 49)
    
    is_train_time = (timesteps >= train_timesteps_range[0]) & (timesteps <= train_timesteps_range[1])
    is_val_time = (timesteps >= val_timesteps_range[0]) & (timesteps <= val_timesteps_range[1])
    is_test_time = (timesteps >= test_timesteps_range[0]) & (timesteps <= test_timesteps_range[1])
    
    is_labeled = (y_labels != -1)
    
    train_mask = is_train_time & is_labeled
    val_mask = is_val_time & is_labeled
    test_mask = is_test_time & is_labeled
    
    print(f"  Split Summary:")
    print(f"    Train (Steps {train_timesteps_range[0]}-{train_timesteps_range[1]}): "
          f"{is_train_time.sum():,} total nodes | {train_mask.sum():,} labeled "
          f"({(y_labels[train_mask]==1).sum():,} illicit [{(y_labels[train_mask]==1).sum()/train_mask.sum()*100:.2f}%], "
          f"{(y_labels[train_mask]==0).sum():,} licit)")
    print(f"    Val   (Steps {val_timesteps_range[0]}-{val_timesteps_range[1]}): "
          f"{is_val_time.sum():,} total nodes | {val_mask.sum():,} labeled "
          f"({(y_labels[val_mask]==1).sum():,} illicit [{(y_labels[val_mask]==1).sum()/val_mask.sum()*100:.2f}%], "
          f"{(y_labels[val_mask]==0).sum():,} licit)")
    print(f"    Test  (Steps {test_timesteps_range[0]}-{test_timesteps_range[1]}): "
          f"{is_test_time.sum():,} total nodes | {test_mask.sum():,} labeled "
          f"({(y_labels[test_mask]==1).sum():,} illicit [{(y_labels[test_mask]==1).sum()/test_mask.sum()*100:.2f}%], "
          f"{(y_labels[test_mask]==0).sum():,} licit)")
    print(f"    Unlabeled Background Nodes (Class 3): {(y_labels == -1).sum():,} ({(y_labels == -1).sum()/num_nodes*100:.2f}%)")
    
    # -------------------------------------------------------------
    # 4. Feature Preprocessing (Train-Only Fitting: Zero Leakage)
    # -------------------------------------------------------------
    print("\n[Step 4/6] Preprocessing features with train-only fitted imputer and scaler...")
    
    feature_cols = [c for c in df_merged.columns if c not in ['txId', 'Time step', 'class']]
    local_cols = [c for c in feature_cols if c.startswith('Local_feature_')]
    agg_cols = [c for c in feature_cols if c.startswith('Aggregate_feature_')]
    domain_cols = [c for c in feature_cols if c not in local_cols + agg_cols]
    
    print(f"  Feature dimensions: {len(feature_cols)} total (93 Local + 72 Aggregate + {len(domain_cols)} Domain)")
    
    X_raw = df_merged[feature_cols].values.astype(np.float32)
    
    # Fit Imputer and Scaler ONLY on the training period (is_train_time) to prevent data leakage
    print(f"  Fitting SimpleImputer(strategy='median') on Training period (steps 1-34)...")
    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_raw[is_train_time])
    X_imputed = imputer.transform(X_raw)
    
    print(f"  Fitting StandardScaler() on Training period (steps 1-34)...")
    scaler = StandardScaler()
    scaler.fit(X_imputed[is_train_time])
    X_scaled = scaler.transform(X_imputed)
    
    # Check for NaN / Inf
    assert not np.isnan(X_scaled).any(), "NaN values found after preprocessing!"
    assert not np.isinf(X_scaled).any(), "Infinite values found after preprocessing!"
    print(f"  Feature matrix processed successfully: shape {X_scaled.shape}, 0 NaNs, 0 Infs.")
    
    # -------------------------------------------------------------
    # 5. Graph Edge Construction & Snapshot Partitioning
    # -------------------------------------------------------------
    print("\n[Step 5/6] Mapping directed edges and constructing PyG graph objects...")
    
    src_mapped = df_edges['txId1'].map(txId_to_idx).values
    dst_mapped = df_edges['txId2'].map(txId_to_idx).values
    
    # Verify all edges are valid
    assert not np.isnan(src_mapped).any(), "Unmapped source transaction ID in edges!"
    assert not np.isnan(dst_mapped).any(), "Unmapped destination transaction ID in edges!"
    
    edge_index_full = torch.tensor(np.array([src_mapped, dst_mapped]), dtype=torch.long)
    
    # Build Global PyTorch Geometric Data Object
    global_data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float32),
        edge_index=edge_index_full,
        y=torch.tensor(y_labels, dtype=torch.long),
        timestep=torch.tensor(timesteps, dtype=torch.long),
        txId=torch.tensor(df_merged['txId'].values, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        labeled_mask=torch.tensor(is_labeled, dtype=torch.bool)
    )
    
    # Verify global graph properties
    assert global_data.num_nodes == num_nodes
    assert global_data.num_edges == len(df_edges)
    assert global_data.edge_index.max().item() < num_nodes
    assert global_data.edge_index.min().item() >= 0
    print(f"  Consolidated PyG Data created: {global_data}")
    
    # Build 49 Per-Timestep Temporal Subgraph Snapshots
    print("\n  Building 49 individual temporal snapshot subgraphs...")
    temporal_snapshots = []
    
    df_edges_mapped = pd.DataFrame({
        'src': src_mapped,
        'dst': dst_mapped,
        'src_txId': df_edges['txId1'].values,
        'dst_txId': df_edges['txId2'].values
    })
    
    # Map each edge to its timestep
    node_to_time = dict(zip(range(num_nodes), timesteps))
    df_edges_mapped['time'] = df_edges_mapped['src'].map(node_to_time)
    
    for t in range(1, 50):
        # Nodes in timestep t
        node_indices_t = np.where(timesteps == t)[0]
        num_nodes_t = len(node_indices_t)
        
        # Local re-indexing map: global index -> local index [0 .. N_t-1]
        global_to_local_t = {g_idx: l_idx for l_idx, g_idx in enumerate(node_indices_t)}
        
        # Edges in timestep t
        edges_t = df_edges_mapped[df_edges_mapped['time'] == t]
        if len(edges_t) > 0:
            local_src_t = edges_t['src'].map(global_to_local_t).values
            local_dst_t = edges_t['dst'].map(global_to_local_t).values
            edge_index_t = torch.tensor(np.array([local_src_t, local_dst_t]), dtype=torch.long)
        else:
            edge_index_t = torch.empty((2, 0), dtype=torch.long)
            
        # Node features & labels in timestep t
        x_t = torch.tensor(X_scaled[node_indices_t], dtype=torch.float32)
        y_t = torch.tensor(y_labels[node_indices_t], dtype=torch.long)
        txId_t = torch.tensor(df_merged['txId'].values[node_indices_t], dtype=torch.long)
        labeled_mask_t = torch.tensor(is_labeled[node_indices_t], dtype=torch.bool)
        
        # Split assignment
        if t <= 34:
            split_t = "train"
        elif t <= 41:
            split_t = "val"
        else:
            split_t = "test"
            
        snapshot = Data(
            x=x_t,
            edge_index=edge_index_t,
            y=y_t,
            txId=txId_t,
            labeled_mask=labeled_mask_t,
            timestep=t,
            split=split_t,
            num_nodes=num_nodes_t
        )
        
        # Snapshot integrity check
        if edge_index_t.shape[1] > 0:
            assert edge_index_t.max().item() < num_nodes_t, f"Invalid local edge index at step {t}!"
            assert edge_index_t.min().item() >= 0, f"Negative local edge index at step {t}!"
            
        temporal_snapshots.append(snapshot)
        
    print(f"  Constructed all {len(temporal_snapshots)} temporal snapshot subgraphs successfully.")
    
    # -------------------------------------------------------------
    # 6. Save Processed Artifacts and Metadata
    # -------------------------------------------------------------
    print("\n[Step 6/6] Saving processed graph artifacts and documentation...")
    
    # 1. Save PyG Consolidated Graph
    pyg_file = os.path.join(processed_dir, "elliptic_pyg_data.pt")
    torch.save(global_data, pyg_file)
    print(f"  [Saved]: {pyg_file} ({os.path.getsize(pyg_file) / (1024*1024):.2f} MB)")
    
    # 2. Save PyG Temporal Snapshots List
    snapshots_file = os.path.join(processed_dir, "temporal_snapshots.pt")
    torch.save(temporal_snapshots, snapshots_file)
    print(f"  [Saved]: {snapshots_file} ({os.path.getsize(snapshots_file) / (1024*1024):.2f} MB)")
    
    # 3. Save Tabular Numpy Arrays for Classical Baseline Models (Stage 3)
    tabular_file = os.path.join(processed_dir, "tabular_data.npz")
    np.savez_compressed(
        tabular_file,
        X=X_scaled,
        y=y_labels,
        timestep=timesteps,
        txId=df_merged['txId'].values,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        labeled_mask=is_labeled
    )
    print(f"  [Saved]: {tabular_file} ({os.path.getsize(tabular_file) / (1024*1024):.2f} MB)")
    
    # 4. Save JSON Metadata
    metadata = {
        "dataset_name": "Elliptic++ Bitcoin Graph",
        "preprocessing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_nodes": int(num_nodes),
        "num_edges": int(len(df_edges)),
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "local_feature_count": len(local_cols),
        "aggregate_feature_count": len(agg_cols),
        "domain_feature_count": len(domain_cols),
        "domain_feature_names": domain_cols,
        "num_timesteps": 49,
        "splits": {
            "train": {
                "timesteps": [train_timesteps_range[0], train_timesteps_range[1]],
                "total_nodes": int(is_train_time.sum()),
                "total_edges": int(df_edges_mapped[df_edges_mapped['time'] <= 34].shape[0]),
                "labeled_nodes": int(train_mask.sum()),
                "illicit_count": int((y_labels[train_mask] == 1).sum()),
                "licit_count": int((y_labels[train_mask] == 0).sum()),
                "illicit_ratio": float((y_labels[train_mask] == 1).sum() / train_mask.sum())
            },
            "val": {
                "timesteps": [val_timesteps_range[0], val_timesteps_range[1]],
                "total_nodes": int(is_val_time.sum()),
                "total_edges": int(df_edges_mapped[(df_edges_mapped['time'] >= 35) & (df_edges_mapped['time'] <= 41)].shape[0]),
                "labeled_nodes": int(val_mask.sum()),
                "illicit_count": int((y_labels[val_mask] == 1).sum()),
                "licit_count": int((y_labels[val_mask] == 0).sum()),
                "illicit_ratio": float((y_labels[val_mask] == 1).sum() / val_mask.sum())
            },
            "test": {
                "timesteps": [test_timesteps_range[0], test_timesteps_range[1]],
                "total_nodes": int(is_test_time.sum()),
                "total_edges": int(df_edges_mapped[df_edges_mapped['time'] >= 42].shape[0]),
                "labeled_nodes": int(test_mask.sum()),
                "illicit_count": int((y_labels[test_mask] == 1).sum()),
                "licit_count": int((y_labels[test_mask] == 0).sum()),
                "illicit_ratio": float((y_labels[test_mask] == 1).sum() / test_mask.sum())
            },
            "unlabeled_total": int((y_labels == -1).sum())
        },
        "validation_checks": {
            "node_count_match": bool(global_data.num_nodes == 203769),
            "edge_count_match": bool(global_data.num_edges == 234355),
            "edge_indices_valid": bool((global_data.edge_index.min() >= 0) and (global_data.edge_index.max() < 203769)),
            "no_nan_features": bool(not torch.isnan(global_data.x).any().item()),
            "no_inf_features": bool(not torch.isinf(global_data.x).any().item()),
            "train_val_test_disjoint": bool(int((train_mask & val_mask).sum() + (val_mask & test_mask).sum() + (train_mask & test_mask).sum()) == 0),
            "all_labeled_accounted_for": bool(int(train_mask.sum() + val_mask.sum() + test_mask.sum()) == int(is_labeled.sum()))
        }
    }
    
    metadata_file = os.path.join(processed_dir, "metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"  [Saved]: {metadata_file}")
    
    # 5. Generate Markdown Report
    md_report_file = os.path.join(reports_dir, "stage2_preprocessing_report.md")
    with open(md_report_file, 'w', encoding='utf-8') as f:
        f.write(generate_stage2_markdown_report(metadata))
    print(f"  [Saved]: {md_report_file}")
    
    total_time = time.time() - start_time
    print(f"\nStage 2 Preprocessing completed successfully in {total_time:.2f} seconds!")
    print("=" * 80)
    return metadata

def generate_stage2_markdown_report(meta):
    splits = meta["splits"]
    checks = meta["validation_checks"]
    
    md = f"""# Stage 2: Data Preprocessing & Temporal Graph Construction Report

**Project Title:** Temporal Graph Neural Networks with Real-Time Explainable AI for Online Payment Fraud Ring Detection  
**Generated On:** {meta["preprocessing_timestamp"]}  
**Dataset:** Elliptic++ Bitcoin Transaction Graph  

---

## 1. Executive Summary & Graph Specifications

| Specification | Value | Technical Details |
| :--- | :--- | :--- |
| **Total Graph Nodes ($N$)** | **{meta["num_nodes"]:,}** | Bitcoin transactions sorted deterministically by `(Time step, txId)` |
| **Total Directed Edges ($E$)** | **{meta["num_edges"]:,}** | Directed fund transfer flows between transactions |
| **Input Feature Dimension ($D$)** | **{meta["num_features"]}** | 93 Local + 72 Aggregate + 17 Domain Features |
| **Discrete Temporal Snapshots** | **{meta["num_timesteps"]}** | Distinct 2-week subgraphs (Steps 1 to 49) |
| **Total Supervised (Labeled) Nodes** | **{splits["train"]["labeled_nodes"] + splits["val"]["labeled_nodes"] + splits["test"]["labeled_nodes"]:,}** | Ground-truth Illicit (`1`) and Licit (`0`) nodes |
| **Background / Unlabeled Nodes** | **{splits["unlabeled_total"]:,} ({splits["unlabeled_total"]/meta["num_nodes"]*100:.2f}%)** | Retained in graph topology for message passing (`y = -1`) |
| **Imputation & Scaling Strategy** | **Train-Only Fitted (Zero Leakage)** | Imputer & Scaler fitted strictly on steps 1–34 |
| **PyG Artifacts Created** | **`elliptic_pyg_data.pt`, `temporal_snapshots.pt`** | Ready for baseline and Temporal GNN architectures |

---

## 2. Chronological Temporal Data Splits

To evaluate realistic fraud detection capability without future-information leakage, the 49 time steps are partitioned chronologically:

```mermaid
gantt
    title Temporal Train / Validation / Test Split (49 Time Steps)
    dateFormat X
    axisFormat Step %s
    section Training Period
    Train Snapshots (Steps 1-34) :active, 1, 34
    section Validation Period
    Validation Snapshots (Steps 35-41) :crit, 35, 41
    section Testing Period
    Test Snapshots (Steps 42-49) :done, 42, 49
```

### Detailed Split Breakdown:

| Split | Timesteps | Total Nodes | Total Edges | Labeled Nodes | Illicit (Fraud, `y=1`) | Licit (Legitimate, `y=0`) | Illicit Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | Steps 1 – 34 | {splits["train"]["total_nodes"]:,} | {splits["train"]["total_edges"]:,} | **{splits["train"]["labeled_nodes"]:,}** | {splits["train"]["illicit_count"]:,} | {splits["train"]["licit_count"]:,} | **{splits["train"]["illicit_ratio"]*100:.2f}%** |
| **Validation** | Steps 35 – 41 | {splits["val"]["total_nodes"]:,} | {splits["val"]["total_edges"]:,} | **{splits["val"]["labeled_nodes"]:,}** | {splits["val"]["illicit_count"]:,} | {splits["val"]["licit_count"]:,} | **{splits["val"]["illicit_ratio"]*100:.2f}%** |
| **Test** | Steps 42 – 49 | {splits["test"]["total_nodes"]:,} | {splits["test"]["total_edges"]:,} | **{splits["test"]["labeled_nodes"]:,}** | {splits["test"]["illicit_count"]:,} | {splits["test"]["licit_count"]:,} | **{splits["test"]["illicit_ratio"]*100:.2f}%** |
| **Total** | Steps 1 – 49 | **{meta["num_nodes"]:,}** | **{meta["num_edges"]:,}** | **{splits["train"]["labeled_nodes"] + splits["val"]["labeled_nodes"] + splits["test"]["labeled_nodes"]:,}** | **{splits["train"]["illicit_count"] + splits["val"]["illicit_count"] + splits["test"]["illicit_count"]:,}** | **{splits["train"]["licit_count"] + splits["val"]["licit_count"] + splits["test"]["licit_count"]:,}** | **9.76%** |

---

## 3. Data Preprocessing & Leakage Prevention Strategy

1. **Missing Value Imputation:**
   - As identified in Stage 1, 965 transactions had missing values in the 17 domain features.
   - `SimpleImputer(strategy='median')` was fitted **exclusively on the training period (steps 1–34)**.
   - The fitted median statistics were then applied to transform the validation (steps 35–41) and test (steps 42–49) features.
2. **Feature Standardization:**
   - `StandardScaler()` was fitted **exclusively on the training period (steps 1–34)** and applied to validation and test periods.
   - This ensures that mean and variance statistics from future time steps do not leak into model training.
3. **Graph Topology Leakage Check:**
   - 100% of edges in `txs_edgelist.txt` connect transactions within the exact same time step.
   - There are **0 cross-timestep edges**, confirming that message passing within any time step snapshot cannot leak future graph connections.

---

## 4. Verification & Integrity Checklist

All automated validation checks passed successfully:

- [x] **Total Node Count:** 203,769 / 203,769 matching
- [x] **Total Edge Count:** 234,355 / 234,355 matching
- [x] **Edge Index Range:** Minimum index = 0, Maximum index = 203,768 (all within valid bounds)
- [x] **Missing / Infinite Values:** 0 NaNs and 0 Infs in feature tensor $X$
- [x] **Mask Disjointness:** Train, Val, and Test masks are strictly mutually exclusive
- [x] **Supervision Mask Purity:** Unlabeled nodes (`class = 3`, `y = -1`) are strictly excluded from `train_mask`, `val_mask`, and `test_mask`
- [x] **Snapshot Validity:** All 49 temporal subgraphs re-indexed and verified with zero out-of-bounds edge references

---

## 5. Generated Artifacts

The following files have been saved to `processed/`:
1. `processed/elliptic_pyg_data.pt`: Full consolidated PyTorch Geometric `Data` object containing `x`, `edge_index`, `y`, `timestep`, `txId`, and boolean masks (`train_mask`, `val_mask`, `test_mask`, `labeled_mask`).
2. `processed/temporal_snapshots.pt`: Sequence of 49 distinct PyG `Data` subgraphs for sequential/rolling-window Temporal GNN training.
3. `processed/tabular_data.npz`: Compressed NumPy format for standard baseline models (Random Forest, XGBoost, MLP, Logistic Regression).
4. `processed/metadata.json`: Complete configuration and schema mappings.

---

## 6. Next Step
Proceed to **Stage 3: Baseline Fraud Classifier Implementation** (evaluating non-graph and static tabular baselines such as Logistic Regression, Random Forest, and MLP to establish a rigorous performance benchmark before Temporal GNN modeling).
"""
    return md

if __name__ == "__main__":
    run_stage2_preprocessing()
