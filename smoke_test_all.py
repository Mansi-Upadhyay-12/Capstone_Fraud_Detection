"""
Comprehensive Smoke Test Suite for Capstone Fraud Detection Pipeline
Tests all data loaders, model initializations, forward passes, and training loops.
"""

import os
import sys
import time
import torch
import numpy as np

# Ensure root in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def run_all_smoke_tests():
    print("=" * 80)
    print("STARTING END-TO-END SMOKE TEST SUITE")
    print("=" * 80)
    start_total = time.time()
    
    # -------------------------------------------------------------------------
    # Test 1: Data Artifact Loading & Integrity Check
    # -------------------------------------------------------------------------
    print("\n[Smoke Test 1/6] Validating Preprocessed Data Artifacts...")
    t0 = time.time()
    tab_path = os.path.join(base_dir, "processed", "tabular_data.npz")
    pyg_path = os.path.join(base_dir, "processed", "elliptic_pyg_data.pt")
    snaps_path = os.path.join(base_dir, "processed", "temporal_snapshots.pt")
    
    assert os.path.exists(tab_path), f"Missing {tab_path}"
    assert os.path.exists(pyg_path), f"Missing {pyg_path}"
    assert os.path.exists(snaps_path), f"Missing {snaps_path}"
    
    tab = np.load(tab_path)
    assert tab['X'].shape == (203769, 182), f"Unexpected X shape: {tab['X'].shape}"
    assert tab['y'].shape == (203769,), f"Unexpected y shape: {tab['y'].shape}"
    assert tab['train_mask'].sum() == 29894, f"Unexpected train_mask count: {tab['train_mask'].sum()}"
    assert tab['val_mask'].sum() == 7829, f"Unexpected val_mask count: {tab['val_mask'].sum()}"
    assert tab['test_mask'].sum() == 8841, f"Unexpected test_mask count: {tab['test_mask'].sum()}"
    
    pyg = torch.load(pyg_path, weights_only=False)
    assert pyg.num_nodes == 203769
    assert pyg.num_edges == 234355
    
    snaps = torch.load(snaps_path, weights_only=False)
    assert len(snaps) == 49
    print(f"  [PASSED] All 3 preprocessed data artifacts validated in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # Test 2: Baselines Smoke Test (LR, RF, HistGB, MLP)
    # -------------------------------------------------------------------------
    print("\n[Smoke Test 2/6] Running Tabular Baselines Smoke Test...")
    t0 = time.time()
    from src.train_baselines import train_and_evaluate_baselines
    base_res = train_and_evaluate_baselines(smoke_test=True)
    assert "Logistic_Regression" in base_res
    assert "Random_Forest" in base_res
    assert "Gradient_Boosting" in base_res
    assert "MLP_Neural_Network" in base_res
    print(f"  [PASSED] Tabular baselines smoke test passed in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # Test 3: Static GNN Smoke Test (GCN & GAT)
    # -------------------------------------------------------------------------
    print("\n[Smoke Test 3/6] Running Static GNN Smoke Test (GCN & GAT)...")
    t0 = time.time()
    from src.train_gnn_static import run_stage4a
    gnn_res = run_stage4a(smoke_test=True)
    assert "GCN" in gnn_res
    assert "GAT" in gnn_res
    print(f"  [PASSED] Static GNN smoke test passed in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # Test 4: Temporal GCN-LSTM Smoke Test
    # -------------------------------------------------------------------------
    print("\n[Smoke Test 4/6] Running Temporal GCN-LSTM Smoke Test...")
    t0 = time.time()
    from src.train_gcn_lstm import run_stage4b
    lstm_res = run_stage4b(smoke_test=True)
    assert "GCN_LSTM" in lstm_res
    print(f"  [PASSED] Temporal GCN-LSTM smoke test passed in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # Test 5: Historical-Only GCN-LSTM Smoke Test
    # -------------------------------------------------------------------------
    print("\n[Smoke Test 5/6] Running Historical-Only GCN-LSTM Smoke Test...")
    t0 = time.time()
    from src.train_gcn_lstm_historical import run_stage4b1_ablation
    hist_res = run_stage4b1_ablation(smoke_test=True)
    assert "GCN_LSTM_Historical" in hist_res
    print(f"  [PASSED] Historical-Only GCN-LSTM smoke test passed in {time.time() - t0:.2f}s")
    
    # -------------------------------------------------------------------------
    # Test 6: Explainability and Fraud Ring Detection Smoke Test
    # -------------------------------------------------------------------------
    print("\n[Smoke Test 6/6] Running Explainability & Ring Detection...")
    t0 = time.time()
    from src.explainability_and_rings import run_stage5_explainability_and_rings
    xai_res = run_stage5_explainability_and_rings()
    assert "feature_importance" in xai_res
    assert "gnn_subgraph_explanation" in xai_res
    assert "fraud_ring_analysis" in xai_res
    print(f"  [PASSED] Explainability & Ring Detection passed in {time.time() - t0:.2f}s")
    
    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"ALL 6 SMOKE TESTS PASSED SUCCESSFULLY IN {total_time:.2f} SECONDS!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = run_all_smoke_tests()
    sys.exit(0 if success else 1)
