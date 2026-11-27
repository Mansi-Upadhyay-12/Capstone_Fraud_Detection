# Stage 4F: Graph Model Performance Optimization Report

**Generated On:** 2026-11-26 16:03:37  
**Selection Protocol:** Train `Steps 1–34` $\to$ Validate/Select `Steps 35–41` (PR-AUC) $\to$ Post-Selection Test `Steps 42–49`  
**Selected Stage 4F Champion:** `F4F_00_Anchor_GCNAvg` (`seed=42`, `params=43,267`, `best_epoch=40`)  
**Post-Selection Test Metrics (`0.01–0.99`, $\tau^*=0.93$):** PR-AUC=`0.5268`, ROC-AUC=`0.8188`, F1=`0.5965`, Prec=`85.39%`, Rec=`45.83%`, Brier=`0.0740`, ECE=`0.1145`, CM=`TP=187/FP=32/TN=8401/FN=221`

---

## 1. Phase A Screening Results (`seed=42`, Validation Steps `35–41`)

| Exp ID | Group | Params | Best Ep | Val PR-AUC | Val ROC-AUC | Val F1 | Val Prec | Val Rec | Brier | ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F4F_00_Anchor_GCNAvg` | `Anchor` | `43,267` | 40 | **0.9051** | 0.9697 | 0.8654 | 94.24% | 80.00% | 0.0657 | 0.1274 |
| `F4F_05_BiDir_ResSwiGLU` | `Tabular_Stem` | `250,945` | 20 | **0.9035** | 0.9577 | 0.8787 | 95.96% | 81.04% | 0.0296 | 0.0734 |
| `F4F_06_Asinh_BiDirGCN` | `Tail_Robustness` | `125,761` | 21 | **0.8965** | 0.9651 | 0.8670 | 89.57% | 84.00% | 0.0597 | 0.1123 |
| `F4F_07_Asinh_110_BiDirGCN` | `Tail_Robustness` | `132,289` | 17 | **0.8940** | 0.9630 | 0.8469 | 91.71% | 78.67% | 0.0699 | 0.1189 |
| `F4F_10_Reg_DropEdge_BiDir` | `Regularization` | `125,761` | 19 | **0.8936** | 0.9582 | 0.8600 | 89.44% | 82.81% | 0.0546 | 0.1121 |
| `F4F_12_Composed_TopTopo_ModBCE_Asinh` | `Validation_Composed` | `250,945` | 19 | **0.8892** | 0.9415 | 0.8649 | 91.42% | 82.07% | 0.0284 | 0.0415 |
| `F4F_11_CausalCtx_BiDir` | `Temporal_Context` | `209,857` | 28 | **0.8853** | 0.9697 | 0.8328 | 86.33% | 80.44% | 0.0459 | 0.0783 |
| `F4F_02_BiDir_InOutGCN` | `Direction_Fix` | `125,761` | 35 | **0.8848** | 0.9689 | 0.8384 | 90.12% | 78.37% | 0.0462 | 0.0912 |
| `F4F_04_BiDir_GATv2_Res` | `Attention` | `167,489` | 14 | **0.8808** | 0.9581 | 0.8599 | 93.55% | 79.56% | 0.0751 | 0.1363 |
| `F4F_09_Focal_a050_g20_BiDir` | `Loss_Calibration` | `125,761` | 48 | **0.8781** | 0.9532 | 0.8386 | 89.98% | 78.52% | 0.0535 | 0.1704 |
| `F4F_08_ModBCE_pw276_BiDir` | `Loss_Calibration` | `125,761` | 14 | **0.8697** | 0.9308 | 0.8386 | 89.98% | 78.52% | 0.0271 | 0.0642 |
| `F4F_01_Undir_GCNAvg` | `Direction_Fix` | `43,267` | 39 | **0.8640** | 0.9649 | 0.8074 | 92.87% | 71.41% | 0.0568 | 0.1052 |
| `F4F_03_BiDir_SAGE_MeanMax` | `Aggregation` | `227,201` | 15 | **0.7876** | 0.9451 | 0.7248 | 83.56% | 64.00% | 0.0788 | 0.1353 |

---

## 2. Phase B Multi-Seed Shortlist Confirmation (Validation Steps `35–41`, Mean ± Std)

| Exp ID | Seeds | Params | Best Ep | Val PR-AUC | Val ROC-AUC | Val F1 | Brier | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F4F_00_Anchor_GCNAvg` | `[42, 123, 456]` | `43,267` | 43.3 ± 4.7 | **0.9040 ± 0.0008** | 0.9690 ± 0.0013 | 0.8633 ± 0.0031 | 0.0673 | 0.1272 |
| `F4F_06_Asinh_BiDirGCN` | `[42, 123, 456]` | `125,761` | 23.3 ± 2.6 | **0.9034 ± 0.0049** | 0.9667 ± 0.0016 | 0.8676 ± 0.0033 | 0.0661 | 0.1255 |
| `F4F_07_Asinh_110_BiDirGCN` | `[42, 123, 456]` | `132,289` | 25.3 ± 7.9 | **0.9032 ± 0.0066** | 0.9685 ± 0.0040 | 0.8601 ± 0.0112 | 0.0723 | 0.1262 |
| `F4F_05_BiDir_ResSwiGLU` | `[42, 123, 456]` | `250,945` | 22.7 ± 3.1 | **0.9008 ± 0.0022** | 0.9657 ± 0.0060 | 0.8654 ± 0.0095 | 0.0444 | 0.0918 |
