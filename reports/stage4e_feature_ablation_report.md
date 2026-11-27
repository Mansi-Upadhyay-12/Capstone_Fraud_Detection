# Stage 4E: Controlled Feature-Group Ablation Report (`GCNAvg`)

**Generated On:** 2026-11-26 12:59:18  
**Architecture Held Fixed:** `SeparatedGraphNet(hidden=128, out=64, dropout=0.3, graph_mode="gcn_avg")`, `gcn_directed`  
**Training Protocol:** Train `Steps 1–34` (`Adam lr=0.002, wd=1e-3, grad_clip=2.0`, weighted BCE `pos_weight=7.6349`) $\to$ Validate/Select `Steps 35–41` $\to$ Post-Selection Test `Steps 42–49`  
**Seeds:** `[42, 123, 456]` | **Max Epochs / Patience:** `50` / `15`  
**Validation-Selected Winner:** `F1_Local93` (Best Seed `42`, Mean Val PR-AUC = `0.9040 ± 0.0008`)

---

## 1. Validation Performance Across Seeds (`Steps 35–41`, Mean ± Std)

| Config | Features | Params | Best Ep (Mean ± Std) | Val PR-AUC | Val ROC-AUC | Val F1 (`0.01–0.99`) | Val Prec | Val Rec | Val Brier | Val ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F1_Local93` | `93` | `43,267` | 43.3 ± 4.7 | **0.9040 ± 0.0008** | 0.9690 ± 0.0013 | 0.8633 ± 0.0031 | 94.38% ± 0.24% | 79.55% ± 0.44% | 0.0673 ± 0.0071 | 0.1272 ± 0.0140 |
| `F2_Local93_Domain17` | `110` | `47,619` | 40.0 ± 10.0 | **0.9007 ± 0.0050** | 0.9660 ± 0.0028 | 0.8577 ± 0.0071 | 91.60% ± 1.69% | 80.69% ± 1.88% | 0.0797 ± 0.0106 | 0.1456 ± 0.0160 |
| `F3_Full182` | `182` | `66,051` | 13.3 ± 13.2 | **0.6909 ± 0.0518** | 0.9146 ± 0.0158 | 0.6406 ± 0.0563 | 72.78% ± 3.02% | 57.83% ± 8.83% | 0.1702 ± 0.0524 | 0.2961 ± 0.1279 |

---

## 2. Post-Selection Test Performance Across Seeds (`Steps 42–49`, `8,841` Nodes, Mean ± Std)

| Config | Features | Test PR-AUC | Test ROC-AUC | Test F1 (`0.01–0.99`) | Test Prec | Test Rec | Test Brier | Test ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F1_Local93` | `93` | **0.5208 ± 0.0043** | 0.8180 ± 0.0039 | **0.5932 ± 0.0049** | 86.03% ± 0.87% | 45.26% ± 0.50% | 0.0776 ± 0.0052 | 0.1182 ± 0.0101 |
| `F2_Local93_Domain17` | `110` | **0.5099 ± 0.0018** | 0.8036 ± 0.0104 | **0.5736 ± 0.0167** | 78.12% ± 6.12% | 45.42% ± 0.64% | 0.0899 ± 0.0083 | 0.1363 ± 0.0115 |
| `F3_Full182` | `182` | **0.4274 ± 0.0504** | 0.8148 ± 0.0131 | **0.4281 ± 0.0359** | 49.99% ± 14.78% | 39.70% ± 4.39% | 0.1874 ± 0.0715 | 0.3335 ± 0.1476 |

---

## 3. Per-Seed Runs (Parameters, Best Epochs, Validation & Post-Selection Test)

| Config | Seed | Params | Best Ep | Val PR-AUC | Val $\tau^*$ | Test PR-AUC | Test ROC-AUC | Test F1 | Test Prec | Test Rec | Test Brier | Test ECE | TP / FP / TN / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `F1_Local93` | 42 | `43,267` | 40 | 0.9051 | `0.93` | 0.5268 | 0.8188 | 0.5965 | 85.39% | 45.83% | 0.0740 | 0.1145 | 187/32/8401/221 |
| `F1_Local93` | 123 | `43,267` | 50 | 0.9034 | `0.94` | 0.5170 | 0.8223 | 0.5862 | 85.45% | 44.61% | 0.0738 | 0.1081 | 182/31/8402/226 |
| `F1_Local93` | 456 | `43,267` | 40 | 0.9035 | `0.95` | 0.5186 | 0.8128 | 0.5968 | 87.26% | 45.34% | 0.0850 | 0.1319 | 185/27/8406/223 |
| `F2_Local93_Domain17` | 42 | `47,619` | 45 | 0.8979 | `0.93` | 0.5080 | 0.7889 | 0.5926 | 86.38% | 45.10% | 0.0968 | 0.1460 | 184/29/8404/224 |
| `F2_Local93_Domain17` | 123 | `47,619` | 49 | 0.9077 | `0.93` | 0.5123 | 0.8113 | 0.5762 | 76.21% | 46.32% | 0.0782 | 0.1202 | 189/59/8374/219 |
| `F2_Local93_Domain17` | 456 | `47,619` | 26 | 0.8966 | `0.94` | 0.5093 | 0.8105 | 0.5520 | 71.76% | 44.85% | 0.0946 | 0.1428 | 183/72/8361/225 |
| `F3_Full182` | 42 | `66,051` | 4 | 0.6967 | `0.69` | 0.3581 | 0.8234 | 0.3888 | 40.37% | 37.50% | 0.2441 | 0.4434 | 153/226/8207/255 |
| `F3_Full182` | 123 | `66,051` | 4 | 0.7512 | `0.64` | 0.4766 | 0.7963 | 0.4198 | 38.72% | 45.83% | 0.2315 | 0.4322 | 187/296/8137/221 |
| `F3_Full182` | 456 | `66,051` | 32 | 0.6247 | `0.91` | 0.4475 | 0.8247 | 0.4756 | 70.87% | 35.78% | 0.0866 | 0.1249 | 146/60/8373/262 |

---

## 4. Isolated Feature-Group Attribution Deltas (Mean Across Seeds)

| Comparison | Isolated Factor | $\Delta$ Val PR-AUC | $\Delta$ Test PR-AUC | $\Delta$ Test ROC-AUC | $\Delta$ Test F1 | $\Delta$ Test Brier | $\Delta$ Test ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `F2_Local93_Domain17 - F1_Local93` | Adding Domain-17 (indices 165..181) to Local-93 | `-0.0033` | `-0.0109` | `-0.0144` | `-0.0196` | `+0.0123` | `+0.0181` |
| `F3_Full182 - F2_Local93_Domain17` | Adding Aggregate-72 (indices 93..164) to Local-93 + Domain-17 | `-0.2098` | `-0.0825` | `+0.0112` | `-0.1455` | `+0.0975` | `+0.1972` |
| `F3_Full182 - F1_Local93` | Adding both Domain-17 and Aggregate-72 to Local-93 | `-0.2131` | `-0.0934` | `-0.0032` | `-0.1651` | `+0.1098` | `+0.2153` |
