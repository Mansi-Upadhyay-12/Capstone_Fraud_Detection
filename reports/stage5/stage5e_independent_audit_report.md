# Stage 5E: Independent End-to-End Verification and Audit Report

**Audit Timestamp:** 2026-11-26 22:01:47  
**Execution Mode:** `FULL_AUDIT` (`elapsed = 540.33s`)  
**Verdict Summary:** `11` PASS | `3` WARNING | `0` FAIL (`total = 14` checks across `68` protected Stage 1–5D artifacts)

> **Audit Principle:** Zero Stage 1–4H or Stage 5A–5D artifacts were edited, retrained, or overwritten. Every numerical and structural check below states explicitly whether it was **Independently Reproduced** from raw data/checkpoints or **Inspection Verified**, and every minor discrepancy or residual limitation is reported honestly.

---

## 1. Complete Audit Scorecard

| ID | Stage | Verification Category | Method | Status | Key Observed Evidence / Discrepancy |
| :---: | :--- | :--- | :---: | :---: | :--- |
| `5A-1` | Stage 5A | CSV dimensions, 182 feature ordering, group alignment & finiteness | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5A-2` | Stage 5A | HGB Val/Test PR-AUC & Validation permutation importance reproduction | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5A-3` | Stage 5A | GNN M=80 IG Step 35 reproduction, M=80 identification & metric-distinction check | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5B-1` | Stage 5B | Frozen threshold 0.57 & full Validation/Test confusion matrices | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5B-2` | Stage 5B | 12 target txIds, 56 directed neighbor edges & Unknown (-1) preservation vs raw dataset | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5B-3` | Stage 5B | M=80 artifact preservation & independent M=200 Composite Midpoint IG reproduction | Independently Reproduced | **WARNING** | Minor documentation omission (non-blocking): stage5b_human_explanations.md does not explicitly print 'M=80' in its header, though stage5b_ig_completeness_review.md and stage5b_ig_final_validation.md cleanly distinguish M=80 vs M=200. |
| `5C-1` | Stage 5C | All snapshots clusters, RingScore criteria & 0/9 post-shutdown GT rings reproduction | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5C-2` | Stage 5C | Raw txId/timestep/directed-edge audit, Unknown(-1) handling & numeric finiteness | Independently Reproduced | **WARNING** | Minor CSV serialization quirk (non-blocking): string placeholder 'None' in ['stage5b_explained_cases'] is parsed as NaN by default pandas.read_csv() unless keep_default_na=False is passed; all numeric columns are 100% finite. |
| `5C-3` | Stage 5C | Candidate hub M=200 IG reproduction & zero-test-label-tuning verification | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5D-1` | Stage 5D | All 8 PNG (300 DPI) + 8 vector PDF integrity & manifest byte parity | Independently Reproduced | **PASS** | Exact match / verified clean |
| `5D-2` | Stage 5D | Figure 3 M=80 provenance, Figure 8 98.9% Unknown exact denominator & exploratory labeling | Independently Reproduced | **WARNING** | Minor rounding note (non-blocking) on Figure 8 Panel B: exact shares across all 351 member nodes of the 55 post-shutdown (Steps 43-49) Potential Fraud-Ring Candidates are Unknown=347/351 (98.860% -> 98.9%), Illicit=2/351 (0.570% -> 0.6%), Licit=2/351 (0.570% -> 0.6%), so the three 1-decimal annotations sum to 100.1% due to independent rounding.; Minor precision note: stage5d_visualization_report.md describes the 98.9% Unknown figure in Steps 43-49 candidates without printing the exact denominator (347/351 nodes across 55 candidates). |
| `XS-1` | Cross-Stage | Frozen champion, threshold 0.57 & Stage 5B<->5C txId/timestep alignment | Independently Reproduced | **PASS** | Exact match / verified clean |
| `DOC-1` | Documentation | Cross-report limitations, intra-snapshot scope & non-criminal-proof claims | Inspection Verified | **PASS** | Exact match / verified clean |
| `PROT-1` | Protection | SHA-256 immutability across all 68 Stage 1-5D protected files | Independently Reproduced | **PASS** | Exact match / verified clean |

---

## 2. Exact Reproduced Highlights vs. Previously Reported Results

1. **Frozen Model & Threshold (`Stage 4H -> 5D`):** Independently loaded `models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt` (`threshold_recal_01_99 = 0.57`) and Train `1–34` `rankgauss_transformer_train_1_34.joblib`. Reproduced HGB Val/Test PR-AUC (`0.9292` / `0.5646`), `ExpA2` Val/Test confusion matrices (`TP=573, FP=33, FN=102, TN=7,121` and `TP=192, FP=43, FN=216, TN=8,390`, with `22,851` / `27,983` Unknown `-1` excluded).
2. **Stage 5A (`M=80` Identified Accurately):** Feature index/group ordering (`93 + 72 + 17 = 182`) exact; permutation PR-AUC drop and `Val Step 35` IG (`M=80 poly4`) reproduced cleanly; Stage 5A artifacts remain untouched at `M=80` (max completeness error `0.0628`), distinct from the `M=200` standard validated later.
3. **Stage 5B (`M=80` Existing vs. `M=200` Validated):** All `12` targets and `56` directed edges verified against raw `dataset/txs_*.txt`; independent `M=200` Composite Midpoint IG reproduction on `3` representative cases matched `stage5b_ig_final_validation.md` (`max_abs_err <= 0.0393`, `repro_diff = 0.00e+00`).
4. **Stage 5C (`0/9` Post-Shutdown Ground-Truth Rings Verified):** Across `Steps 43–49`, independent inspection of `G[y == 1]` found **exactly `9` connected GT illicit rings** (`size >= 2`); the **highest `ExpA2` probability of any member node across all 9 post-shutdown GT rings is `0.1077` (< `0.57`)**, explaining why `0/9` rings were detected at $\tau^* = 0.57$.
5. **Stage 5D Figures (`300 DPI` PNG + Vector PDF) & `98.9%` Unknown Denominator:** All `8` PNGs confirmed `300 DPI` and all `8` PDFs confirmed vector (`/Font`, `/Page`, `0` `/Subtype /Image` XObjects). In Figure 8 Panel B, the exact `Steps 43–49` candidate node denominator across `55` candidates is **`347 / 351` Unknown (`98.8604%`)**, `2 / 351` Illicit (`0.5698%`), `2 / 351` Licit (`0.5698%`).

---

## 3. Discrepancies Found & Residual Limitations

### Discrepancies (All Minor / Non-Blocking — Zero Failed Correctness Checks)
- **`5B-3` (WARNING):** Minor documentation omission (non-blocking): stage5b_human_explanations.md does not explicitly print 'M=80' in its header, though stage5b_ig_completeness_review.md and stage5b_ig_final_validation.md cleanly distinguish M=80 vs M=200.
- **`5C-2` (WARNING):** Minor CSV serialization quirk (non-blocking): string placeholder 'None' in ['stage5b_explained_cases'] is parsed as NaN by default pandas.read_csv() unless keep_default_na=False is passed; all numeric columns are 100% finite.
- **`5D-2` (WARNING):** Minor rounding note (non-blocking) on Figure 8 Panel B: exact shares across all 351 member nodes of the 55 post-shutdown (Steps 43-49) Potential Fraud-Ring Candidates are Unknown=347/351 (98.860% -> 98.9%), Illicit=2/351 (0.570% -> 0.6%), Licit=2/351 (0.570% -> 0.6%), so the three 1-decimal annotations sum to 100.1% due to independent rounding.
- **`5D-2` (WARNING):** Minor precision note: stage5d_visualization_report.md describes the 98.9% Unknown figure in Steps 43-49 candidates without printing the exact denominator (347/351 nodes across 55 candidates).

### Residual Limitations
1. **Unlabeled (`Unknown = -1`) Majority (`77.15%` globally; `98.89%` in `43–49` candidates):** Limits ground-truth ring precision auditing on newly flagged post-shutdown clusters.
2. **Post-Shutdown Regime Shift (`Steps 43–49`):** `0/9` published GT rings detected at frozen $\tau^* = 0.57$; distribution shift requires domain-adaptive or continual retraining beyond frozen Stage 4H scope.
3. **Attribution Method Split Across Sub-Stages:** Existing Stage 5A (`M=80`) and Stage 5B CSV/JSON (`M=80`) retain their original `poly4` outputs (preserving immutability), whereas Stage 5B Final Validation (`stage5b_ig_final_validation.md`) and Stage 5C Ring Hubs use the tighter `M=200` Composite Midpoint grid (`< 0.04` abs / `< 1%` rel). **Optional Future Recommendation:** If an exhibition/final archival bundle requires unified `M=200` tables everywhere, run a dedicated post-5E Stage 5F archival refresh without altering frozen checkpoints.

---

## 4. Protected Artifacts Verified via SHA-256

All `68` files across `models/stage4h/`, `models/hist_gradient_boosting.joblib`, `processed/`, `reports/stage3–4h`, `src/stage5/`, `results/stage5/`, `reports/stage5/` (5A–5D), and `figures/stage5/` passed identical pre/post SHA-256 verification (`modified_files = []`).
