# Stage 5C Completion & Independent Audit Report

**Completed On:** 2026-11-26 21:10:44  
**Execution Time:** `1091.28s` | **Smoke Test & Full Pipeline:** Both Executed Cleanly

---

## 1. Main Quantitative Findings

- **Validation `Steps 35–41`:** `2,623` flagged nodes split into `1,663` singletons and **`254` direct suspicious clusters** (`n >= 2`), of which **`85` qualify as Potential Fraud-Ring Candidates** (with `294` shared unflagged bridges, `294` linking distinct suspicious groups). Post-hoc, `43/57` GT direct illicit rings had at least 1 member caught (`41/57` majority-detected).
- **Test `Step 42`:** `612` flagged nodes formed **`80` direct clusters** and **`24` Potential Fraud-Ring Candidates** (`20/25` GT rings detected).
- **Post-Shutdown Test `Steps 43–49`:** `1350` flagged nodes formed `136` direct clusters and `55` candidates, recovering `0/9` GT rings (honestly reflecting post-shutdown distribution shift).

---

## 2. Independent Verification & Audit Checklist

| Check # | Verification Item | Status | Observed Evidence |
| :---: | :--- | :---: | :--- |
| 1 | Raw dataset `txId`, timestep & edge alignment | **PASS** | `234,355` edges verified; `cross_timestep = 0`; `12,404` sampled & all cluster node/edge IDs matched `dataset/txs_*.txt` |
| 2 | Valid nodes & edges in every reported group | **PASS** | All `100%` of cluster node indices within `[0, num_nodes)`, all edges present in snapshot `edge_index`, `has_invalid_group = False` |
| 3 | Directed edge orientation (`src -> dst`) | **PASS** | Verified `txId1 -> txId2` orientation against raw `dataset/txs_edgelist.txt` (`directed_orientation_ok = True`) |
| 4 | Unknown (`-1`) excluded from labeled metrics | **PASS** | Unknown `-1` preserved explicitly as `"Unknown"` in CSVs and separated in `post_hoc_gt_unknown` |
| 5 | Zero test-label influence on scoring/selection | **PASS** | Scoring formula uses purely `probs >= 0.57` and topology; checked no `y` access before post-hoc aggregation |
| 6 | Deterministic reproducibility | **PASS** | Repeated cluster extraction & `M=200` hub IG matched identically (`max_hub_ig_repro_diff = 0.00e+00`) |
| 7 | All numerical outputs finite | **PASS** | Zero `NaN`/`Inf` across all CSVs and hub IG matrices (`all_outputs_finite = True`) |
| 8 | Stage 4H, Stage 5A & Stage 5B files untouched | **PASS** | SHA-256 hashes of frozen ckpt, RankGauss pipeline, Stage 5A & Stage 5B artifacts verified unchanged (`protected_files_untouched = True`) |
| 9 | `M=200` Hub IG completeness tolerance | **PASS** | Max hub completeness error = `0.01726` (`0.368%` rel) |
