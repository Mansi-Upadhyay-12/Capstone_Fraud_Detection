# Stage 5C: Suspicious Subgraph & Potential Fraud-Ring Analysis Report

**Generated On:** 2026-11-26 21:10:44  
**Frozen Champion Evaluated:** `ExpA2_RankGauss182_BiDirGCN` (`seed=42`, threshold $\tau^* = 0.57`)  
**Attribution Standard:** Stage 5B PASS-verified $M=200$ 3-zone Composite Midpoint IG (`60@[0,0.005] + 60@[0.005,0.05] + 80@[0.05,1.0]`, zero baseline)

> **Safeguard Statement:** All subgraphs, scores, and candidacy tiers are determined exclusively from frozen model probabilities ($P \ge 0.57$) and snapshot directed edges—**never from validation or test labels**. Ground-truth labels (`Unknown = -1` kept strictly separate from `Licit = 0`) appear solely in post-hoc evaluation columns. Clusters are termed *suspicious transaction clusters* or *potential fraud-ring candidates*, never confirmed criminal rings.

---

## 1. Exact Scoring & Structural Tier Methodology

1. **Directed Graph Alignment:** Verified across raw [`txs_edgelist.txt`](file:///c:/Users/mansi/Downloads/Capstone_Fraud_Detection/dataset/txs_edgelist.txt) that **all `234,355` edges occur strictly within the same timestep (`0` cross-timestep edges)**; each snapshot $t \in \{1..49\}$ is therefore analyzed independently while preserving `src -> dst` edge orientation.
2. **Tier Separation (Avoiding Naive Component Inflation):**
   - **Ordinary Connected Components:** Weakly connected components (WCCs) of the entire snapshot graph (e.g., giant background components up to thousands of nodes).
   - **Isolated Suspicious Singletons:** Nodes with $P \ge 0.57$ having zero directly adjacent $P \ge 0.57$ neighbors.
   - **Direct Suspicious Clusters ($|C| \ge 2$):** Maximal WCCs of the induced predicted-suspicious subgraph $G[V_{\text{sus}}]$, partitioned into **Suspicious Pairs** ($|C|=2$), **Multi-Node Suspicious Clusters** ($|C|\ge 3$ not meeting candidate thresholds), and **Potential Fraud-Ring Candidates**.
   - **Potential Fraud-Ring Candidate Criteria (Unbiased, Pre-Specified, Label-Free):**
     1. $|C| \ge 3$ directly linked predicted-suspicious nodes ($P \ge 0.57$);
     2. Mean risk $\bar{P}(C) \ge 0.75$ and at least `2` high-confidence core nodes ($P \ge 0.85$);
     3. $\ge 2$ internal directed edges (`|E(C)| >= 2`).
   - **Composite Ring Score:** $\text{RingScore}(C) = \bar{P}(C) \cdot \log_2(1 + |C|) \cdot \bigl(1 + |E(C)|/|C|\bigr)$.
   - **Shared Unflagged Bridge Analysis:** Separately tracks unflagged nodes ($P < 0.57$) adjacent to $\ge 2$ predicted-suspicious nodes (`relay`, `shared_input`, `shared_output` motifs).

---

## 2. Structural Tier Summary Across Temporal Splits

| Split Bucket | Snapshots | Total Nodes | Pred Sus ($P\ge0.57$) | Isolated Singletons | Direct Sus Clusters ($\ge2$) | Potential Ring Candidates | Shared Bridges ($P<0.57$) | Inter-Cluster Bridges | Post-Hoc GT Direct Rings Hit ($\ge1$ / $\ge50\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_1_34` | `34` | `136,265` | `22,855` | `8,833` | `2,993` | `1,376` | `2,415` | `2,284` | `273` / `273` of `275` |
| `val_35_41` | `7` | `30,680` | `2,623` | `1,663` | `254` | `85` | `294` | `294` | `43` / `41` of `57` |
| `test_42` | `1` | `7,140` | `612` | `359` | `80` | `24` | `52` | `49` | `20` / `20` of `25` |
| `test_43_49` | `7` | `29,684` | `1,350` | `833` | `136` | `55` | `122` | `119` | `0` / `0` of `9` |

---

## 3. Top Candidate Fraud Rings: Validation (`Steps 35–41`) vs. Exploratory Test (`Steps 42` & `43–49`)

### 3.1 Top Validation Candidates (`Steps 35–41`)

| Cluster ID | Step | Tier | Motif | Size / Edges | Mean P | RingScore | Hub `txId` | Post-Hoc GT (`I / L / ?`) | Stage 5B Match |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `C_t40_001` | `40` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `32 / 31` | `0.9416` | `9.351` | `96599871` | `0 / 0 / 32` | `None` |
| `C_t35_002` | `35` | `Potential_Fraud_Ring_Candidate` | `complex_mesh` | `24 / 23` | `0.9761` | `8.877` | `69692260` | `23 / 0 / 1` | `None` |
| `C_t35_003` | `35` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `22 / 21` | `0.9751` | `8.622` | `56469542` | `0 / 0 / 22` | `None` |
| `C_t35_001` | `35` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `24 / 23` | `0.9414` | `8.561` | `63622261` | `0 / 0 / 24` | `None` |
| `C_t38_001` | `38` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `26 / 25` | `0.9052` | `8.443` | `49146952` | `0 / 0 / 26` | `None` |
| `C_t41_001` | `41` | `Potential_Fraud_Ring_Candidate` | `complex_mesh` | `20 / 20` | `0.9403` | `8.260` | `12779932` | `16 / 0 / 4` | `None` |
| `C_t40_002` | `40` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `18 / 17` | `0.9037` | `7.465` | `98591628` | `0 / 0 / 18` | `None` |
| `C_t35_004` | `35` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `17 / 16` | `0.9173` | `7.426` | `56469351` | `0 / 0 / 17` | `None` |
| `C_t35_005` | `35` | `Potential_Fraud_Ring_Candidate` | `complex_mesh` | `15 / 14` | `0.9594` | `7.419` | `70360850` | `11 / 1 / 3` | `VAL_FP_1;VAL_TP_1` |
| `C_t35_009` | `35` | `Potential_Fraud_Ring_Candidate` | `peeling_chain` | `12 / 11` | `0.9668` | `6.857` | `69729866` | `0 / 0 / 12` | `None` |

### 3.2 Exploratory Top Candidates: Test Step 42 & Post-Shutdown Steps `43–49`

| Cluster ID | Bucket | Step | Motif | Size / Edges | Mean P | RingScore | Hub `txId` | Post-Hoc GT (`I / L / ?`) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `C_t42_001` | `test_42` | `42` | `complex_mesh` | `38 / 37` | `0.9905` | `10.333` | `73282397` | `30 / 0 / 8` |
| `C_t42_003` | `test_42` | `42` | `complex_mesh` | `8 / 7` | `0.9965` | `5.923` | `72321311` | `7 / 0 / 1` |
| `C_t42_002` | `test_42` | `42` | `peeling_chain` | `9 / 8` | `0.9120` | `5.722` | `72704473` | `0 / 0 / 9` |
| `C_t42_004` | `test_42` | `42` | `peeling_chain` | `6 / 5` | `0.9056` | `4.661` | `43168337` | `0 / 0 / 6` |
| `C_t42_005` | `test_42` | `42` | `peeling_chain` | `5 / 4` | `0.9254` | `4.306` | `72397987` | `0 / 0 / 5` |
| `C_t45_001` | `test_43_49` | `45` | `peeling_chain` | `25 / 24` | `0.9722` | `8.957` | `164716518` | `0 / 0 / 25` |
| `C_t49_001` | `test_43_49` | `49` | `peeling_chain` | `16 / 15` | `0.9642` | `7.636` | `189167060` | `0 / 0 / 16` |
| `C_t49_002` | `test_43_49` | `49` | `peeling_chain` | `15 / 14` | `0.9804` | `7.582` | `193989788` | `0 / 0 / 15` |
| `C_t49_003` | `test_43_49` | `49` | `fan_in_consolidation` | `13 / 12` | `0.9713` | `7.112` | `179810303` | `0 / 0 / 13` |
| `C_t43_001` | `test_43_49` | `43` | `peeling_chain` | `17 / 16` | `0.8592` | `6.955` | `94656838` | `0 / 0 / 17` |

---

## 4. Hub Feature Attribution (`M=200` Composite Midpoint IG) & Key Insights

- **`C_t40_001` Hub (`txId=96599871`, Step `40`, `P=0.9941`):** Self IG=`+2.117` logits; completeness err=`0.00981` (`0.104%`, `finite=True`, `repro=0.00e+00`). Top +drivers: `Local_feature_76` (`+5.983`), `Local_feature_88` (`+4.977`), `Local_feature_21` (`+4.223`); Top -drivers: `Aggregate_feature_8` (`-6.748`), `Aggregate_feature_7` (`-6.521`).
- **`C_t35_002` Hub (`txId=69692260`, Step `35`, `P=0.5912`):** Self IG=`+3.992` logits; completeness err=`0.01726` (`0.368%`, `finite=True`, `repro=0.00e+00`). Top +drivers: `Local_feature_90` (`+7.676`), `Local_feature_14` (`+6.999`), `Local_feature_89` (`+6.991`); Top -drivers: `Aggregate_feature_10` (`-9.867`), `Aggregate_feature_7` (`-9.633`).
- **`C_t42_001` Hub (`txId=73282397`, Step `42`, `P=0.9947`):** Self IG=`+9.170` logits; completeness err=`0.00864` (`0.090%`, `finite=True`, `repro=0.00e+00`). Top +drivers: `Local_feature_3` (`+5.235`), `Local_feature_14` (`+4.770`), `Local_feature_90` (`+4.736`); Top -drivers: `Aggregate_feature_10` (`-6.709`), `Aggregate_feature_8` (`-6.252`).
- **`C_t45_001` Hub (`txId=164716518`, Step `45`, `P=0.9963`):** Self IG=`+5.220` logits; completeness err=`0.01709` (`0.172%`, `finite=True`, `repro=0.00e+00`). Top +drivers: `Local_feature_88` (`+5.084`), `Local_feature_76` (`+4.670`), `Local_feature_21` (`+3.643`); Top -drivers: `Aggregate_feature_7` (`-6.577`), `Aggregate_feature_8` (`-6.037`).

---

## 5. Methodological Limitations

1. **Unlabeled (`Unknown = -1`) Majority in Bitcoin Transaction Graphs:** A large fraction of nodes inside candidate subgraphs carry `Unknown (-1)` ground-truth status (`post_hoc_unknown_share_all`), so labeled-only precision metrics evaluate only the subset with published labels and cannot confirm whether unlabelled nodes inside dense suspicious clusters are true darknet actors or benign exchange users.
2. **Post-Shutdown Regime Shift (`Steps 43–49`):** Following Step 42's darknet market closure (diagnosed in Stage 4G), flagged node counts drop sharply and true coordinated illicit clusters after Step 42 largely escape threshold `0.57`, shifting detected `43–49` clusters toward low post-hoc labeled precision / higher false-alarm share.
3. **Cross-Snapshot Wallet Tracking Unavailable:** Elliptic++ edges exist strictly within individual snapshots; cross-timestep laundering across multi-week horizons cannot be reconstructed solely from snapshot edges.
4. **Correlation vs. Legal Attribution:** Structural cohesion and high GNN scores establish model-predicted risk concentration, **not** legal proof of a criminal fraud ring.
