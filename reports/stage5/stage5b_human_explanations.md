# Stage 5B: Human-Readable Individual Fraud Prediction Explanations

**Generated On:** 2026-11-26 19:52:43  
**Model:** `ExpA2_RankGauss182_BiDirGCN` (Checkpoint: `models/stage4h/ExpA2_RankGauss182_BiDirGCN_seed42.pt`)  
**Preprocessing:** `models/stage4h/rankgauss_transformer_train_1_34.joblib` (fitted on Train Steps `1–34`)  
**Frozen Decision Threshold:** `0.57` (`threshold_recal_01_99`)

> **Important Note on Evidence vs. Truth:** Integrated Gradients (IG) attributions and neighbor predictions reflect how the frozen GNN model arrived at its probability score relative to a zero RankGauss reference. **Model explanations do NOT prove that a transaction is factually fraudulent or licit**, especially for False Positives, False Negatives, and unlabeled (`Unknown = -1`) neighboring transactions.

---

## 1. Selection Protocol

- **Primary Explanation Dataset (`Validation Steps 35–41`, `8` cases):** Deterministic selection preferring connected transactions (`in_degree + out_degree >= 2`, and both incoming & outgoing neighbors where available), picking `2` representative examples each from **TP**, **FP**, **FN**, and **TN** (`VAL_TP_1`, `VAL_TP_2`, `VAL_FP_1`, `VAL_FP_2`, `VAL_FN_1`, `VAL_FN_2`, `VAL_TN_1`, `VAL_TN_2`).
- **Exploratory Post-Hoc Test Dataset (`Test Steps 42–49`, `4` cases):** Symmetric exploratory selection (`TEST_TP_42`, `TEST_FP_1`, `TEST_FN_POST43`, `TEST_TN_1`) inspecting Step 42 vs. post-shutdown (`Steps 43–49`) model behavior without any test tuning.

---

## 2. Individual Transaction Explanations

### VAL_TP_1 (Validation_Primary) — `TP`
*Selection Rationale:* Highest-probability connected Validation TP

Transaction Node: `70361269` (Snapshot node index `4912`)

Timestep: `35`

Ground-Truth Dataset Label: `Illicit` (`y=1`)

Predicted Fraud Probability: `0.9984` (logit `+6.4339` vs. baseline `-4.3255`, $\Delta = +10.7594$)

Prediction: `Illicit` (using frozen threshold `0.57`)

Classification: `TP`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_70` (idx `162`, `Aggregate72`): `+3.52139` (self: `+2.14067`, nbrs: `+1.38072`; raw=`-0.1652`, RankGauss=`-2.089`)
  - `Local_feature_14` (idx `13`, `Local93`): `+2.18345` (self: `+1.61295`, nbrs: `+0.57051`; raw=`-0.0563`, RankGauss=`-2.799`)
  - `Aggregate_feature_39` (idx `131`, `Aggregate72`): `+2.12679` (self: `+1.41812`, nbrs: `+0.70867`; raw=`-0.1702`, RankGauss=`-2.504`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-3.83818` (self: `-1.81227`, nbrs: `-2.02591`; raw=`1.9870`, RankGauss=`+5.201`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-3.40611` (self: `-1.63974`, nbrs: `-1.76637`; raw=`1.9316`, RankGauss=`+5.202`)
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-3.19717` (self: `-1.52581`, nbrs: `-1.67137`; raw=`1.9914`, RankGauss=`+5.201`)

Graph Context (Strictly within Timestep `35`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `1/1`, mean P=`0.898`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `1/1`, mean P=`0.998`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `9`
- IG Attribution Origin Split (Logits): Self=`+1.9797`, 1-Hop In=`+3.9928`, 1-Hop Out=`+5.1608`, 2-Hop Only=`-0.2571`
- Relevant neighbouring predictions: Strongly concentrated suspicious neighborhood: 2/2 (100.0%) 1-hop neighbors predicted Illicit

Explanation:
The frozen model assigns P(Illicit) = 0.9984 (>= tau*=0.57), matching the dataset Illicit label (TP). Positive fraud attribution is led by Aggregate_feature_70 (+3.521) and Local_feature_14 (+2.183) (total positive IG = +39.664), outweighing mitigating negative evidence from Aggregate_feature_10 (-3.838) and Aggregate_feature_7 (-3.406) (-28.787). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (2/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = +8.897 logits, versus self-node IG = +1.980 logits).

---

### VAL_TP_2 (Validation_Primary) — `TP`
*Selection Rationale:* Borderline-confidence connected Validation TP (just above tau*)

Transaction Node: `40467500` (Snapshot node index `340`)

Timestep: `38`

Ground-Truth Dataset Label: `Illicit` (`y=1`)

Predicted Fraud Probability: `0.5868` (logit `+0.3506` vs. baseline `-4.3255`, $\Delta = +4.6761$)

Prediction: `Illicit` (using frozen threshold `0.57`)

Classification: `TP`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Local_feature_80` (idx `79`, `Local93`): `+7.39433` (self: `+5.34412`, nbrs: `+2.05021`; raw=`-1.4678`, RankGauss=`-1.467`)
  - `Local_feature_3` (idx `2`, `Local93`): `+5.76044` (self: `+2.78110`, nbrs: `+2.97934`; raw=`1.4190`, RankGauss=`+1.473`)
  - `Local_feature_81` (idx `80`, `Local93`): `+4.67380` (self: `+3.60906`, nbrs: `+1.06474`; raw=`-1.4667`, RankGauss=`-1.462`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-13.35774` (self: `-7.63855`, nbrs: `-5.71919`; raw=`2.2891`, RankGauss=`+5.201`)
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-11.60518` (self: `-5.95460`, nbrs: `-5.65057`; raw=`2.2829`, RankGauss=`+5.201`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-9.92134` (self: `-5.17329`, nbrs: `-4.74805`; raw=`2.2172`, RankGauss=`+5.202`)

Graph Context (Strictly within Timestep `38`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `0/1`, mean P=`0.559`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `1/1`, mean P=`0.874`; GT labels: `1` Illicit, `0` Licit, `0` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `1`
- IG Attribution Origin Split (Logits): Self=`+5.1716`, 1-Hop In=`-0.1629`, 1-Hop Out=`-0.8944`, 2-Hop Only=`+0.5068`
- Relevant neighbouring predictions: Strongly concentrated suspicious neighborhood: 1/2 (50.0%) 1-hop neighbors predicted Illicit

Explanation:
The frozen model assigns P(Illicit) = 0.5868 (>= tau*=0.57), matching the dataset Illicit label (TP). Positive fraud attribution is led by Local_feature_80 (+7.394) and Local_feature_3 (+5.760) (total positive IG = +101.435), outweighing mitigating negative evidence from Aggregate_feature_8 (-13.358) and Aggregate_feature_10 (-11.605) (-96.814). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (1/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -0.550 logits, versus self-node IG = +5.172 logits).

---

### VAL_FP_1 (Validation_Primary) — `FP`
*Selection Rationale:* Highest-probability connected Validation FP false alarm

Transaction Node: `54915900` (Snapshot node index `1572`)

Timestep: `35`

Ground-Truth Dataset Label: `Licit` (`y=0`)

Predicted Fraud Probability: `0.9216` (logit `+2.4642` vs. baseline `-4.3255`, $\Delta = +6.7897$)

Prediction: `Illicit` (using frozen threshold `0.57`)

Classification: `FP`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_70` (idx `162`, `Aggregate72`): `+7.83545` (self: `+3.67516`, nbrs: `+4.16029`; raw=`-0.1112`, RankGauss=`-0.939`)
  - `Aggregate_feature_39` (idx `131`, `Aggregate72`): `+6.42008` (self: `+0.79970`, nbrs: `+5.62038`; raw=`-0.1702`, RankGauss=`-0.400`)
  - `Local_feature_3` (idx `2`, `Local93`): `+5.14498` (self: `+3.05270`, nbrs: `+2.09228`; raw=`1.4190`, RankGauss=`+1.473`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-12.53305` (self: `-8.51442`, nbrs: `-4.01864`; raw=`1.9869`, RankGauss=`+5.201`)
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-11.80430` (self: `-7.57376`, nbrs: `-4.23054`; raw=`1.9914`, RankGauss=`+5.201`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-10.49622` (self: `-6.63076`, nbrs: `-3.86545`; raw=`1.9315`, RankGauss=`+5.202`)

Graph Context (Strictly within Timestep `35`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `0/1`, mean P=`0.004`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `1/1`, mean P=`0.998`; GT labels: `1` Illicit, `0` Licit, `0` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `1`
- IG Attribution Origin Split (Logits): Self=`-3.2882`, 1-Hop In=`+0.7774`, 1-Hop Out=`+6.9850`, 2-Hop Only=`+2.2275`
- Relevant neighbouring predictions: Strongly concentrated suspicious neighborhood: 1/2 (50.0%) 1-hop neighbors predicted Illicit

Explanation:
Although the dataset records this transaction as Licit, the frozen model assigns an elevated P(Illicit) = 0.9216 (>= tau*=0.57), creating a False Positive alarm. The model's score is driven upward primarily by Aggregate_feature_70 (+7.835) and Aggregate_feature_39 (+6.420) (total positive IG = +115.233) which resemble illicit patterns under RankGauss scaling, overriding negative Licit cues from Aggregate_feature_10 (-12.533) and Aggregate_feature_8 (-11.804). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (1/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = +9.990 logits, versus self-node IG = -3.288 logits). This attribution explains why the classifier reacted as it did, not that the transaction is fraudulent.

---

### VAL_FP_2 (Validation_Primary) — `FP`
*Selection Rationale:* Borderline-confidence connected Validation FP (just above tau*)

Transaction Node: `97404552` (Snapshot node index `3167`)

Timestep: `40`

Ground-Truth Dataset Label: `Licit` (`y=0`)

Predicted Fraud Probability: `0.6043` (logit `+0.4234` vs. baseline `-4.3255`, $\Delta = +4.7489$)

Prediction: `Illicit` (using frozen threshold `0.57`)

Classification: `FP`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_49` (idx `141`, `Aggregate72`): `+6.49111` (self: `+2.13528`, nbrs: `+4.35584`; raw=`0.0392`, RankGauss=`+1.197`)
  - `Aggregate_feature_15` (idx `107`, `Aggregate72`): `+6.47802` (self: `+2.63746`, nbrs: `+3.84056`; raw=`0.3391`, RankGauss=`+2.255`)
  - `Aggregate_feature_21` (idx `113`, `Aggregate72`): `+6.39788` (self: `+2.44550`, nbrs: `+3.95239`; raw=`0.8515`, RankGauss=`+2.375`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-10.61738` (self: `-4.79431`, nbrs: `-5.82308`; raw=`2.4879`, RankGauss=`+5.201`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-10.51817` (self: `-3.99840`, nbrs: `-6.51977`; raw=`2.4080`, RankGauss=`+5.202`)
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-9.35404` (self: `-3.72123`, nbrs: `-5.63282`; raw=`2.4806`, RankGauss=`+5.201`)

Graph Context (Strictly within Timestep `40`, original direction preserved):
- Incoming neighbours (`w -> target`): `2` (predicted Illicit: `0/2`, mean P=`0.091`; GT labels: `0` Illicit, `2` Licit, `0` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `2` (predicted Illicit: `0/2`, mean P=`0.100`; GT labels: `0` Illicit, `1` Licit, `1` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `4` / `6`
- IG Attribution Origin Split (Logits): Self=`+0.1420`, 1-Hop In=`-4.4177`, 1-Hop Out=`+9.6559`, 2-Hop Only=`-0.6307`
- Relevant neighbouring predictions: Clean 1-hop neighborhood (0/4 predicted Illicit), but 1/6 2-hop neighbors flagged Illicit

Explanation:
Although the dataset records this transaction as Licit, the frozen model assigns an elevated P(Illicit) = 0.6043 (>= tau*=0.57), creating a False Positive alarm. The model's score is driven upward primarily by Aggregate_feature_49 (+6.491) and Aggregate_feature_15 (+6.478) (total positive IG = +134.964) which resemble illicit patterns under RankGauss scaling, overriding negative Licit cues from Aggregate_feature_8 (-10.617) and Aggregate_feature_7 (-10.518). Its direct snapshot neighborhood has 2 incoming and 2 outgoing edges (0/4 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = +4.607 logits, versus self-node IG = +0.142 logits). This attribution explains why the classifier reacted as it did, not that the transaction is fraudulent.

---

### VAL_FN_1 (Validation_Primary) — `FN`
*Selection Rationale:* Lowest-probability connected Validation FN missed fraud

Transaction Node: `97729143` (Snapshot node index `3496`)

Timestep: `40`

Ground-Truth Dataset Label: `Illicit` (`y=1`)

Predicted Fraud Probability: `0.0029` (logit `-5.8315` vs. baseline `-4.3255`, $\Delta = -1.5060$)

Prediction: `Licit` (using frozen threshold `0.57`)

Classification: `FN`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_20` (idx `112`, `Aggregate72`): `+2.28706` (self: `+1.62083`, nbrs: `+0.66622`; raw=`2.2142`, RankGauss=`+2.678`)
  - `Aggregate_feature_22` (idx `114`, `Aggregate72`): `+1.76482` (self: `+1.21317`, nbrs: `+0.55164`; raw=`2.5274`, RankGauss=`+3.021`)
  - `Local_feature_3` (idx `2`, `Local93`): `+1.57926` (self: `+0.75314`, nbrs: `+0.82611`; raw=`1.4190`, RankGauss=`+1.473`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-2.20774` (self: `-1.62242`, nbrs: `-0.58532`; raw=`2.4077`, RankGauss=`+5.202`)
  - `Local_feature_21` (idx `20`, `Local93`): `-1.68507` (self: `-0.97625`, nbrs: `-0.70882`; raw=`-1.3963`, RankGauss=`-1.393`)
  - `Aggregate_feature_66` (idx `158`, `Aggregate72`): `-1.44498` (self: `-0.81750`, nbrs: `-0.62747`; raw=`1.5205`, RankGauss=`+1.503`)

Graph Context (Strictly within Timestep `40`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `0/1`, mean P=`0.003`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `0/1`, mean P=`0.006`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `20`
- IG Attribution Origin Split (Logits): Self=`-1.4693`, 1-Hop In=`-0.1591`, 1-Hop Out=`-0.0346`, 2-Hop Only=`+0.1162`
- Relevant neighbouring predictions: Predominantly benign predicted neighborhood: 0/2 1-hop neighbors predicted Illicit (mean P=0.004, max P=0.006)

Explanation:
Although the dataset records this transaction as Illicit, the frozen model assigns P(Illicit) = 0.0029 (< tau*=0.57), resulting in a missed detection (FN). Negative (licit-pulling) attribution from Aggregate_feature_7 (-2.208) and Local_feature_21 (-1.685) (total negative IG = -31.277) and weak/absent illicit neighbor support suppressed the logit below the 0.57 decision boundary relative to positive cues from Aggregate_feature_20 (+2.287). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (0/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -0.078 logits, versus self-node IG = -1.469 logits).

---

### VAL_FN_2 (Validation_Primary) — `FN`
*Selection Rationale:* Near-threshold connected Validation FN (just below tau*)

Transaction Node: `116055520` (Snapshot node index `2554`)

Timestep: `38`

Ground-Truth Dataset Label: `Illicit` (`y=1`)

Predicted Fraud Probability: `0.5083` (logit `+0.0330` vs. baseline `-4.3255`, $\Delta = +4.3585$)

Prediction: `Licit` (using frozen threshold `0.57`)

Classification: `FN`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Local_feature_90` (idx `89`, `Local93`): `+6.24550` (self: `+6.71423`, nbrs: `-0.46873`; raw=`-0.7255`, RankGauss=`-2.516`)
  - `Aggregate_feature_49` (idx `141`, `Aggregate72`): `+6.09399` (self: `+4.90846`, nbrs: `+1.18553`; raw=`4.7492`, RankGauss=`+2.235`)
  - `Local_feature_14` (idx `13`, `Local93`): `+5.78903` (self: `+6.44388`, nbrs: `-0.65485`; raw=`-0.0563`, RankGauss=`-2.799`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-9.20603` (self: `-6.94926`, nbrs: `-2.25678`; raw=`2.2829`, RankGauss=`+5.201`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-8.84776` (self: `-6.67246`, nbrs: `-2.17530`; raw=`2.2172`, RankGauss=`+5.202`)
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-8.20205` (self: `-6.67793`, nbrs: `-1.52412`; raw=`2.2891`, RankGauss=`+5.201`)

Graph Context (Strictly within Timestep `38`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `0/1`, mean P=`0.002`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `0/1`, mean P=`0.003`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `70`
- IG Attribution Origin Split (Logits): Self=`+8.7837`, 1-Hop In=`-4.4956`, 1-Hop Out=`+0.0609`, 2-Hop Only=`-0.0171`
- Relevant neighbouring predictions: Clean 1-hop neighborhood (0/2 predicted Illicit), but 20/70 2-hop neighbors flagged Illicit

Explanation:
Although the dataset records this transaction as Illicit, the frozen model assigns P(Illicit) = 0.5083 (< tau*=0.57), resulting in a missed detection (FN). Negative (licit-pulling) attribution from Aggregate_feature_10 (-9.206) and Aggregate_feature_7 (-8.848) (total negative IG = -79.530) and weak/absent illicit neighbor support suppressed the logit below the 0.57 decision boundary relative to positive cues from Local_feature_90 (+6.246). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (0/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -4.452 logits, versus self-node IG = +8.784 logits).

---

### VAL_TN_1 (Validation_Primary) — `TN`
*Selection Rationale:* Lowest-probability connected Validation TN benign baseline

Transaction Node: `97506379` (Snapshot node index `3328`)

Timestep: `40`

Ground-Truth Dataset Label: `Licit` (`y=0`)

Predicted Fraud Probability: `0.0022` (logit `-6.1339` vs. baseline `-4.3255`, $\Delta = -1.8084$)

Prediction: `Licit` (using frozen threshold `0.57`)

Classification: `TN`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Local_feature_88` (idx `87`, `Local93`): `+0.45317` (self: `+0.16860`, nbrs: `+0.28458`; raw=`0.0990`, RankGauss=`+1.177`)
  - `Aggregate_feature_9` (idx `101`, `Aggregate72`): `+0.31709` (self: `+0.18256`, nbrs: `+0.13453`; raw=`-0.0874`, RankGauss=`+1.858`)
  - `Aggregate_feature_15` (idx `107`, `Aggregate72`): `+0.30313` (self: `+0.14410`, nbrs: `+0.15903`; raw=`1.6647`, RankGauss=`+2.393`)
- **Strongest Negative Drivers (Toward Licit):**
  - `out_BTC_min` (idx `177`, `Domain17`): `-0.50705` (self: `-0.20336`, nbrs: `-0.30370`; raw=`2.0942`, RankGauss=`+2.596`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-0.49594` (self: `-0.17527`, nbrs: `-0.32066`; raw=`2.4077`, RankGauss=`+5.202`)
  - `Local_feature_16` (idx `15`, `Local93`): `-0.42055` (self: `-0.17857`, nbrs: `-0.24198`; raw=`2.0931`, RankGauss=`+2.591`)

Graph Context (Strictly within Timestep `40`, original direction preserved):
- Incoming neighbours (`w -> target`): `3` (predicted Illicit: `0/3`, mean P=`0.003`; GT labels: `0` Illicit, `2` Licit, `1` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `0/1`, mean P=`0.002`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `4` / `15`
- IG Attribution Origin Split (Logits): Self=`-0.4880`, 1-Hop In=`-1.0172`, 1-Hop Out=`-0.2162`, 2-Hop Only=`-0.0917`
- Relevant neighbouring predictions: Predominantly benign predicted neighborhood: 0/4 1-hop neighbors predicted Illicit (mean P=0.003, max P=0.003)

Explanation:
The frozen model assigns P(Illicit) = 0.0022 (< tau*=0.57), consistent with the dataset Licit label (TN). Strong negative attribution from out_BTC_min (-0.507) and Aggregate_feature_7 (-0.496) (total negative IG = -8.604) along with non-suspicious feature/neighborhood context keeps the prediction safely below tau*=0.57. Its direct snapshot neighborhood has 3 incoming and 1 outgoing edges (0/4 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -1.325 logits, versus self-node IG = -0.488 logits).

---

### VAL_TN_2 (Validation_Primary) — `TN`
*Selection Rationale:* Elevated-score connected Validation TN (below tau* but non-trivial risk)

Transaction Node: `201433814` (Snapshot node index `5587`)

Timestep: `36`

Ground-Truth Dataset Label: `Licit` (`y=0`)

Predicted Fraud Probability: `0.4688` (logit `-0.1251` vs. baseline `-4.3255`, $\Delta = +4.2004$)

Prediction: `Licit` (using frozen threshold `0.57`)

Classification: `TN`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_49` (idx `141`, `Aggregate72`): `+7.21530` (self: `+3.14865`, nbrs: `+4.06664`; raw=`0.0461`, RankGauss=`+1.264`)
  - `Aggregate_feature_21` (idx `113`, `Aggregate72`): `+6.30458` (self: `+2.58202`, nbrs: `+3.72256`; raw=`0.5007`, RankGauss=`+2.329`)
  - `Aggregate_feature_15` (idx `107`, `Aggregate72`): `+5.31214` (self: `+1.74955`, nbrs: `+3.56259`; raw=`0.2285`, RankGauss=`+2.213`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-9.79775` (self: `-2.85897`, nbrs: `-6.93878`; raw=`2.0270`, RankGauss=`+5.202`)
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-7.97306` (self: `-1.26516`, nbrs: `-6.70791`; raw=`2.0909`, RankGauss=`+5.201`)
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-7.90651` (self: `-0.94431`, nbrs: `-6.96219`; raw=`2.0859`, RankGauss=`+5.201`)

Graph Context (Strictly within Timestep `36`, original direction preserved):
- Incoming neighbours (`w -> target`): `4` (predicted Illicit: `0/4`, mean P=`0.142`; GT labels: `0` Illicit, `4` Licit, `0` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `1/1`, mean P=`0.992`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `5` / `1`
- IG Attribution Origin Split (Logits): Self=`+11.7440`, 1-Hop In=`-6.6891`, 1-Hop Out=`+0.1741`, 2-Hop Only=`-1.0701`
- Relevant neighbouring predictions: Mixed neighborhood with localized suspicion: 1/5 (20.0%) 1-hop neighbors predicted Illicit (max P=0.992)

Explanation:
The frozen model assigns P(Illicit) = 0.4688 (< tau*=0.57), consistent with the dataset Licit label (TN). Strong negative attribution from Aggregate_feature_7 (-9.798) and Aggregate_feature_8 (-7.973) (total negative IG = -117.193) along with non-suspicious feature/neighborhood context keeps the prediction safely below tau*=0.57. Its direct snapshot neighborhood has 4 incoming and 1 outgoing edges (1/5 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -7.585 logits, versus self-node IG = +11.744 logits).

---

### TEST_TP_42 (Test_Exploratory) — `TP`
*Selection Rationale:* Exploratory Test Step 42 high-confidence connected TP

Transaction Node: `71989375` (Snapshot node index `2105`)

Timestep: `42`

Ground-Truth Dataset Label: `Illicit` (`y=1`)

Predicted Fraud Probability: `0.9982` (logit `+6.2997` vs. baseline `-4.3255`, $\Delta = +10.6251$)

Prediction: `Illicit` (using frozen threshold `0.57`)

Classification: `TP`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_20` (idx `112`, `Aggregate72`): `+4.02124` (self: `+2.56661`, nbrs: `+1.45463`; raw=`1.6816`, RankGauss=`+1.917`)
  - `Local_feature_3` (idx `2`, `Local93`): `+3.97848` (self: `+2.97367`, nbrs: `+1.00482`; raw=`2.0007`, RankGauss=`+2.814`)
  - `Local_feature_14` (idx `13`, `Local93`): `+3.42807` (self: `+3.54262`, nbrs: `-0.11456`; raw=`-0.0563`, RankGauss=`-2.799`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-5.30413` (self: `-3.36395`, nbrs: `-1.94017`; raw=`2.5984`, RankGauss=`+5.202`)
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-5.22713` (self: `-3.17983`, nbrs: `-2.04730`; raw=`2.6778`, RankGauss=`+5.201`)
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-4.63147` (self: `-2.99076`, nbrs: `-1.64071`; raw=`2.6862`, RankGauss=`+5.201`)

Graph Context (Strictly within Timestep `42`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `0/1`, mean P=`0.022`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `0/1`, mean P=`0.038`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `50`
- IG Attribution Origin Split (Logits): Self=`+5.1414`, 1-Hop In=`+2.6673`, 1-Hop Out=`+2.8620`, 2-Hop Only=`-0.0380`
- Relevant neighbouring predictions: Clean 1-hop neighborhood (0/2 predicted Illicit), but 48/50 2-hop neighbors flagged Illicit

Explanation:
The frozen model assigns P(Illicit) = 0.9982 (>= tau*=0.57), matching the dataset Illicit label (TP). Positive fraud attribution is led by Aggregate_feature_20 (+4.021) and Local_feature_3 (+3.978) (total positive IG = +63.186), outweighing mitigating negative evidence from Aggregate_feature_7 (-5.304) and Aggregate_feature_10 (-5.227) (-52.553). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (0/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = +5.491 logits, versus self-node IG = +5.141 logits).

---

### TEST_FP_1 (Test_Exploratory) — `FP`
*Selection Rationale:* Exploratory Test high-confidence connected FP

Transaction Node: `106364923` (Snapshot node index `3229`)

Timestep: `44`

Ground-Truth Dataset Label: `Licit` (`y=0`)

Predicted Fraud Probability: `0.9934` (logit `+5.0157` vs. baseline `-4.3255`, $\Delta = +9.3412$)

Prediction: `Illicit` (using frozen threshold `0.57`)

Classification: `FP`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Local_feature_88` (idx `87`, `Local93`): `+8.04117` (self: `+7.81957`, nbrs: `+0.22160`; raw=`1.9277`, RankGauss=`+1.751`)
  - `Aggregate_feature_49` (idx `141`, `Aggregate72`): `+6.83330` (self: `+5.46202`, nbrs: `+1.37128`; raw=`5.9586`, RankGauss=`+2.300`)
  - `Local_feature_90` (idx `89`, `Local93`): `+6.18008` (self: `+5.79027`, nbrs: `+0.38981`; raw=`-0.7255`, RankGauss=`-2.516`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-8.33322` (self: `-5.53362`, nbrs: `-2.79960`; raw=`2.7890`, RankGauss=`+5.202`)
  - `Aggregate_feature_10` (idx `102`, `Aggregate72`): `-7.68288` (self: `-4.96604`, nbrs: `-2.71684`; raw=`2.8753`, RankGauss=`+5.201`)
  - `Aggregate_feature_8` (idx `100`, `Aggregate72`): `-6.82274` (self: `-4.24389`, nbrs: `-2.57886`; raw=`2.8849`, RankGauss=`+5.201`)

Graph Context (Strictly within Timestep `44`, original direction preserved):
- Incoming neighbours (`w -> target`): `1` (predicted Illicit: `0/1`, mean P=`0.037`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `0/1`, mean P=`0.005`; GT labels: `0` Illicit, `1` Licit, `0` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `2` / `48`
- IG Attribution Origin Split (Logits): Self=`-0.4365`, 1-Hop In=`+3.7154`, 1-Hop Out=`+6.3084`, 2-Hop Only=`-0.3770`
- Relevant neighbouring predictions: Clean 1-hop neighborhood (0/2 predicted Illicit), but 7/48 2-hop neighbors flagged Illicit

Explanation:
Although the dataset records this transaction as Licit, the frozen model assigns an elevated P(Illicit) = 0.9934 (>= tau*=0.57), creating a False Positive alarm. The model's score is driven upward primarily by Local_feature_88 (+8.041) and Aggregate_feature_49 (+6.833) (total positive IG = +92.106) which resemble illicit patterns under RankGauss scaling, overriding negative Licit cues from Aggregate_feature_7 (-8.333) and Aggregate_feature_10 (-7.683). Its direct snapshot neighborhood has 1 incoming and 1 outgoing edges (0/2 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = +9.647 logits, versus self-node IG = -0.436 logits). This attribution explains why the classifier reacted as it did, not that the transaction is fraudulent.

---

### TEST_FN_POST43 (Test_Exploratory) — `FN`
*Selection Rationale:* Exploratory Test post-shutdown (Steps 43-49) connected FN collapse example

Transaction Node: `113621810` (Snapshot node index `4627`)

Timestep: `44`

Ground-Truth Dataset Label: `Illicit` (`y=1`)

Predicted Fraud Probability: `0.0031` (logit `-5.7885` vs. baseline `-4.3255`, $\Delta = -1.4630$)

Prediction: `Licit` (using frozen threshold `0.57`)

Classification: `FN`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_6` (idx `98`, `Aggregate72`): `+0.88447` (self: `+0.48959`, nbrs: `+0.39489`; raw=`-2.4534`, RankGauss=`-2.251`)
  - `Local_feature_3` (idx `2`, `Local93`): `+0.57745` (self: `+0.33974`, nbrs: `+0.23772`; raw=`1.4190`, RankGauss=`+1.473`)
  - `Aggregate_feature_55` (idx `147`, `Aggregate72`): `+0.53607` (self: `+0.30530`, nbrs: `+0.23078`; raw=`1.6627`, RankGauss=`+1.328`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-1.12920` (self: `-0.41829`, nbrs: `-0.71090`; raw=`2.7887`, RankGauss=`+5.202`)
  - `Aggregate_feature_70` (idx `162`, `Aggregate72`): `-0.93893` (self: `-0.93406`, nbrs: `-0.00488`; raw=`0.7529`, RankGauss=`+2.217`)
  - `Aggregate_feature_68` (idx `160`, `Aggregate72`): `-0.60173` (self: `-0.65048`, nbrs: `+0.04874`; raw=`0.2025`, RankGauss=`+2.241`)

Graph Context (Strictly within Timestep `44`, original direction preserved):
- Incoming neighbours (`w -> target`): `2` (predicted Illicit: `0/2`, mean P=`0.004`; GT labels: `0` Illicit, `0` Licit, `2` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `1` (predicted Illicit: `0/1`, mean P=`0.007`; GT labels: `0` Illicit, `0` Licit, `1` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `3` / `2`
- IG Attribution Origin Split (Logits): Self=`-0.1767`, 1-Hop In=`-1.6858`, 1-Hop Out=`+0.5194`, 2-Hop Only=`-0.1173`
- Relevant neighbouring predictions: Predominantly benign predicted neighborhood: 0/3 1-hop neighbors predicted Illicit (mean P=0.005, max P=0.007)

Explanation:
Although the dataset records this transaction as Illicit, the frozen model assigns P(Illicit) = 0.0031 (< tau*=0.57), resulting in a missed detection (FN). Negative (licit-pulling) attribution from Aggregate_feature_7 (-1.129) and Aggregate_feature_70 (-0.939) (total negative IG = -12.569) and weak/absent illicit neighbor support suppressed the logit below the 0.57 decision boundary relative to positive cues from Aggregate_feature_6 (+0.884). Its direct snapshot neighborhood has 2 incoming and 1 outgoing edges (0/3 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -1.284 logits, versus self-node IG = -0.177 logits).

---

### TEST_TN_1 (Test_Exploratory) — `TN`
*Selection Rationale:* Exploratory Test lowest-probability connected TN

Transaction Node: `94333210` (Snapshot node index `2607`)

Timestep: `43`

Ground-Truth Dataset Label: `Licit` (`y=0`)

Predicted Fraud Probability: `0.0021` (logit `-6.1468` vs. baseline `-4.3255`, $\Delta = -1.8213$)

Prediction: `Licit` (using frozen threshold `0.57`)

Classification: `TN`

Important Features (Signed Graph Integrated Gradients across Self + 2-Hop Neighbors; `+` pushes Illicit, `-` pushes Licit):
- **Strongest Positive Drivers (Toward Fraud):**
  - `Aggregate_feature_20` (idx `112`, `Aggregate72`): `+0.18074` (self: `+0.07023`, nbrs: `+0.11051`; raw=`2.2142`, RankGauss=`+2.678`)
  - `Aggregate_feature_22` (idx `114`, `Aggregate72`): `+0.13677` (self: `+0.03001`, nbrs: `+0.10676`; raw=`1.4215`, RankGauss=`+1.932`)
  - `Aggregate_feature_46` (idx `138`, `Aggregate72`): `+0.13203` (self: `+0.02377`, nbrs: `+0.10827`; raw=`2.2022`, RankGauss=`+2.242`)
- **Strongest Negative Drivers (Toward Licit):**
  - `Aggregate_feature_70` (idx `162`, `Aggregate72`): `-0.40827` (self: `-0.08685`, nbrs: `-0.32142`; raw=`1.6469`, RankGauss=`+2.587`)
  - `Aggregate_feature_7` (idx `99`, `Aggregate72`): `-0.36341` (self: `-0.04180`, nbrs: `-0.32161`; raw=`2.6935`, RankGauss=`+5.202`)
  - `Aggregate_feature_68` (idx `160`, `Aggregate72`): `-0.22624` (self: `-0.08112`, nbrs: `-0.14512`; raw=`1.5753`, RankGauss=`+3.195`)

Graph Context (Strictly within Timestep `43`, original direction preserved):
- Incoming neighbours (`w -> target`): `24` (predicted Illicit: `0/24`, mean P=`0.003`; GT labels: `0` Illicit, `8` Licit, `16` Unknown `-1`)
- Outgoing neighbours (`target -> w`): `2` (predicted Illicit: `0/2`, mean P=`0.002`; GT labels: `0` Illicit, `2` Licit, `0` Unknown `-1`)
- Unique 1-hop / 2-hop neighbours: `26` / `156`
- IG Attribution Origin Split (Logits): Self=`-0.2434`, 1-Hop In=`-1.5360`, 1-Hop Out=`-0.0023`, 2-Hop Only=`-0.0467`
- Relevant neighbouring predictions: Predominantly benign predicted neighborhood: 0/26 1-hop neighbors predicted Illicit (mean P=0.003, max P=0.008)

Explanation:
The frozen model assigns P(Illicit) = 0.0021 (< tau*=0.57), consistent with the dataset Licit label (TN). Strong negative attribution from Aggregate_feature_70 (-0.408) and Aggregate_feature_7 (-0.363) (total negative IG = -5.355) along with non-suspicious feature/neighborhood context keeps the prediction safely below tau*=0.57. Its direct snapshot neighborhood has 24 incoming and 2 outgoing edges (0/26 1-hop neighbors predicted Illicit; combined 1-hop + 2-hop neighbor IG contribution = -1.585 logits, versus self-node IG = -0.243 logits).

---
