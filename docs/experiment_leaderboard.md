# Kaggle Submission Results

## Overview
This document records the performance of various trained models on the Kaggle Public and Private Leaderboards.

**Common Settings:**
- `use_stacking`: False
- `use_dist_matching`: False

## Results Table

| Timestamp | Model | Epochs | AWP | Loss Weights | CV Best Avg | Per-Fold CV (Best Avg) | Public LB | Private LB | Gap (CV-Pub) | Gap (Pub-Priv) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **20251201_155017** | `deberta-v3-large` | 10 | 0.001 | `BCE:0.0/Rnk:0.5/Spr:0.5` | 0.42169 | 0.4215 \| 0.4196 \| 0.4247 \| 0.4234 \| 0.4192 | **0.40254** | **0.38233** | 0.01915 | 0.02021 |
| **20251201_154836** | `deberta-v3-large` | 10 | 0.001 | `BCE:0.5/Rnk:1.0/Spr:0.5` | 0.42159 | 0.4228 \| 0.4174 \| 0.4253 \| 0.4239 \| 0.4186 | 0.40043 | 0.38097 | 0.02116 | 0.01946 |
| **20251201_200919** | `deberta-v3-base` | 10 | 0.001 | `BCE:0.5/Rnk:1.0/Spr:0.5` | 0.40179 | 0.4023 \| 0.4001 \| 0.4050 \| 0.4034 \| 0.3982 | 0.38505 | 0.36178 | 0.01674 | 0.02327 |
| **20251201_155128** | `deberta-v3-base` | 6 | 0.001 | `BCE:0.5/Rnk:1.0/Spr:0.5` | 0.39041 | 0.3878 \| 0.3893 \| 0.3996 \| 0.3916 \| 0.3838 | 0.37238 | 0.35158 | 0.01803 | 0.02080 |
| **20251202_051559** | `deberta-v3-large` | 6 | 0.001 | `BCE:0.5/Rnk:1.0/Spr:0.5` | 0.41131 | 0.4101 \| 0.4111 \| 0.4173 \| 0.4130 \| 0.4051 | 0.38732 | 0.36741 | 0.02399 | 0.01991 |

## Inference Logic (No Stacking / No Dist Matching)
When `use_stacking=False` and `use_dist_matching=False`, the submission is generated using **Simple Averaging**:
1.  **Inference**: The model predicts probabilities for the test set using each of the 5 trained folds.
2.  **Averaging**: The predictions from all 5 folds are averaged element-wise:
    $$ \text{Final Prediction} = \frac{1}{5} \sum_{fold=0}^{4} \text{Prediction}_{fold} $$
3.  **Output**: This averaged prediction is saved directly to `submission.csv` (after clipping to [0, 1]).

## Analysis Notes

### 1. CV vs LB Correlation
- **Strong Correlation**: The ranking of models by CV perfectly matches the ranking by Public and Private LB.
    - Highest CV (0.42169) -> Highest LB (0.40254)
    - Lowest CV (0.39041) -> Lowest LB (0.37238)
- **Conclusion**: Your local Cross-Validation (5 Folds) is **highly reliable** for model selection. You can trust that improving your local CV will likely improve your LB score.

### 2. Score Gap & Anomalies
- **Is there an anomaly?** No, the gap is consistent across all models, suggesting a systematic difference between the training/validation distribution and the test distribution (Public/Private), rather than a bug in a specific model.
- **Gap Breakdown**:
    - **CV to Public LB**: ~0.02 drop.
    - **Public LB to Private LB**: ~0.02 drop.
    - **Total Drop**: ~0.04 from CV to Private LB.
- **Implication**: Do not be alarmed by the absolute drop in score. Focus on the **relative improvement** in CV.

### 3. Model Size Impact
- **Large vs Base**: `deberta-v3-large` significantly outperforms `deberta-v3-base`.
    - Comparing similar configs (10 epochs, Hybrid Loss):
        - Large (154836): CV 0.42159 / Pub 0.40043
        - Base (200919): CV 0.40179 / Pub 0.38505
    - Improvement: **~0.02** in CV and **~0.015** in Public LB.

### 4. Loss Function Sensitivity
- **Hybrid vs Ranking-Only**:
    - Model `155017` (No BCE, only Ranking/Spearman) slightly outperformed `154836` (Hybrid with BCE) on the Leaderboard.
    - Public LB: 0.40254 vs 0.40043 (+0.00211)
    - Private LB: 0.38233 vs 0.38097 (+0.00136)
    - CV was almost identical (0.42169 vs 0.42159).
    - **Insight**: Pure ranking loss might generalize slightly better than mixing in BCE, or at least BCE didn't add value in this specific large model configuration.

### 5. Training Duration
- **10 Epochs vs 6 Epochs** (Base model):
    - 10 Epochs (200919): CV 0.40179 / Pub 0.38505
    - 6 Epochs (155128): CV 0.39041 / Pub 0.37238
    - **Insight**: Longer training (10 epochs) is clearly beneficial for this dataset, yielding a significant boost (~0.011 CV, ~0.013 LB).

## Final Kaggle Competition Leaderboard (Private LB - Top 10)

| Rank | Team Name | Private LB Score | Number of Entries | Time Since |
| :--- | :--- | :--- | :--- | :--- |
| 1 | bibimorph | 0.43100 | 95 | 6y |
| 2 | Berts and the holy Grail | 0.42895 | 66 | 6y |
| 3 | sakami | 0.42819 | 110 | 6y |
| 4 | BaBaConda & Jigsaw winners | 0.42788 | 125 | 6y |
| 5 | CGII | 0.42714 | 58 | 6y |
| 6 | CoolKidsOfUtrecht | 0.42705 | 90 | 6y |
| 7 | 武汉加油！中国加油！ | 0.42578 | 198 | 6y |
| 8 | on the way | 0.42563 | 272 | 6y |
| 9 | No shake for us !!! | 0.42476 | 121 | 6y |
| 10 | What else.. | 0.42430 | 199 | 6y |

**Your Current Best**: Private LB 0.38233 (Model 20251201_155017)

**Gap to Top 10**: +0.04867 (approximately 0.049)
**Gap to 1st Place**: +0.04867
