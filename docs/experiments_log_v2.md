# Experiment Log

This document tracks the experiments conducted for the Google QUEST Q&A Labeling competition.

## Baseline

### Experiment ID: `20251129_024721`
- **Model**: `google-bert/bert-base-uncased`
- **CV Score (Spearman)**: `0.4195`
- **Key Settings**:
    - Max Seq Length: 512
    - Batch Size: 8
    - LR (BERT): 5e-5
    - LR (Head): 5e-4
    - Epochs: 3 (Q) + 3 (Q&A)
- **Logs**:
    - Config: `models/20251129_024721/config.json`
    - Summary: `models/20251129_024721/summary.json`
    - Training Log: `models/20251129_024721/training_log.jsonl`
- **Description**: 
    - **Original 23rd place solution**.
    - Uses `bert-base-uncased`.
    - Uses `_trim_input` (Head+Tail truncation logic).
    - Uses `[CLS] + [SEP]` concatenation pooling.
    - **Full 5-fold CV**.
    - Includes LightGBM Stacking.

---

## Experiments

| ID | Model | CV Score | Valid Loss (Q&A) | Changes | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | BERT Base Uncased | **0.4195** | ~0.1682 | Original 23rd place solution. | ✅ Done |
| **Exp 1** | DeBERTa v3 (Basic) | ~0.3465 (Fold 0) | ~0.1700 | Corrected Tokenizer, Mean Pooling, Lower LR. | ✅ Done (Local) |
| **Exp 2** | DeBERTa v3 (Advanced) | ~0.3586 (Fold 0) | ~0.1696 | + WLP, MSD, LLRD. | ✅ Done (Local) |
| **Exp 3** | DeBERTa v3 (Adv + H+T) | **~0.3642 (Fold 0)** | **~0.1692** | + Head+Tail Truncation. Seq Len 512. | ✅ Done (Local) |
| **Exp 4** | DeBERTa v3 (Adv + H+T + Seq1024) | **~0.3797 (Fold 0)** | **~0.1678** | Increased Seq Len to 1024. **No AWP**. | ✅ Done (Local) |
| **Exp 5** | DeBERTa v3 (Adv + H+T + AWP + Seq1024) | ~0.2734 (Q Only) | ~0.1538 (Q Only) | + AWP (eps=1e-2). Seq Len 1024. | ❌ Failed (Drop) |
| **Exp 6** | DeBERTa v3 (Adv + H+T + Low AWP + Seq1024) | ~0.3354 (Fold 0) | ~0.1700 | + AWP (eps=1e-3). Seq Len 1024. | ❌ Failed (Drop) |
| **Exp 7** | DeBERTa v3 Large (Adv + H+T + Seq1024) | **0.4218** | ~0.15 (est) | **Large Model**. **No AWP**. | ✅ Done |

### Detailed Results

#### Exp 1: DeBERTa v3 (Basic)
- **Log**: `models/20251129_025244/training_log.jsonl` (Early entries)
- **Fold 0 Metrics (Epoch 3)**:
    - Q Only: Loss 0.1522, Spearman 0.2795
    - Q&A: Loss 0.1700, Spearman 0.3465

#### Exp 2: DeBERTa v3 (Advanced)
- **Log**: `models/20251129_035208/training_log.jsonl` (Middle entries)
- **Fold 0 Metrics (Epoch 3)**:
    - Q Only: Loss 0.1517, Spearman 0.2795
    - Q&A: Loss 0.1696, Spearman 0.3586

#### Exp 3: DeBERTa v3 (Adv + H+T)
- **Log**: `models/20251129_044436/training_log.jsonl` (Also in `025244` later entries)
- **Fold 0 Metrics (Epoch 3)**:
    - Q Only: Loss 0.1516, Spearman 0.2879
    - Q&A: Loss 0.1692, Spearman 0.3642

#### Exp 4: DeBERTa v3 (Adv + H+T + Seq1024)
- **Log**: `models/20251129_054500/training_log.jsonl`
- **Fold 0 Metrics (Epoch 3)**:
    - Q Only: Loss 0.1498, Spearman 0.2974
    - Q&A: Loss 0.1678, Spearman 0.3797
    - **Note**: Surpassed Baseline (0.3782) in single-fold performance!

#### Exp 5: DeBERTa v3 (Adv + H+T + AWP + Seq1024)
- **Log**: `models/20251129_055237/training_log.jsonl`
- **Fold 0 Metrics (Epoch 3)**:
    - Q Only: Loss 0.1538, Spearman 0.2734
    - Q&A: Loss 0.1707, Spearman 0.3356
    - **Note**: Significant performance drop observed (0.3797 -> 0.3356). AWP epsilon 1e-2 was too aggressive.

#### Exp 6: DeBERTa v3 (Adv + H+T + Low AWP + Seq1024)
- **Log**: `models/20251129_061146/training_log.jsonl`
- **Fold 0 Metrics (Epoch 3)**:
    - Q Only: Loss 0.1530, Spearman 0.2750
    - Q&A: Loss 0.1700, Spearman 0.3354
    - **Note**: Lowering AWP epsilon to 1e-3 did not recover performance (0.3797 -> 0.3354). Decided to disable AWP for Large model training.

#### Exp 7: DeBERTa v3 Large (Adv + H+T + Seq1024)
- **Log**: `models/20251129_063948/training_log.jsonl`
- **CV Score (Spearman)**: **0.4218** (Local Eval with Stacking)
- **Configuration**:
    - Model: `microsoft/deberta-v3-large`
    - Max Seq Length: 1024
    - AWP: **Disabled** (`adv_lr=0`)
    - Folds: 5 (Full CV)
- **Note**: 
    - **Surpassed Best Baseline (0.4195)**.
    - Successfully scaled up to Large model with 1024 sequence length (AWP disabled).
    - Includes LightGBM Stacking on local evaluation.

## Analysis Notes

- **Baseline Strength**: The baseline achieves a very high CV of 0.4195. This is likely due to the combination of:
    1.  **Effective Input Trimming**: The `_trim_input` function preserves critical information at the start and end of the text.
    2.  **Full CV & Stacking**: The score includes LightGBM stacking, which usually boosts performance significantly.
    3.  **Robustness**: Even with suboptimal Tokenizer usage, the model learns well.

- **Optimization Progress**:
    - Our "Advanced" model (Exp 2) improved over the "Basic" one (Exp 1).
    - **Head+Tail Truncation (Exp 3)** provided a significant boost.
    - **Seq Len 1024 (Exp 4)**: Increasing sequence length to 1024 provided a massive boost (+0.0155), surpassing the Baseline's single-fold performance. This confirms that truncation was a major bottleneck.
    - **AWP (Exp 5)**: Initial run with `eps=1e-2` caused performance regression. Tuning epsilon to `1e-3` for Exp 6.

- **Next Steps**:
    - Verify Exp 4 (AWP) results.
    - Run Full 5-Fold CV on the best configuration to compare fairly with Baseline's 0.4195.
    - Implement Rank Averaging / Post-processing.
