# Google QUEST Q&A Labeling - Experiment Settings Log

This document records the hyperparameter configurations for the various experiments conducted to optimize the DeBERTa-v3-Large model.

## Common Settings
- **Model**: `microsoft/deberta-v3-large`
- **Max Sequence Length**: 1024
- **Pooling**: Weighted Layer Pooling (Last 4 Layers) + Attention Pooling
- **Optimizer**: AdamW (Encoder LR=1e-5, Head LR=1e-4)

---

## Experiment 1: Baseline (No BCE)
*Focus: Testing Ranking & Spearman loss only.*
- **Epochs**: 3 (Phase 1) + 3 (Phase 2)
- **Batch Size**: 8 (Accumulation=1)
- **Loss Weights**:
  - `bce_weight`: 0.0
  - `ranking_weight`: 0.5
  - `spearman_weight`: 0.5
- **AWP**: Disabled
- **Auto Weighting**: Disabled

- Training log: /home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_005927/training_log.jsonl

## Experiment 2: Long Training (No BCE)
*Focus: Testing if longer training helps without BCE.*
- **Epochs**: 5 (Phase 1) + 5 (Phase 2)
- **Batch Size**: 8 (Accumulation=1)
- **Loss Weights**:
  - `bce_weight`: 0.0
  - `ranking_weight`: 0.5
  - `spearman_weight`: 0.5
- **AWP**: Disabled
- **Auto Weighting**: Disabled
- Training log: /home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_010011/training_log.jsonl

## Experiment 3: Hybrid Loss (Strong Baseline)
*Focus: Adding BCE back for stability and signal.*
- **Epochs**: 3 (Phase 1) + 3 (Phase 2)
- **Batch Size**: 8 (Accumulation=1)
- **Loss Weights**:
  - `bce_weight`: 0.5
  - `ranking_weight`: 1.0
  - `spearman_weight`: 0.5
- **AWP**: Disabled
- **Auto Weighting**: Disabled
- Training log: /home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_010150/training_log.jsonl

## Experiment 4: Hybrid Loss + AWP (Cancelled)
*Note: This experiment was cancelled/invalidated due to a bug in the AWP implementation.*
- **Epochs**: 3 (Phase 1) + 3 (Phase 2)
- **Loss Weights**: Same as Exp 3
- **AWP**: Enabled (`awp_lr=1e-4`, `awp_eps=1e-2`, `start_epoch=2`)

## Experiment 5: Automatic Loss Weighting
*Focus: Letting the model learn optimal loss weights.*
- **Epochs**: 4 (Phase 1) + 4 (Phase 2)
- **Batch Size**: 4 (Accumulation=2)
- **Loss Weights**: Dynamic (Auto Weighting Enabled)
  - `use_auto_weighting`: True
- **AWP**: Disabled
- Training log: /home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_011054/training_log.jsonl

## Experiment 6: The "All-In" Run (Current Best Config)
*Focus: Combining Hybrid Loss stability, AWP generalization, and long training convergence.*
- **Epochs**: 5 (Phase 1) + 5 (Phase 2)
- **Batch Size**: 4 (Accumulation=2) -> Effective Batch Size = 8
- **Loss Weights**:
  - `bce_weight`: 0.5
  - `ranking_weight`: 1.0
  - `spearman_weight`: 0.5
- **AWP**: Enabled
  - `awp_lr`: 1e-4
  - `awp_eps`: 1e-2
  - `awp_start_epoch`: 2
- **Auto Weighting**: Disabled
- Training log: /home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_011650/training_log.jsonl

---

## Experiment Analysis (2025-12-01)

## Experiment Analysis (2025-12-01)

### 1. Best Performance per Fold (Valid Spearman)

| Fold | Exp 0 (Old Long) | Exp 1 (Base) | Exp 2 (Long) | Exp 3 (Hybrid) | Exp 5 (Auto) | **Exp 6 (All-In)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **0.4239** 👑 | 0.4116 | 0.4217 | 0.4110 | 0.4194 | 0.4216 |
| **1** | **0.4205** 👑 | 0.4133 | 0.4186 | 0.4135 | 0.4135 | 0.4176 |
| **2** | 0.4247 | 0.4190 | 0.4228 | 0.4192 | 0.4188 | **0.4255** 👑 |
| **3** | 0.4241 | 0.4126 | 0.4230 | 0.4097 | 0.4193 | **0.4242** 👑 |
| **4** | **0.4209** 👑 | 0.4088 | 0.4190 | 0.4082 | 0.4133 | 0.4203 |
| **AVG**| **0.4228** 👑 | 0.4131 | 0.4210 | 0.4123 | 0.4169 | 0.4218 |
| **STD**| 0.0018 | 0.0033 | 0.0019 | 0.0038 | 0.0029 | 0.0028 |

### 2. Key Insights
- **Exp 0 (Old Long)**: Remains the strongest overall performer with the highest average score (0.4228) and wins in 3 out of 5 folds (0, 1, 4). This confirms that the "No BCE" + Long Training strategy is extremely effective.
- **Exp 6 (All-In)**: Very close second (Avg 0.4218) and wins in the hardest folds (2 & 3). The addition of AWP and Hybrid Loss provides stability and peak performance in difficult cases, but slightly lags behind Exp 0 in easier folds.
- **Exp 2 (Long Train)**: Performs consistently well (Avg 0.4210), reinforcing the value of longer training schedules (10 epochs).
- **Conclusion**: Exp 0 and Exp 6 are the top contenders. AWP in Exp 6 seems to help with generalization in harder folds, while Exp 0's simpler objective works best for the others. An ensemble of Exp 0 and Exp 6 would likely yield the best results.
