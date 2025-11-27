# Experiment Log: Google QUEST Q&A Labeling

## Overview
This document records the iterative experiments conducted to improve the DeBERTa-v3-large model performance.

**Goal**: Improve Spearman Correlation metric by adapting tricks from the 1st and 2nd place solutions.

## Experiment Summary

| ID | Experiment Name | Key Changes | Best Val Score (Spearman) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Exp 0** | **Baseline** | Pure Ranking Loss (Rank=0.65, Spear=0.35) | **0.4036** | Baseline |
| **Exp 1** | **Hybrid Loss v1** | Added BCE Loss (Weight=1.0) | 0.3954 | Failed (Regression) |
| **Exp 2** | **Hybrid Loss v2** | **Rebalanced Weights (BCE=0.5, Rank=1.0)** | **0.4056** | **Success (Golden Recipe)** |
| **Exp 3** | **Architecture** | Weighted Layer Pooling + Multi-Sample Dropout | 0.4004 | Failed (Regression) |

---

## Detailed Analysis

### Experiment 0: Baseline
*   **Configuration**:
    *   Model: `microsoft/deberta-v3-large`
    *   Loss: Pairwise Ranking (0.65) + Soft Spearman (0.35)
    *   Pooling: Concatenation of last 4 layers
*   **Result**: 0.4036 (Epoch 4)
*   **Observation**: Strong baseline, but potential instability in gradients due to lack of pointwise loss (BCE).

### Experiment 1: Hybrid Loss v1
*   **Hypothesis**: Adding BCE Loss will stabilize training and improve pointwise accuracy.
*   **Configuration**:
    *   Added `BCEWithLogitsLoss` with `weight=1.0`.
*   **Result**: 0.3954 (Epoch 4)
*   **Analysis**: The strong BCE weight diluted the ranking signal. The model focused too much on binary classification (0/1) rather than the subtle ranking required for the metric.

### Experiment 2: Hybrid Loss v2 (The Golden Recipe)
*   **Hypothesis**: Reducing BCE weight will allow Ranking Loss to lead while still providing stability.
*   **Configuration**:
    *   `bce_loss_weight`: **0.5** (Auxiliary)
    *   `ranking_loss_weight`: **1.0** (Primary)
    *   `spearman_loss_weight`: **0.5**
*   **Result**: **0.4056** (Epoch 4)
*   **Analysis**:
    *   **Best Performance**: Surpassed baseline.
    *   **Convergence**: Fast convergence (Epoch 1: 0.3522).
    *   **Conclusion**: This balance effectively combines the stability of BCE with the metric-optimization of Ranking Loss.

### Experiment 3: Architecture Optimization
*   **Hypothesis**: "Weighted Layer Pooling" (from 2nd place solution) and "Multi-Sample Dropout" will improve feature extraction and generalization.
*   **Configuration**:
    *   Replaced Last-4-Layer Concatenation with Learnable Weighted Sum of all layers.
    *   Added Multi-Sample Dropout (5 samples).
*   **Result**: 0.4004 (Epoch 4)
*   **Analysis**:
    *   **Regression**: Score dropped compared to Exp 2.
    *   **Reason**: Likely over-engineering for the current dataset size/setup. The simpler concatenation strategy proved more robust.
*   **Action**: Reverted changes.

---

## Final Configuration (Golden Recipe)

The code has been reverted to the state of **Experiment 2**.

### Key Parameters
*   **Model**: `microsoft/deberta-v3-large`
*   **Loss Weights**:
    *   BCE: 0.5
    *   Ranking: 1.0
    *   Spearman: 0.5
*   **Architecture**:
    *   Pooler: Concatenation of Last 4 Hidden Layers
    *   Head: Dual-Head (CLS + SEP)
*   **Training**:
    *   Epochs: 8 (User setting) / 6 (Recommended)
    *   Post-processing: Enabled for Inference ONLY (Disabled for Validation)
