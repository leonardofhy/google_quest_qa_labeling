# Presentation Outline Plan

This document maps the content from `project_report.md` to your proposed slide structure.

## Narrative Strategy
The core narrative arc should be: **"The Trap of Naive Upgrades vs. The Power of First-Principles Engineering."**
*   **Conflict**: Simply swapping BERT for DeBERTa (Baseline 2) *failed* initially.
*   **Resolution**: We had to fix the *Context* (1024 tokens) and the *Objective* (Ranking Loss) to unlock the performance.

---

## Slide Breakdown

### 1. Problem Introduction
*   **Title**: Google QUEST Q&A Labeling
*   **The Goal**: Predict 30 subjective quality scores (e.g., "Helpfulness", "Coherence") for Q&A pairs.
*   **The Challenges**:
    *   **Data Scarcity**: Only ~6k training samples (High Overfitting Risk).
    *   **Label Noise**: Subjective human ratings (e.g., 0.33 vs 0.66).
    *   **Input Complexity**: Long technical text (StackExchange) > 512 tokens.
    *   **Constraint**: 2-hour Inference Limit (No LLMs allowed).

### 2. Baseline 1: The Reference (23rd Place)
*   **Model**: `bert-base-uncased` (2018 Architecture).
*   **Method**:
    *   Sequence Length: 512.
    *   Loss: Binary Cross Entropy (BCE).
    *   Strategy: Head+Tail Truncation.
*   **Performance**: CV 0.4195.
*   **Pros**: Stable, reproducible.

### 3. Baseline 2: The Naive Upgrade (DeBERTa-Base)
*   **Model**: `microsoft/deberta-v3-base` (SOTA 2021 Architecture).
*   **Method**:
    *   *Identical settings to Baseline 1* (Seq 512, BCE).
    *   "Drop-in replacement".
*   **Performance**: CV 0.3642 (Fail).
*   **Observation**: Newer architecture performed *worse* out of the box.

### 4. Comparing Baselines
*   **Baseline 1 (BERT)** vs. **Baseline 2 (DeBERTa)**.
*   **The Paradox**: Why did the superior architecture fail?
*   **Analysis**:
    *   DeBERTa needs context to shine.
    *   512 tokens forced severe truncation, "starving" the model.
    *   BERT was optimized for this constraint; DeBERTa was not.

### 5. Weakness of Existing Solutions
*   **Context Bottleneck**: Both baselines discard critical information due to the 512 limit.
*   **Objective Mismatch**:
    *   Both use **BCE Loss**, which optimizes for *absolute values* (fitting the noise).
    *   The Metric is **Spearman**, which cares about *relative rank*.
*   **Architectural Stagnation**: Encoder-only models need specific tuning to compete with modern standards.

### 6. Our Proposal: The "Trifecta" Optimization
*   **1. Context Expansion**:
    *   Increase Sequence Length to **1024**.
    *   Result: Immediate jump to CV 0.3797 (Context is King).
*   **2. Loss Engineering**:
    *   Abandon BCE.
    *   Adopt **Ranking Loss + Soft Spearman Loss**.
    *   Align training objective with competition metric.
*   **3. Architecture Scale**:
    *   Upgrade to `deberta-v3-large`.
    *   **Two-Phase Training**: Question-Only Pre-training -> Full Q&A Fine-tuning.

### 7. Results
*   **Visual**: A step chart showing the evolution.
    1.  **Baseline 1 (BERT)**: 0.4195
    2.  **Baseline 2 (Naive DeBERTa)**: 0.3642 (The Dip)
    3.  **+ Long Context (1024)**: 0.3797 (The Recovery)
    4.  **+ Ranking Loss**: 0.4026 (The Breakthrough)
    5.  **+ Large Model**: 0.4218
    6.  **+ 2-Phase Training (Final)**: **0.4228**
*   **Key Takeaway**: We beat the strong baseline by **+0.0103** (Raw Score) through engineering, not just scaling.

### 8. Conclusion
*   **Summary**:
    *   DeBERTa-v3 is superior, but only when fed enough context (1024).
    *   Ranking Loss > BCE for subjective/noisy labels.
*   **Future Work**:
    *   Bridge the CV-LB gap (0.4228 vs 0.4066) using **Advanced Post-Processing** (Stacking/Distribution Matching) to replicate the Baseline's ensemble stability.
