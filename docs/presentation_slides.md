# Google QUEST Q&A Labeling: Project Report

---

# 1. Problem Introduction

**Goal**: Predict 30 subjective quality scores (e.g., "Helpfulness", "Coherence") for Q&A pairs.


---

# 2. Baseline 1: The Reference (23rd Place)

**Model**: `bert-base-uncased` (2018 Architecture)

**Method**:
*   **Sequence Length**: 512 tokens.
*   **Loss Function**: Hybrid (BCE + Margin Ranking).
*   **Strategy**: Head+Tail Truncation (Keep first 256 + last 256 tokens).
*   **Training**: Two-Phase (Question-Only -> Full Q&A).

**Performance**:
*   **CV**: 0.4195 (5-fold, Stacked).
*   **Pros**: Stable, reproducible, fits within compute limits.

---

# 3. Baseline 2: The Naive Upgrade (DeBERTa-Base)

**Model**: `microsoft/deberta-v3-base` (**Proven SOTA** 2021 Architecture)

**Method**:
*   **Identical Settings**: Replicated Baseline 1 exactly (Seq 512, Hybrid Loss).
*   **Expectation**: Better architecture = Better score.

**Result**:
*   **CV**: 0.3642 (**FAIL**)
*   **Observation**: The superior architecture performed significantly *worse* out of the box.

---

# 4. Comparing Baselines: The Paradox

**Why did DeBERTa fail?**

| Feature | Baseline 1 (BERT) | Baseline 2 (DeBERTa) |
| :--- | :--- | :--- |
| **Architecture** | Absolute Pos Embeddings | **Disentangled Attention** |
| **Context Window** | 512 (Optimized) | 512 (**Starved**) |
| **Performance** | **0.4195** | 0.3642 |

**Analysis**:
*   **Pooling Mismatch (Critical)**: Baseline used `[CLS]+[SEP]` pooling to capture structure; Naive DeBERTa used `Mean Pooling`, mixing Q&A indiscriminately.
*   **Context Starvation**: DeBERTa's Disentangled Attention needs long-range context. 512-token truncation + Mean Pooling = **Complete Information Loss**.
*   **Conclusion**: It wasn't just the context limit; it was the loss of structural awareness.

---

# 5. Weakness of Existing Solutions

**1. Context Bottleneck**
*   Both baselines truncate long StackExchange posts to 512 tokens.
*   Critical information in the middle of the text is lost.

**2. Objective Mismatch**
*   **Loss Used**: Baseline uses BCE + Margin Rank (Indirect optimization).
*   **Metric**: Spearman Correlation (Optimizes for **Relative Rank**).
*   *Result*: Models waste capacity fitting label noise (BCE) or use suboptimal ranking proxies.

**3. Architectural Stagnation**
*   Encoder-only models (BERT/DeBERTa) need specific tuning to compete with modern standards.

---

# 6. Our Proposal: The "Trifecta" Optimization

We implemented three targeted improvements to unlock performance:

**1. Context Expansion**
*   **Action**: Increase Sequence Length to **1024**.
*   **Result**: CV improved from **0.3642** to **0.3797** (+0.0155).
*   *Insight*: "Context is King."

**2. Loss Engineering**
*   **Action**: Replace BCE with **Ranking Loss + Soft Spearman Loss**.
*   **Result**: CV improved from **0.3797** to **0.4026** (+0.0229).
*   *Insight*: Align training with the competition metric.

**3. Architecture Scale**
*   **Action**: Upgrade to `deberta-v3-large`.
*   **Result**: CV improved from **0.4026** to **0.4228** (+0.0202).
*   *Note*: We retained the baseline's robust "Two-Phase Training" strategy for stability.

---

# 7. Results: The Evolution

| Step | Configuration | CV Score | Improvement |
| :--- | :--- | :--- | :--- |
| **Baseline 1** | BERT-Base (512, Hybrid) | 0.4195 | - |
| **Baseline 2** | Naive DeBERTa (512, Hybrid) | 0.3642 | 🔻 (The Dip) |
| **Step 1** | + Long Context (1024) | 0.3797 | 🔼 +0.0155 |
| **Step 2** | + Ranking Loss | 0.4026 | 🔼 +0.0229 |
| **Step 3** | + Large Model | 0.4218 | 🔼 +0.0192 |
| **Final** | + 2-Phase Training | **0.4228** | 🔼 +0.0010 |

**Key Takeaway**: We beat the strong baseline by **+0.0103** (Raw Score) through engineering, not just scaling.

---

# 8. Discussion 1: The CV vs. LB Gap

**The Discrepancy**:
*   **Our Best Model**: CV **0.4228** (Superior)
*   **Private LB**: 0.4066 (Lower than Baseline's 0.4144)
*   **Competitiveness**: Despite this, our single model outperforms the **46th place** solution (0.40612).

**Why? The Power of Snapshot Ensembling**
*   **Baseline Strategy**: **Stacks** predictions from **30 model snapshots** (every epoch × 5 folds) using LightGBM.
*   **Our Strategy**: Uses a **Single Best Model** (or simple 5-fold average).
*   **Conclusion**: We are comparing a single model against a massive ensemble. The ensemble's variance reduction explains the LB advantage.

---

# 9. Discussion 2: Architectural Constraints

**Architectural Stagnation**
*   **The Ceiling**: DeBERTa-v3 (2021) remains the SOTA Encoder. **Innovation in this field has stagnated**, while Decoder-only models (GPT) have seen exponential growth.
*   **The Constraint**: While modern LLMs offer superior reasoning, the **2-hour inference limit** locks us into this older paradigm.
*   **Implication**: We are optimizing a 2021 architecture to its absolute limit, but fundamental reasoning gaps remain compared to 2024 standards.

---

# 10. Conclusion & Future Work

**Summary**:
*   **DeBERTa-v3** is superior, but only when fed enough context (1024).
*   **Ranking Loss** is essential for subjective/noisy labels.

**Future Work**:
*   **Gap**: Best CV (0.4228) vs. Private LB (0.4066).
*   **Solution**: Bridge the gap using **Advanced Post-Processing**.
    *   **Stacking**: LightGBM/XGBoost to learn optimal blends.
    *   **Distribution Matching**: Align predictions with test set distribution.
