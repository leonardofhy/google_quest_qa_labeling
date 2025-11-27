# Google QUEST Q&A Labeling 改進方案分析

這份文檔基於第 23 名解決方案 (`src/train_and_inference_bert_23rd_place.py`) 的架構分析，並結合現代 NLP 技術提出改進建議。

## 1. 當前架構解析 (23rd Place Solution)

這份代碼的核心亮點在於它如何處理長文本以及如何分階段訓練模型。

### 雙頭架構 (Dual-Head Architecture)
*   **機制**: 模型不僅使用了 `[CLS]` token（代表整句語義），還特別提取了分隔 Question 和 Answer 的 `[SEP]` token。
*   **特徵融合**: 提取 BERT **最後 4 層**的 `[CLS]` 和 `[SEP]` 向量進行拼接 (Concatenation)。
*   **目的**: `[CLS]` 捕捉全局信息，而中間的 `[SEP]` 更能捕捉 "問題" 與 "回答" 之間的交互關係。

### 課程學習 (Curriculum Learning / Two-Stage Training)
*   **第一階段**: 先只訓練 **21 個與問題相關的標籤 (Question Targets)**。
*   **第二階段**: 再訓練所有 **30 個標籤 (Q + A)**。
*   **原理**: 問題類的標籤（如 `question_fact_seeking`）通常比較容易學習且只依賴前半段文本。先學好這些能讓模型在進入更難的 Q&A 匹配任務前，具備更好的特徵表示。

### 平衡截斷 (Balanced Trimming)
*   **機制**: 當文本過長時，代碼中的 `_trim_input` 函數不會簡單地截斷末尾，而是會同時保留 Question 和 Answer 的**開頭與結尾**，確保關鍵信息不丟失。

### LightGBM Stacking
*   **機制**: 將每個 Epoch 的預測結果作為特徵，訓練了一個 LightGBM 模型來做最終融合，而非簡單平均。

---

## 2. 改進方向 (基於 2025 視角)

為了將方案提升到 SOTA (State-of-the-Art) 水平，建議從以下方向改進：

### A. 模型與架構升級 (Model & Architecture)
1.  **升級 Backbone (BERT -> DeBERTa-v3)**:
    *   **建議**: 將 `BertModel` 替換為 `AutoModel.from_pretrained("microsoft/deberta-v3-large")`。
    *   **理由**: DeBERTa-v3 的 Disentangled Attention 機制對成對文本 (Pairwise) 的理解能力遠強於 BERT。

2.  **多樣本 Dropout (Multi-Sample Dropout)**:
    *   **建議**: 在最後的分類層使用 5 個不同的 Dropout mask，計算 5 次 Loss 並取平均。
    *   **理由**: 能顯著加速收斂並提高泛化能力。

### B. 訓練策略優化 (Training Strategy)
3.  **對抗訓練 (AWP - Adversarial Weight Perturbation)**:
    *   **建議**: 在訓練過程中，對模型權重加入微小噪聲。
    *   **理由**: 這是目前 Kaggle 競賽標配，能強迫模型在更平坦的 Loss 地形上尋找最優解，極大提升魯棒性。

4.  **分層學習率衰減 (Layer-wise Learning Rate Decay, LLRD)**:
    *   **建議**: 對 Transformer 的每一層設置不同的學習率（如每層衰減 0.95）。
    *   **理由**: 保護預訓練模型的底層特徵不被破壞，同時讓頂層快速適應新任務。

### C. 損失函數與後處理 (Loss & Post-processing)
5.  **引入 Spearman Loss**:
    *   **建議**: 引入 **Soft Spearman Loss** (可微分的排序損失)。
    *   **理由**: 直接優化比賽的評估指標 (Spearman Correlation)。

6.  **目標分佈匹配 (Target Distribution Matching, TDM)**:
    *   **建議**: 強制讓預測結果的數值分佈（Rank Distribution）與訓練集的真實標籤分佈一致。
    *   **理由**: 相比 LightGBM Stacking 更簡單且穩定有效。

## 總結
保留 **"課程學習"** 和 **"雙頭架構"** 的思路，將底層模型換成 **DeBERTa-v3**，並加上 **AWP** 和 **Multi-Sample Dropout**，將構成極具競爭力的方案。
