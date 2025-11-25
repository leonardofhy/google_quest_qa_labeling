# Google Quest Q&A Labeling - 技術方案文檔

## 📋 目錄
1. [問題定義](#1-問題定義)
2. [當前解決方案](#2-當前解決方案)
3. [未來優化方向](#3-未來優化方向)

---

## 1. 問題定義

### 1.1 競賽背景

**Google Quest Q&A Labeling** 是一個多標籤回歸問題，目標是預測 Q&A 配對的 30 個品質指標。

| 項目 | 說明 |
|------|------|
| **任務類型** | 多標籤回歸 (Multi-label Regression) |
| **目標數量** | 30 個連續標籤 (範圍 0-1) |
| **評估指標** | Mean Spearman Correlation (跨 30 個標籤取平均) |
| **數據規模** | 訓練集 ~6,079 筆，測試集 ~476 筆 |

### 1.2 目標標籤分類

```
Question 相關 (21 個):
├── 意圖理解: question_asker_intent_understanding, question_body_critical, ...
├── 問題類型: question_type_compare, question_type_definition, question_type_entity, ...
└── 品質評估: question_well_written, question_interestingness_others, ...

Answer 相關 (9 個):
├── 回答品質: answer_helpful, answer_plausible, answer_relevance, ...
└── 回答類型: answer_type_instructions, answer_type_procedure, ...
```

### 1.3 核心挑戰

| 挑戰 | 描述 | 影響 |
|------|------|------|
| **小數據集** | 僅 ~6,000 筆訓練數據 | 容易過擬合，需要強正則化 |
| **標籤不平衡** | 某些標籤極度稀疏（如 `question_type_spelling`） | 難以學習稀疏標籤的分布 |
| **排名評估** | Spearman 只關心排名，不關心絕對值 | 需要針對排名優化的後處理 |
| **群組相關性** | 同一問題可能有多個答案 | 需要 GroupKFold 防止數據洩漏 |

---

## 2. 當前解決方案

### 2.1 模型架構

```
┌─────────────────────────────────────────────────────────────┐
│                    QuestDebertaModel                        │
├─────────────────────────────────────────────────────────────┤
│  Input: [CLS] question_title [SEP] question_body [SEP]      │
│         answer [SEP]                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           DeBERTa-v3-base/large                      │   │
│  │           (output_hidden_states=True)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Weighted Layer Pooling                       │   │
│  │   [CLS]₀, [CLS]₁, ..., [CLS]₁₂ → Weighted Sum       │   │
│  │   weights = softmax(learnable_params)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Multi-Sample Dropout (5x)                    │   │
│  │   dropout₁(x), dropout₂(x), ..., dropout₅(x)        │   │
│  │   → Average all outputs                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Linear(hidden_size, 30) + Sigmoid            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: 30 probability scores ∈ [0, 1]                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 關鍵技術細節

#### A. Weighted Layer Pooling
自動學習每個 Transformer 層的權重，對所有層的 `[CLS]` token 進行加權求和。初始化時偏向最後幾層，但訓練過程中會自動調整以適應任務需求。

**優點**: 自動發現哪些層的表示最適合此任務

#### B. Multi-Sample Dropout (5x)
使用 5 個不同的 dropout 層對相同輸入進行多次前向傳播，最後取平均作為最終輸出。

**優點**: 
- 訓練時等效於 5 個模型的 ensemble
- 增加正則化，減少過擬合
- 推理時保持 dropout 效果的平均

#### C. 差異學習率 (Differential Learning Rate)
Backbone 使用較小學習率（1e-5），Classification Head 使用較大學習率（5e-5）。

**原因**: 預訓練的 backbone 已經學到良好表示，只需微調；新初始化的 head 需要較快學習

### 2.3 訓練策略

| 策略 | 配置 | 說明 |
|------|------|------|
| **交叉驗證** | GroupKFold (5-fold) | 按 `question_title` 分組，防止同問題洩漏 |
| **損失函數** | BCEWithLogitsLoss | 數值穩定性優於 BCELoss |
| **優化器** | AdamW (weight_decay=0.01) | 帶權重衰減的 Adam |
| **學習率調度** | Linear warmup + decay | 10% warmup steps |
| **梯度裁剪** | max_norm=1.0 | 防止梯度爆炸 |
| **早停** | patience=3 | 防止過擬合 |

### 2.4 後處理：Distribution Matching

這是**冠軍方案**的核心技巧，針對 Spearman 排名相關性優化。

**工作原理**:
1. 按預測值對測試樣本排序
2. 根據訓練集中每個值的比例，在測試集中分配相應數量的樣本
3. 相同比例區間的樣本獲得相同的分數

**應用策略**:
- 當前應用於 **17 個欄位**（稀疏分布或離散值的欄位）
- 包括：`question_conversational`, `question_type_compare`, `question_type_definition`, `question_type_entity`, `question_has_commonly_accepted_answer`, 等

**效果**: 冠軍報告 +0.027~0.030 的提升

### 2.5 推理流程

```
┌─────────────────────────────────────────────────────────────┐
│                    5-Fold Ensemble                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Model₁ ──→ Predictions₁ ─┐                               │
│   Model₂ ──→ Predictions₂ ─┼──→ Average ──→ Ensemble Pred  │
│   Model₃ ──→ Predictions₃ ─┤                               │
│   Model₄ ──→ Predictions₄ ─┤                               │
│   Model₅ ──→ Predictions₅ ─┘                               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Ensemble Pred ──→ Distribution Matching ──→ Final Pred   │
│                     (17 selected columns)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 未來優化方向

### 🎯 優化策略總覽

基於「**最小力氣，最大收益**」原則，以下按照**實現難度**和**預期效果**分類：

```
立即可做 (1-2 小時):
├── ✅ EMA (Exponential Moving Average)
├── ✅ Layer-wise LR Decay
└── ✅ Ranking Loss (作為輔助 loss)

半天工作量:
├── 🔄 AWP (Adversarial Weight Perturbation)
├── 🔄 ModernBERT 作為 backbone
└── 🔄 優化 Distribution Matching 欄位

可選實驗 (需要驗證):
├── 📊 結構化輸入格式
├── 📊 雙塔架構
└── 📊 模型融合
```

---

### 3.1 模型層面

#### A. Backbone 模型選擇

**🔍 架構適用性分析**

| 架構類型 | 代表模型 | 適用性 | 說明 |
|----------|----------|--------|------|
| **Encoder-only** | DeBERTa-v3, ModernBERT | ⭐⭐⭐ 最適合 | 專為理解任務設計 |
| **Encoder-Decoder** | T5, BART, Flan-T5 | ⭐⭐ 可用 | 生成能力在此任務浪費 |
| **Decoder-only** | Llama, Mistral, Qwen | ⭐ 不建議 | 回歸任務需額外設計 |

**🔥 2024-2025 推薦模型**

| 模型 | 參數量 | Context Length | 預期效果 | 備註 |
|------|--------|----------------|----------|------|
| `deberta-v3-large` | 304M | 512 | +0.01~0.02 | ✅ 當前使用 |
| `answerdotai/ModernBERT-large` | 395M | 8192 | +0.015~0.025 | 🔥 2024 新架構 |
| `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | 1.5B | 32K | +0.02~0.03 | Embedding 專用 |
| `deberta-v2-xlarge` | 900M | 512 | +0.02~0.03 | 需要 A100 GPU |

**⚠️ 重要考量**

這個競賽有特殊性：
- 測試集僅 **~476 筆**
- 評估指標是 **Spearman correlation**（排名相關）
- Distribution matching 會重新分配排名

這意味著：
1. **更大模型不一定更好** — 小數據 + 排名評估 = 容易過擬合
2. **模型差異可能被後處理抹平** — 後處理對排名的影響很大

💡 **建議優先級**：
```
1. ModernBERT-large (新架構 + 更長 context)
2. 保持 deberta-v3-large + 優化訓練技巧
3. Qwen2-1.5B (如果資源充足)
```

#### B. 模型融合 (Model Blending)

融合不同架構模型的預測結果，通過加權平均提升穩定性。候選模型包括 DeBERTa-v3-base/large、RoBERTa-large、ALBERT-xxlarge-v2 等。權重可通過驗證集優化確定。

**預期效果**: +0.01~0.02

#### C. 知識蒸餾 (Knowledge Distillation)

使用大模型（如 deberta-v2-xxlarge）作為 Teacher 生成軟標籤，訓練小模型（如 deberta-v3-base）作為 Student。可在保持精度的同時加快推理速度。

**適用場景**: 需要在推理速度和精度之間取得平衡時

### 3.2 數據層面

#### ⚠️ LLM 數據增強的風險分析（強烈不推薦）

##### 核心問題：分佈不匹配 (Distribution Shift)

本任務有特殊性，使其對 LLM 數據增強極其敏感：

| 特性 | 影響 |
|------|------|
| **標籤來自人工標註** | 反映「人類評審者」主觀判斷，LLM 難以模擬 |
| **評估指標是 Spearman** | 排名敏感，分佈偏移直接影響排名相關性 |
| **測試集極小 (~476)** | 微小分佈偏移被放大，性能下降明顯 |
| **標籤是連續值 (0-1)** | LLM 難以準確模擬人類評分的微妙差異 |

##### 三大風險

**1. 標籤分佈偏移**

```
原始訓練集標籤分佈:
  question_well_written: mean=0.72, std=0.18, 呈現左偏分佈

LLM 生成的數據:
  ├── LLM 傾向生成「高品質」文本 → 標籤偏高
  ├── LLM 缺乏人類評審的「嚴格度校準」
  └── 結果: 生成數據的標籤分佈 ≠ 真實分佈

實際後果:
  └── 模型學到的「什麼是 0.7 分的問題」與人類評審不一致
      在 Spearman 排名評估中直接降低分數
```

**2. 語言風格偏移**

| 維度 | 原始數據（Stack Exchange） | LLM 生成數據 |
|------|--------------------------|-----------|
| 口語化程度 | 混合（有錯字、口語） | 過於正式、流暢 |
| 領域術語 | 真實用戶的不完美表達 | 標準化的技術用語 |
| 問題結構 | 不規則、有冗餘 | 結構清晰、簡潔 |

**3. 稀疏標籤無法模擬**

```
question_type_spelling (拼寫問題):
  ├── 訓練集中 ~95% 的值是 0.0
  ├── 只有 ~5% 有非零值
  └── LLM 如何知道什麼樣的問題「看起來像」拼寫問題？

question_not_really_a_question:
  └── 這需要人類判斷「這不是一個真正的問題」
      LLM 很難準確模擬這種邊界判斷
```

##### 最嚴重風險：Distribution Matching 失效

當前的後處理策略依賴訓練集分佈作為參考：

```
訓練集分佈 = 原始分佈 + LLM 分佈（偏移）← 污染
                    ↓
        後處理的「參考分佈」被污染
                    ↓
        測試集被映射到錯誤的分佈
                    ↓
              排名錯誤 → CV 分數下降
```

由於當前方案依賴 Distribution Matching 帶來的 +0.027~0.030 提升，任何對訓練集分佈的污染都會抵消這部分收益。

##### 📊 定量分析

根據經驗法則，什麼時候數據增強有效：

| 條件 | 本任務情況 | 增強效果 |
|------|-----------|---------|
| 數據量 < 1000 | ❌ 6,079 筆（相對充足） | 邊際收益低 |
| 標籤是客觀的 | ❌ 主觀評分（人工標註） | 難以準確生成 |
| 評估指標是準確率 | ❌ Spearman（排名敏感） | 分佈敏感度高 |
| 測試集夠大 | ❌ 僅 476 筆 | 小偏移被放大 |

**預期結果**: LLM 增強帶來的分佈偏移可能抵消 0.005~0.020 的性能，遠超其可能帶來的收益

##### 結論

❌ **不推薦使用 LLM 數據增強**

理由：
- 風險 > 收益（分佈污染成本 > 數據增加收益）
- 破壞現有的後處理優化
- 這個任務的瓶頸是「排名優化」，不是「數據量」

---

#### A. 改進 Input Format

**當前格式**: `[CLS] question_title [SEP] question_body [SEP] answer [SEP]`

**潛在問題**:
1. 沒有明確區分 Question vs Answer 的語義邊界
2. 未利用 `category` 和 `host` 等元信息

**改進方案**:
- **方案 1 - 結構化提示**: 使用自然語言標記（如 "Question Title:", "Answer:"）明確標示各部分
- **方案 2 - 雙塔架構**: 分別編碼 Question 和 Answer，再通過 Cross-attention 融合

**⚠️ 注意**: 這是 6 年前的競賽，簡單方法可能已被充分探索。改變 input format 的預期收益有限（+0.003~0.008）。

#### B. 安全的替代方案（推薦）

與其嘗試風險高的 LLM 增強，以下方案更安全且已被證實有效：

##### 1️⃣ Mixup（Embedding 層面）

```python
def mixup_embeddings(embed1, label1, embed2, label2, alpha=0.2):
    """在特徵空間混合，不生成新文本"""
    lam = np.random.beta(alpha, alpha)
    mixed_embed = lam * embed1 + (1 - lam) * embed2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_embed, mixed_label
```

**優點**: 
- 保持在原始數據的「凸包」內，不會產生 OOD 數據
- 不污染訓練集分佈
- 已在當前架構中易於集成

**預期效果**: +0.003~0.008

##### 2️⃣ 多樣本 Dropout（已實現）

當前模型已使用 **Multi-Sample Dropout (5x)**，訓練時等效於 5 個模型的 Ensemble。這本質上是最安全的數據增強形式。

##### 3️⃣ 交叉驗證最大化利用

5-Fold CV 讓每個樣本都被用於驗證，已達成「數據利用最優化」。

**💡 結論**: 對於本任務，訓練技巧優化（EMA, LLRD, AWP）的收益遠高於數據增強

### 3.3 訓練策略優化

#### A. 🔥 損失函數優化（高優先級）

**核心問題**: BCELoss 優化「預測值接近真實值」，但 Spearman 只關心「排名順序」

| Loss 類型 | 優化目標 | 與 Spearman 對齊度 | 推薦度 |
|----------|----------|-------------------|--------|
| BCELoss | 數值接近 | ❌ 不對齊 | 當前使用 |
| **Pairwise Ranking Loss** | 排名順序 | ✅ 完全對齊 | ⭐⭐⭐⭐⭐ |
| **Soft Spearman Loss** | 排名相關性 | ✅ 完全對齊 | ⭐⭐⭐⭐ |
| Combined Loss | 數值 + 排名 | ⚠️ 部分對齊 | ⭐⭐⭐⭐ |

**💡 推薦方案**

**1. Pairwise Ranking Loss**
- 對每對樣本 (i, j)，如果 target[i] > target[j]，則希望 pred[i] > pred[j]
- 使用 Margin Ranking Loss 確保正確的相對排序
- 只考慮 target 有明顯差異的樣本對（閾值 > 0.05）

**2. Soft Spearman Loss**
- 使用 soft ranking（sigmoid 近似）計算可微分的排名
- 計算預測排名和真實排名的 Pearson correlation
- Loss = 1 - mean correlation

**3. Combined Loss（推薦）**
- 結合 BCE、Ranking 和 Spearman Loss
- 權重建議：α=0.5 (BCE), β=0.3 (Ranking), 0.2 (Spearman)

**⚠️ 注意事項**:
- Ranking loss 計算複雜度是 O(N²)，batch size 建議 ≤16
- 需要調整 loss 權重以平衡各項
- 預期提升：+0.005~0.015

#### B. 🔥 現代 Kaggle 優化技巧（高性價比）

**1. EMA (Exponential Moving Average)** — 立即可做 ✅

維護模型權重的指數移動平均，推理時使用平滑後的權重。

**實現要點**:
- 訓練時：每個 step 後更新 shadow weights (decay=0.9995)
- 推理時：臨時替換為 shadow weights，預測後恢復

**預期效果**: +0.002~0.008

**2. Layer-wise LR Decay (LLRD)** — 立即可做 ✅

為 Transformer 的不同層設置遞減的學習率，越靠近輸入層學習率越小。

**實現要點**:
- Embedding 層：最小學習率（base_lr × decay^n_layers）
- Encoder 層：逐層遞增（base_lr × decay^(n_layers - layer_i)）
- Head 層：最大學習率（base_lr × 5）
- 推薦 decay=0.9

**預期效果**: +0.003~0.008

**3. AWP (Adversarial Weight Perturbation)** — 半天工作量 🔄

對模型權重添加對抗擾動以提高泛化能力（2020年後 Kaggle NLP 比賽的標準技巧）。

**實現要點**:
- 在正常訓練步驟後，沿梯度方向擾動權重
- 使用擾動後的模型計算對抗損失並反向傳播
- 恢復原始權重並更新
- 建議每 2-4 個 step 執行一次，避免計算開銷過大

**預期效果**: +0.005~0.015

#### C. 學習率調度

推薦使用 **Cosine Annealing with Warm Restarts**，每個 epoch 重啟學習率並逐漸擴大週期（T_mult=2），最小學習率設為 1e-7。

### 3.4 後處理優化

#### A. 優化 Distribution Matching 欄位選擇（高優先級）

**當前問題**: 固定應用於 17 個欄位，但不是所有欄位都受益

**改進方案**: 通過 OOF 預測自動選擇最佳欄位

**實現步驟**:
1. 對每個欄位，計算應用 DM 前後的 CV Spearman score
2. 只保留有明顯提升（improvement > 0.001）的欄位
3. 按提升幅度排序，選擇 Top-K 欄位

**預期效果**: 當前固定 17 個欄位 → 自動選擇最優欄位，+0.003~0.008

#### B. 測試時增強 (TTA) - 可選

對同一輸入生成多個變體（通過不同隨機種子或 dropout），取預測平均值。

**注意**: 當前模型已有 multi-sample dropout (5x)，TTA 的額外收益可能有限（+0.001~0.003）

---

### 3.5 📊 優化方向總結與優先級

#### 🎯 立即可做（1-2 小時，高收益）

| 方法 | 預期提升 | 實現難度 | 優先級 |
|------|----------|----------|--------|
| **EMA** | +0.002~0.008 | ⭐ 很低 | 🔴 最高 |
| **Layer-wise LR Decay** | +0.003~0.008 | ⭐ 很低 | 🔴 最高 |
| **Ranking Loss (輔助)** | +0.005~0.012 | ⭐⭐ 低 | 🔴 最高 |
| **優化 DM 欄位選擇** | +0.003~0.008 | ⭐ 很低 | 🔴 高 |

**實施順序**:
```bash
1. 添加 EMA (30 分鐘)
2. 添加 Layer-wise LR Decay (30 分鐘)
3. 實驗 Ranking Loss (1 小時)
4. 優化 Distribution Matching 欄位 (30 分鐘)
```

#### 🔧 半天工作量（高性價比）

| 方法 | 預期提升 | 實現難度 | 優先級 |
|------|----------|----------|--------|
| **AWP 對抗訓練** | +0.005~0.015 | ⭐⭐ 中 | 🟡 高 |
| **ModernBERT-large** | +0.015~0.025 | ⭐⭐ 中 | 🟡 高 |
| **Combined Loss** | +0.008~0.015 | ⭐⭐ 中 | 🟡 中 |

#### 📦 需要驗證（1-2 天）

| 方法 | 預期提升 | 實現難度 | 優先級 |
|------|----------|----------|--------|
| **模型融合 (2-3 models)** | +0.01~0.02 | ⭐⭐⭐ 高 | 🟢 中 |
| **結構化輸入/雙塔** | +0.003~0.008 | ⭐⭐⭐ 高 | 🟢 低 |
| **Pseudo labeling** | +0.005~0.015 | ⭐⭐⭐ 高 | 🟢 低 |

#### ❌ 不推薦（低性價比或高風險）

| 方法 | 原因 |
|------|------|
| **LLM 數據增強** | 🔴 **高風險** — 分佈偏移將污染訓練集，破壞 Distribution Matching 效果，預期負收益 -0.005~-0.020 |
| **Back Translation** | 計算成本高，效果有限（+0.002~0.005） |
| **Cutout (Token 遮蔽)** | 可能丟失關鍵信息，不適合排名任務 |
| 更大模型（>2B）| 小測試集容易過擬合，邊際收益遞減 |
| 複雜 Ensemble（>5 models）| 工程量大，收益不如優化單模型 |

---

### 3.6 🚀 推薦實施路線圖

```
第 1 週：快速提升
├── Day 1: 添加 EMA + Layer-wise LR Decay
├── Day 2: 實驗 Ranking Loss
├── Day 3: 優化 Distribution Matching
└── Day 4: 驗證提升，提交 Kaggle

第 2 週：進階優化
├── Day 1-2: 實現 AWP 對抗訓練
├── Day 3-4: 測試 ModernBERT-large
└── Day 5: 模型融合實驗

第 3 週：精細調優
├── 超參數搜索（LR, dropout, warmup）
├── 後處理優化（DM 欄位微調）
└── 最終提交
```

**預期總提升**: 
- 保守估計：+0.015~0.030
- 樂觀估計：+0.025~0.045

---

## 附錄：快速參考

### ⚠️ 關鍵警告

**不要使用 LLM 數據增強！**

本任務對分佈偏移極其敏感，LLM 生成的數據會：
1. 改變訓練集標籤分佈（污染）
2. 破壞 Distribution Matching 後處理（預期 -0.005~-0.020）
3. 提升收益遠小於成本

**替代方案**：使用 EMA、Layer-wise LR Decay、AWP 等訓練技巧，風險更低、收益更高。

---

### 訓練命令

```bash
# 快速迭代（單 fold）
python src/train_and_inference_deberta.py --mode train --n_folds 1

# 完整訓練（5 fold）
python src/train_and_inference_deberta.py --mode train --n_folds 5

# 使用更大模型
python src/train_and_inference_deberta.py --mode both --model_name microsoft/deberta-v3-large

# 僅推理（不重新訓練）
python src/train_and_inference_deberta.py --mode inference
```

### 上傳模型到 Kaggle

```bash
# 上傳 5-fold 模型
./upload_models.sh
```

### 關鍵超參數

```python
Config:
    model_name = "microsoft/deberta-v3-large"
    max_len = 512
    batch_size = 16
    lr = 1e-5          # backbone learning rate
    head_lr = 5e-5     # head learning rate
    epochs = 5
    n_folds = 5
```