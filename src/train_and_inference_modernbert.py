"""
Google Quest Q&A Labeling - Training & Inference with ModernBERT

This script implements a ModernBERT-based model for multi-label classification of Q&A pairs.

Overview:
- Training: Fine-tune ModernBERT-base on 30 Q&A quality labels
- Inference: Generate predictions on test data with post-processing
- Architecture: Weighted layer pooling + multi-sample dropout

ModernBERT Features:
- Rotary Positional Embeddings (RoPE) for long-context support
- Local-Global Alternating Attention for efficiency
- Native 8,192 token context length
- Pre-trained on 2 trillion tokens of English and code

Usage:
    # Training
    python train_and_inference_modernbert.py --mode train
    
    # Inference only
    python train_and_inference_modernbert.py --mode inference
    
    # Both training and inference
    python train_and_inference_modernbert.py --mode both
"""

import os
import gc
import json
import argparse
from datetime import datetime
from collections import Counter
from typing import List, Text, Optional, Callable, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Fix tokenizers forking warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Enable anomaly detection for debugging (can be disabled in production)
torch.autograd.set_detect_anomaly(False)

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    """Training and model configuration"""
    model_name = "answerdotai/ModernBERT-large"
    
    # ModernBERT supports up to 8192 tokens - use longer context for better understanding
    max_len = 2048  # Increased from 1024 to leverage ModernBERT's long context capability
    batch_size = 8  # Reduced batch size to accommodate longer sequences (adjust based on GPU memory)
    accum_iter = 2  # Increased accumulation to maintain effective batch size
    
    # Learning rates - ModernBERT may need slightly different tuning
    lr = 2e-5               # Encoder learning rate
    head_lr = 1e-4          # Classification head learning rate
    
    # Training configuration
    epochs = 3              
    n_folds = 1
    validation_split = 0.2  # Only used when n_folds=1 (fast iteration mode)
    seed = 42
    num_workers = 8         # For data loading
    
    # Loss configuration (Combined Ranking Loss)
    ranking_loss_weight = 0.6    # Weight for pairwise ranking loss
    spearman_loss_weight = 0.4   # Weight for soft spearman loss
    ranking_margin = 0.1         # Margin for ranking loss
    ranking_threshold = 0.05     # Only consider pairs with target difference > threshold
    spearman_temperature = 1.0   # Temperature for soft ranking
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    sample_submission_csv = "data/sample_submission.csv"
    output_dir = "models/modernbert_large"  # Will be appended with timestamp in train_loop()
    
    # Grouping strategy: Use question_title to prevent data leakage
    # (Same question can have multiple answers; we group them together)
    grouping_column = 'question_title'
    
    target_cols = [
        'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
        'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
        'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
        'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice',
        'question_type_compare', 'question_type_consequence', 'question_type_definition',
        'question_type_entity', 'question_type_instructions', 'question_type_procedure',
        'question_type_reason_explanation', 'question_type_spelling', 'question_well_written',
        'answer_helpful', 'answer_level_of_information', 'answer_plausible', 'answer_relevance',
        'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure',
        'answer_type_reason_explanation', 'answer_well_written'
    ]


def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility


# ==========================================
# 2. Custom Loss Functions (Optimized for Spearman Correlation)
# ==========================================
class PairwiseRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss for optimizing Spearman correlation.
    
    For each pair (i, j) where target[i] > target[j], we want pred[i] > pred[j].
    Uses margin ranking loss to enforce correct relative ordering.
    
    This directly optimizes for ranking, which is what Spearman correlation measures.
    """
    
    def __init__(self, margin=0.1, threshold=0.05):
        """
        Args:
            margin: Minimum difference required between pred[i] and pred[j]
            threshold: Only consider pairs where |target[i] - target[j]| > threshold
        """
        super().__init__()
        self.margin = margin
        self.threshold = threshold
    
    def forward(self, preds, targets):
        """
        Args:
            preds: (batch_size, num_labels) - model predictions (logits or sigmoid outputs)
            targets: (batch_size, num_labels) - ground truth labels
            
        Returns:
            Scalar loss value
        """
        batch_size, num_labels = preds.shape
        device = preds.device
        
        # Apply sigmoid to get probabilities if needed
        preds = torch.sigmoid(preds)
        
        total_loss = torch.tensor(0.0, device=device)
        num_valid_pairs = 0
        
        # Process each label independently
        for label_idx in range(num_labels):
            pred_col = preds[:, label_idx]
            target_col = targets[:, label_idx]
            
            # Create all pairwise comparisons
            pred_diff = pred_col.unsqueeze(0) - pred_col.unsqueeze(1)  # (batch, batch)
            target_diff = target_col.unsqueeze(0) - target_col.unsqueeze(1)  # (batch, batch)
            
            # Only consider pairs where target difference is significant
            valid_mask = torch.abs(target_diff) > self.threshold
            
            if valid_mask.sum() == 0:
                continue
            
            # Sign of target difference: +1 if i should rank higher than j, -1 otherwise
            target_sign = torch.sign(target_diff)
            
            # Margin ranking loss: max(0, -sign * (pred_i - pred_j) + margin)
            # We want pred_i > pred_j when target_i > target_j
            loss = torch.clamp(self.margin - target_sign * pred_diff, min=0.0)
            
            # Apply mask and average
            loss = (loss * valid_mask.float()).sum()
            total_loss += loss
            num_valid_pairs += valid_mask.sum()
        
        if num_valid_pairs > 0:
            return total_loss / num_valid_pairs
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class SoftSpearmanLoss(nn.Module):
    """
    Differentiable approximation of Spearman correlation loss.
    
    Uses soft ranking (sigmoid-based) to compute differentiable ranks,
    then computes Pearson correlation between soft ranks.
    
    Loss = 1 - mean(correlation across labels)
    """
    
    def __init__(self, temperature=1.0, eps=1e-8):
        """
        Args:
            temperature: Controls sharpness of soft ranking (lower = sharper)
            eps: Small value for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def soft_rank(self, x):
        """
        Compute soft (differentiable) ranks using sigmoid approximation.
        
        For each element x[i], its soft rank is approximately:
        rank[i] = sum_j sigmoid((x[i] - x[j]) / temperature)
        
        This gives a differentiable approximation to the rank of each element.
        
        Args:
            x: (batch_size,) tensor
            
        Returns:
            (batch_size,) tensor of soft ranks
        """
        batch_size = x.shape[0]
        
        # Compute pairwise differences: diff[i, j] = x[i] - x[j]
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # (batch_size, batch_size)
        
        # Apply sigmoid to get soft comparison
        # sigmoid((x[i] - x[j]) / temp) ≈ 1 if x[i] > x[j], ≈ 0 if x[i] < x[j]
        soft_compare = torch.sigmoid(diff / self.temperature)
        
        # Sum along columns to get soft rank
        # Higher values get higher ranks
        soft_ranks = soft_compare.sum(dim=1)  # (batch_size,)
        
        return soft_ranks
    
    def pearson_correlation(self, x, y):
        """
        Compute Pearson correlation between two tensors.
        
        Args:
            x, y: (batch_size,) tensors
            
        Returns:
            Scalar correlation value
        """
        # Center the variables
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        
        # Compute correlation
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum() + self.eps)
        
        return numerator / denominator
    
    def forward(self, preds, targets):
        """
        Args:
            preds: (batch_size, num_labels) - model predictions
            targets: (batch_size, num_labels) - ground truth labels
            
        Returns:
            Scalar loss value (1 - mean correlation)
        """
        batch_size, num_labels = preds.shape
        device = preds.device
        
        # Apply sigmoid to get probabilities
        preds = torch.sigmoid(preds)
        
        correlations = []
        
        for label_idx in range(num_labels):
            pred_col = preds[:, label_idx]
            target_col = targets[:, label_idx]
            
            # Skip if all values are the same
            if torch.std(pred_col) < 1e-6 or torch.std(target_col) < 1e-6:
                continue
            
            # Compute soft ranks
            pred_ranks = self.soft_rank(pred_col)
            target_ranks = self.soft_rank(target_col)
            
            # Compute correlation between ranks
            corr = self.pearson_correlation(pred_ranks, target_ranks)
            correlations.append(corr)
        
        if len(correlations) > 0:
            mean_corr = torch.stack(correlations).mean()
            # Return 1 - correlation as loss (maximize correlation = minimize loss)
            return 1.0 - mean_corr
        else:
            return torch.tensor(1.0, device=device, requires_grad=True)


class CombinedRankingLoss(nn.Module):
    """
    Combined loss function for optimizing Spearman correlation.
    
    Combines:
    1. Pairwise Ranking Loss - Directly optimizes pairwise ordering
    2. Soft Spearman Loss - Differentiable approximation of Spearman correlation
    
    Both losses are designed to optimize for ranking, which is what
    Spearman correlation measures.
    """
    
    def __init__(
        self,
        ranking_weight=0.6,
        spearman_weight=0.4,
        ranking_margin=0.1,
        ranking_threshold=0.05,
        spearman_temperature=1.0
    ):
        """
        Args:
            ranking_weight: Weight for pairwise ranking loss
            spearman_weight: Weight for soft spearman loss
            ranking_margin: Margin for ranking loss
            ranking_threshold: Threshold for significant pairs in ranking loss
            spearman_temperature: Temperature for soft ranking
        """
        super().__init__()
        
        self.ranking_weight = ranking_weight
        self.spearman_weight = spearman_weight
        
        self.ranking_loss = PairwiseRankingLoss(
            margin=ranking_margin,
            threshold=ranking_threshold
        )
        self.spearman_loss = SoftSpearmanLoss(
            temperature=spearman_temperature
        )
    
    def forward(self, preds, targets):
        """
        Args:
            preds: (batch_size, num_labels) - model predictions (logits)
            targets: (batch_size, num_labels) - ground truth labels
            
        Returns:
            Dictionary with total loss and individual components
        """
        # Compute individual losses
        ranking_loss = self.ranking_loss(preds, targets)
        spearman_loss = self.spearman_loss(preds, targets)
        
        # Combine losses
        total_loss = (
            self.ranking_weight * ranking_loss +
            self.spearman_weight * spearman_loss
        )
        
        return {
            'loss': total_loss,
            'ranking_loss': ranking_loss.detach(),
            'spearman_loss': spearman_loss.detach()
        }


# ==========================================
# 3. Dataset Class (Optimized for ModernBERT)
# ==========================================

# Target column definitions
QUESTION_TARGETS = [
    "question_asker_intent_understanding", "question_body_critical", "question_conversational",
    "question_expect_short_answer", "question_fact_seeking", "question_has_commonly_accepted_answer",
    "question_interestingness_others", "question_interestingness_self", "question_multi_intent",
    "question_not_really_a_question", "question_opinion_seeking", "question_type_choice",
    "question_type_compare", "question_type_consequence", "question_type_definition",
    "question_type_entity", "question_type_instructions", "question_type_procedure",
    "question_type_reason_explanation", "question_type_spelling", "question_well_written",
]

ANSWER_TARGETS = [
    "answer_helpful", "answer_level_of_information", "answer_plausible", "answer_relevance",
    "answer_satisfaction", "answer_type_instructions", "answer_type_procedure",
    "answer_type_reason_explanation", "answer_well_written",
]

ALL_TARGETS = QUESTION_TARGETS + ANSWER_TARGETS


class QuestDatasetModernBERT(Dataset):
    """
    Optimized Dataset for ModernBERT.
    
    Key differences from standard BERT dataset:
    1. No token_type_ids - ModernBERT doesn't use segment embeddings
    2. Supports longer sequences (up to 8192 tokens)
    3. Uses explicit segment markers instead of token_type_ids
    4. More aggressive text preservation with longer context
    
    Args:
        data_df: DataFrame with question/answer data
        tokenizer: ModernBERT tokenizer
        max_length: Maximum sequence length (default 512, can go up to 8192)
        target_cols: Target columns to use (default: all targets)
        answer_ratio: Ratio of max_length reserved for answer (default 0.4)
        title_ratio: Ratio of question portion for title (default 0.3)
        use_segment_markers: If True, add [Q] and [A] markers (default True)
        use_title: Include question title
        use_body: Include question body
        use_answer: Include answer
        title_col, body_col, answer_col: Column names in dataframe
        title_transform, body_transform, answer_transform: Optional text transforms
        mode: "train" or "test"
    """
    
    # Segment markers - these will be added as regular text since ModernBERT
    # doesn't have special tokens for Q/A separation
    QUESTION_MARKER = "[QUESTION]"
    ANSWER_MARKER = "[ANSWER]"
    TITLE_MARKER = "[TITLE]"
    BODY_MARKER = "[BODY]"
    
    def __init__(
        self,
        data_df,
        tokenizer,
        max_length: int = 2048,  # Increased default for ModernBERT
        target_cols: Optional[List[str]] = None,
        answer_ratio: float = 0.4,
        title_ratio: float = 0.3,
        use_segment_markers: bool = True,
        use_detailed_markers: bool = False,
        use_title: bool = True,
        use_body: bool = True,
        use_answer: bool = True,
        title_col: str = "question_title",
        body_col: str = "question_body",
        answer_col: str = "answer",
        title_transform: Optional[Callable] = None,
        body_transform: Optional[Callable] = None,
        answer_transform: Optional[Callable] = None,
        mode: str = "train",
        preserve_full_text: bool = True,  # NEW: Try to preserve complete text when possible
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_ratio = answer_ratio
        self.title_ratio = title_ratio
        self.use_segment_markers = use_segment_markers
        self.use_detailed_markers = use_detailed_markers
        self.mode = mode
        self.preserve_full_text = preserve_full_text  # NEW
        
        # Resolve target columns
        if target_cols is None:
            self.target_cols = ALL_TARGETS if mode != "test" else None
        else:
            self.target_cols = target_cols
            
        # Load targets if in training mode
        if self.target_cols is not None and mode != "test":
            self.targets = data_df[self.target_cols].values
        else:
            self.targets = None
            
        # Load text columns
        self.question_title = data_df[title_col].values if use_title else None
        self.question_body = data_df[body_col].values if use_body else None
        self.answer = data_df[answer_col].values if use_answer else None
        
        # Text transforms
        self.title_transform = title_transform
        self.body_transform = body_transform
        self.answer_transform = answer_transform
        
        # Pre-tokenize markers for efficiency
        self._cache_marker_tokens()
        
    def _cache_marker_tokens(self):
        """Pre-tokenize segment markers"""
        if self.use_segment_markers:
            self.question_marker_ids = self.tokenizer.encode(
                self.QUESTION_MARKER, add_special_tokens=False
            )
            self.answer_marker_ids = self.tokenizer.encode(
                self.ANSWER_MARKER, add_special_tokens=False
            )
            if self.use_detailed_markers:
                self.title_marker_ids = self.tokenizer.encode(
                    self.TITLE_MARKER, add_special_tokens=False
                )
                self.body_marker_ids = self.tokenizer.encode(
                    self.BODY_MARKER, add_special_tokens=False
                )
            else:
                self.title_marker_ids = []
                self.body_marker_ids = []
        else:
            self.question_marker_ids = []
            self.answer_marker_ids = []
            self.title_marker_ids = []
            self.body_marker_ids = []
    
    def __len__(self) -> int:
        if self.answer is not None:
            return len(self.answer)
        elif self.question_title is not None:
            return len(self.question_title)
        elif self.question_body is not None:
            return len(self.question_body)
        else:
            return 0
    
    def _apply_transform(self, text: Optional[str], transform: Optional[Callable], idx: int) -> Optional[str]:
        """Apply optional text transformation"""
        if transform is not None and text is not None:
            return transform(text, idx=idx)
        return text
    
    def _get_text(self, index: int) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get and optionally transform text for given index"""
        title = self.question_title[index] if self.question_title is not None else None
        body = self.question_body[index] if self.question_body is not None else None
        answer = self.answer[index] if self.answer is not None else None
        
        title = self._apply_transform(title, self.title_transform, index)
        body = self._apply_transform(body, self.body_transform, index)
        answer = self._apply_transform(answer, self.answer_transform, index)
        
        return title, body, answer
    
    @staticmethod
    def _balance_segments_adaptive(
        first_tokens: List[int],
        second_tokens: List[int],
        second_ratio: float,
        max_length: int,
        preserve_full: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Adaptively balance two segments to fit within max_length.
        
        NEW: When preserve_full=True, tries to keep complete text if it fits,
        only truncating when necessary. This better utilizes ModernBERT's long context.
        
        Args:
            first_tokens: Token IDs for first segment
            second_tokens: Token IDs for second segment
            second_ratio: Preferred ratio for second segment
            max_length: Maximum total length
            preserve_full: If True, prioritize keeping complete text
            
        Returns:
            Truncated (first_tokens, second_tokens) tuples
        """
        first_len = len(first_tokens)
        second_len = len(second_tokens)
        total_len = first_len + second_len
        
        # If everything fits, keep it all (leverage long context!)
        if total_len <= max_length:
            return first_tokens, second_tokens
        
        if preserve_full:
            # Adaptive truncation: truncate the longer segment first
            overflow = total_len - max_length
            
            # Calculate how much each segment exceeds its "fair share"
            first_budget = int((1 - second_ratio) * max_length)
            second_budget = int(second_ratio * max_length)
            
            first_excess = max(0, first_len - first_budget)
            second_excess = max(0, second_len - second_budget)
            
            if first_excess + second_excess > 0:
                # Distribute truncation proportionally to excess
                total_excess = first_excess + second_excess
                first_cut = int(overflow * first_excess / total_excess)
                second_cut = overflow - first_cut
            else:
                # Both within budget but still overflow - use ratio
                first_cut = int(overflow * (1 - second_ratio))
                second_cut = overflow - first_cut
            
            final_first = first_tokens[:max(1, first_len - first_cut)]
            final_second = second_tokens[:max(1, second_len - second_cut)]
        else:
            # Original fixed-ratio logic
            first_budget = int((1 - second_ratio) * max_length)
            second_budget = int(second_ratio * max_length)
            
            first_overflow = max(0, first_budget - first_len)
            second_overflow = max(0, second_budget - second_len)
            
            final_first = first_tokens[:min(first_len, first_budget + second_overflow)]
            final_second = second_tokens[:min(second_len, second_budget + first_overflow)]
        
        return final_first, final_second
    
    def _prepare_features(
        self, 
        title: Optional[str], 
        body: Optional[str], 
        answer: Optional[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Prepare input features for ModernBERT.
        
        Format: [CLS] [QUESTION] title [SEP] body [SEP] [ANSWER] answer [SEP]
        
        UPDATED: Uses adaptive balancing to better utilize long context window.
        When text is short enough, preserves everything. Only truncates when necessary.
        
        Returns:
            input_ids: Token IDs
            attention_mask: Attention mask (1 for real tokens, 0 for padding)
        """
        # Calculate budget accounting for special tokens and markers
        special_tokens_count = 3  # [CLS], [SEP] after question, [SEP] after answer
        marker_tokens_count = len(self.question_marker_ids) + len(self.answer_marker_ids)
        if self.use_detailed_markers:
            marker_tokens_count += len(self.title_marker_ids) + len(self.body_marker_ids)
        
        available_length = self.max_length - special_tokens_count - marker_tokens_count
        
        # ModernBERT's maximum position embedding limit
        # We use this as a safe truncation limit to avoid tokenizer warnings
        # while still allowing adaptive balancing within our desired max_length
        safe_max_tokens = 8192
        
        # Tokenize all text with a safe maximum to avoid tokenizer warnings
        # We'll do our own truncation in _balance_segments_adaptive
        title_tokens = self.tokenizer.encode(
            str(title) if title else "", 
            add_special_tokens=False,
            truncation=True,
            max_length=safe_max_tokens
        ) if title else []
        
        body_tokens = self.tokenizer.encode(
            str(body) if body else "", 
            add_special_tokens=False,
            truncation=True,
            max_length=safe_max_tokens
        ) if body else []
        
        answer_tokens = self.tokenizer.encode(
            str(answer) if answer else "", 
            add_special_tokens=False,
            truncation=True,
            max_length=safe_max_tokens
        ) if answer else []
        
        # Combine title + body as "question" segment
        # First balance title vs body within question budget
        question_budget = int((1 - self.answer_ratio) * available_length)
        title_tokens, body_tokens = self._balance_segments_adaptive(
            title_tokens, 
            body_tokens, 
            1 - self.title_ratio,  # body gets more space
            question_budget,
            preserve_full=self.preserve_full_text
        )
        
        # Now balance question (title+body) vs answer
        question_tokens = title_tokens + ([self.tokenizer.sep_token_id] if title_tokens and body_tokens else []) + body_tokens
        question_tokens, answer_tokens = self._balance_segments_adaptive(
            question_tokens,
            answer_tokens,
            self.answer_ratio,
            available_length,
            preserve_full=self.preserve_full_text
        )
        
        # Build input sequence
        input_ids = [self.tokenizer.cls_token_id]
        
        # Add question section
        if self.use_segment_markers:
            input_ids.extend(self.question_marker_ids)
        
        input_ids.extend(question_tokens)
        input_ids.append(self.tokenizer.sep_token_id)
        
        # Add answer section
        if answer_tokens:
            if self.use_segment_markers:
                input_ids.extend(self.answer_marker_ids)
            input_ids.extend(answer_tokens)
            input_ids.append(self.tokenizer.sep_token_id)
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        elif padding_length < 0:
            # Safety truncation (shouldn't happen with proper balancing)
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        return input_ids, attention_mask
    
    def __getitem__(self, index: int):
        title, body, answer = self._get_text(index)
        input_ids, attention_mask = self._prepare_features(title, body, answer)
        
        output = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }
        
        if self.targets is not None:
            labels = self.targets[index].copy()
            # Safety checks
            labels = np.clip(labels, 0.0, 1.0)
            labels = np.nan_to_num(labels, nan=0.5, posinf=1.0, neginf=0.0)
            output['labels'] = torch.tensor(labels, dtype=torch.float32)
        
        return output


# Backward compatibility alias
QuestDataset = QuestDatasetModernBERT


# ==========================================
# 4. Model Class
# ==========================================
class QuestModernBertModel(nn.Module):
    """ModernBERT model with weighted layer pooling and multi-sample dropout"""
    
    def __init__(self, model_name=Config.model_name, num_labels=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        
        # Load ModernBERT with Flash Attention if available
        try:
            self.model = AutoModel.from_pretrained(
                model_name, 
                config=self.config,
                attn_implementation="sdpa",
                dtype=torch.bfloat16
            )
            print("✓ Using PyTorch SDPA (Scaled Dot-Product Attention)")
        except Exception as e:
            print(f"⚠ PyTorch SDPA not available, using default attention: {e}")
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Weighted layer pooling with stable initialization
        n_weights = self.config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        weights_init.data[-1] = 0  # Give more weight to last layer initially
        self.layer_weights = nn.Parameter(weights_init)
        
        # Multi-sample dropout for regularization
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        # Removed Sigmoid for BCEWithLogitsLoss stability

    def forward(self, input_ids, attention_mask):
        # ModernBERT doesn't use token_type_ids
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states 
        
        # Stack [CLS] tokens from all layers
        # For ModernBERT, the first token is the [CLS] token
        cls_outputs = torch.stack([layer[:, 0, :] for layer in hidden_states], dim=1)
        
        # Convert to float32 for stability if using bfloat16
        cls_outputs = cls_outputs.float()
        
        # Weighted sum across layers
        weights = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1)
        weighted_cls = (weights * cls_outputs).sum(dim=1)
        
        # Multi-sample dropout averaging
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(weighted_cls)))
        avg_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        
        return avg_logits


# ==========================================
# 5. Utilities
# ==========================================
def compute_spearmanr(trues, preds):
    """Compute mean Spearman correlation across all labels"""
    scores = []
    for i in range(trues.shape[1]):
        # Handle edge cases where all values are the same
        if len(np.unique(trues[:, i])) == 1 or len(np.unique(preds[:, i])) == 1:
            # If all predictions or all true values are the same, skip this column
            continue
        
        corr = spearmanr(trues[:, i], preds[:, i]).correlation
        
        # Skip NaN correlations
        if not np.isnan(corr):
            scores.append(corr)
    
    # Return mean of valid scores, or 0 if no valid scores
    return np.mean(scores) if len(scores) > 0 else 0.0


def create_dataloaders(train_df, val_df, tokenizer, max_length=None):
    """Create train and validation dataloaders"""
    if max_length is None:
        max_length = Config.max_len
        
    train_dataset = QuestDatasetModernBERT(
        train_df, 
        tokenizer, 
        max_length=max_length,
        target_cols=list(Config.target_cols),
        mode="train"
    )
    val_dataset = QuestDatasetModernBERT(
        val_df, 
        tokenizer, 
        max_length=max_length,
        target_cols=list(Config.target_cols),
        mode="train"
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.batch_size, 
        shuffle=True, 
        num_workers=Config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.batch_size, 
        shuffle=False, 
        num_workers=Config.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.01):
    """Differential learning rate for backbone and head"""
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm"]
    optimizer_parameters = [
        # Backbone parameters with weight decay
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        # Backbone parameters without weight decay
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        # Head parameters
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, accum_iter, scaler=None):
    """Train one epoch with gradient accumulation and mixed precision (bfloat16)"""
    model.train()
    train_loss = 0
    ranking_loss_sum = 0
    spearman_loss_sum = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Use bfloat16 autocast without GradScaler (bfloat16 is more stable)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids, mask)
            # Ensure outputs are float32 for loss computation
            outputs = outputs.float()
            loss_dict = loss_fn(outputs, labels)
            loss = loss_dict['loss']
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠ Warning: NaN/Inf loss detected at step {step}, skipping batch")
            optimizer.zero_grad()
            continue
        
        # Gradient accumulation
        loss = loss / accum_iter
        loss.backward()

        if (step + 1) % accum_iter == 0:
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * accum_iter
        ranking_loss_sum += loss_dict['ranking_loss'].item()
        spearman_loss_sum += loss_dict['spearman_loss'].item()
        
        progress_bar.set_postfix({
            'loss': train_loss / (step + 1),
            'rank': ranking_loss_sum / (step + 1),
            'spear': spearman_loss_sum / (step + 1)
        })
    
    return train_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, device):
    """Validation loop"""
    model.eval()
    val_preds = []
    val_trues = []
    
    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, mask)
        # Apply sigmoid for metrics since we removed it from model
        outputs = torch.sigmoid(outputs.float())
        
        val_preds.append(outputs.cpu().numpy())
        val_trues.append(labels.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_trues = np.concatenate(val_trues)
    
    # Add safety checks for predictions
    val_preds = np.clip(val_preds, 0, 1)
    val_preds = np.nan_to_num(val_preds, nan=0.5, posinf=1.0, neginf=0.0)
    
    score = compute_spearmanr(val_trues, val_preds)
    return score, val_preds, val_trues


# ==========================================
# 6. Training Pipeline (K-Fold with Multi-GPU)
# ==========================================
def train_loop():
    """Main training pipeline with GroupKFold and multi-GPU support"""
    # Validate configuration
    if Config.n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {Config.n_folds}")
    if not (0 < Config.validation_split < 1):
        raise ValueError(f"validation_split must be in (0, 1), got {Config.validation_split}")
    if Config.grouping_column not in ['question_title', 'qa_id', None]:
        print(f"⚠ Warning: Unusual grouping_column '{Config.grouping_column}'")
    
    # Append timestamp to output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.output_dir = f"{Config.output_dir}_{timestamp}"
    
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(Config.train_csv)
    print(f"Total samples: {len(train_df):,}\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
    
    # Print loss configuration
    print("Loss Configuration:")
    print(f"  Pairwise Ranking Loss weight: {Config.ranking_loss_weight}")
    print(f"  Soft Spearman Loss weight: {Config.spearman_loss_weight}")
    print(f"  Ranking margin: {Config.ranking_margin}")
    print(f"  Ranking threshold: {Config.ranking_threshold}")
    print(f"  Spearman temperature: {Config.spearman_temperature}\n")
    
    # ========================================
    # Select fold strategy based on n_folds
    # ========================================
    if Config.n_folds == 1:
        # FAST ITERATION MODE: Single train/val split
        print("📍 FAST ITERATION MODE")
        print(f"Using single train/val split (validation_split={Config.validation_split})")
        if Config.grouping_column:
            print(f"Grouping by: {Config.grouping_column} (prevents same-question leakage)\n")
            gss = GroupShuffleSplit(
                n_splits=1, 
                test_size=Config.validation_split, 
                random_state=Config.seed
            )
            fold_splits = list(gss.split(
                train_df, 
                train_df[Config.target_cols], 
                groups=train_df[Config.grouping_column]
            ))
        else:
            print("No grouping applied\n")
            gss = GroupShuffleSplit(
                n_splits=1, 
                test_size=Config.validation_split, 
                random_state=Config.seed
            )
            fold_splits = list(gss.split(train_df, train_df[Config.target_cols]))
    else:
        # FINAL TRAINING MODE: K-Fold cross-validation
        print("📍 FINAL TRAINING MODE")
        print(f"Using {Config.n_folds}-Fold Cross Validation")
        if Config.grouping_column:
            print(f"Grouping by: {Config.grouping_column} (prevents same-question leakage)\n")
            gkf = GroupKFold(n_splits=Config.n_folds)
            fold_splits = list(gkf.split(
                train_df, 
                train_df[Config.target_cols], 
                groups=train_df[Config.grouping_column]
            ))
        else:
            print("No grouping applied\n")
            gkf = GroupKFold(n_splits=Config.n_folds)
            fold_splits = list(gkf.split(train_df, train_df[Config.target_cols]))
    
    # Store OOF predictions
    oof_preds = np.zeros((len(train_df), len(Config.target_cols)))
    
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\n{'='*50}")
        if Config.n_folds == 1:
            print(f"Training (Single Split)")
        else:
            print(f"Fold {fold+1}/{Config.n_folds}")
        print(f"{'='*50}")
        
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train samples: {len(train_data):,} | Val samples: {len(val_data):,}")

        # Initialize tokenizer and dataloaders
        tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        train_loader, val_loader = create_dataloaders(train_data, val_data, tokenizer)

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = QuestModernBertModel()
        model.to(device)

        # Differential Learning Rate
        optimizer_parameters = get_optimizer_params(model, encoder_lr=Config.lr, decoder_lr=Config.head_lr)
        
        optimizer = torch.optim.AdamW(optimizer_parameters, weight_decay=0.01)
        
        # Use Combined Ranking Loss (optimized for Spearman correlation)
        loss_fn = CombinedRankingLoss(
            ranking_weight=Config.ranking_loss_weight,
            spearman_weight=Config.spearman_loss_weight,
            ranking_margin=Config.ranking_margin,
            ranking_threshold=Config.ranking_threshold,
            spearman_temperature=Config.spearman_temperature
        )
        
        # Don't use GradScaler with bfloat16 (it's not needed and not supported)
        # bfloat16 has better numerical stability than float16
        scaler = None
        
        # Scheduler
        num_train_steps = int(len(train_loader) / Config.accum_iter * Config.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * num_train_steps), 
            num_training_steps=num_train_steps
        )

        best_score = -1.0
        best_model_path = os.path.join(Config.output_dir, f"best_model_fold{fold+1}.pth")
        patience_counter = 0
        patience = 3  # Early stopping patience
        
        # Track training history for this fold
        training_history = {
            'epochs': [],
            'train_loss': [],
            'val_score': []
        }

        for epoch in range(Config.epochs):
            # Training
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, Config.accum_iter, scaler)
            
            # Validation
            val_score, val_preds, _ = validate(model, val_loader, device)
            
            # Record history
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(float(train_loss))
            training_history['val_score'].append(float(val_score))
            
            print(f"Epoch {epoch+1:2d}/{Config.epochs} | Loss: {train_loss:.4f} | Val Score: {val_score:.5f}", end="")
            
            # Save best model
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)
                print(f" ✓ (New best)")
                print(f"  Model saved to: {best_model_path}")
                patience_counter = 0
            else:
                print()
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
        # Load best model for OOF
        model.load_state_dict(torch.load(best_model_path))
        
        _, val_preds, _ = validate(model, val_loader, device)
        oof_preds[val_idx] = val_preds
        
        if Config.n_folds == 1:
            print(f"\nBest validation score: {best_score:.5f}")
        else:
            print(f"\nFold {fold+1} best score: {best_score:.5f}")
        
        # Save training history for this fold
        history_path = os.path.join(Config.output_dir, f"training_history_fold{fold+1}.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader, scaler
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate overall CV score
    if Config.n_folds == 1:
        # For single split, only compute score on validation set (OOF contains only val predictions)
        # The training set portion of oof_preds is still zeros, so we can't use the full array
        val_idx = fold_splits[0][1]  # Get validation indices from the single split
        overall_score = compute_spearmanr(
            train_df.iloc[val_idx][Config.target_cols].values, 
            oof_preds[val_idx]
        )
        print(f"\n{'='*50}")
        print(f"Validation Spearman Score: {overall_score:.5f}")
        print(f"{'='*50}\n")
    else:
        # For k-fold CV, all samples have OOF predictions
        overall_score = compute_spearmanr(train_df[Config.target_cols].values, oof_preds)
        print(f"\n{'='*50}")
        print(f"Overall CV Spearman Score: {overall_score:.5f}")
        print(f"{'='*50}\n")
    
    # Save OOF predictions
    np.save(os.path.join(Config.output_dir, "oof_preds.npy"), oof_preds)
    
    # Save overall training summary
    summary = {
        'model_name': Config.model_name,
        'n_folds': Config.n_folds if Config.n_folds > 1 else 1,
        'mode': 'cross_validation' if Config.n_folds > 1 else 'single_split',
        'validation_split': Config.validation_split if Config.n_folds == 1 else None,
        'epochs': Config.epochs,
        'max_length': Config.max_len,
        'batch_size': Config.batch_size,
        'learning_rate': Config.lr,
        'head_learning_rate': Config.head_lr,
        'loss_configuration': {
            'ranking_loss_weight': Config.ranking_loss_weight,
            'spearman_loss_weight': Config.spearman_loss_weight,
            'ranking_margin': Config.ranking_margin,
            'ranking_threshold': Config.ranking_threshold,
            'spearman_temperature': Config.spearman_temperature
        },
        'overall_cv_score': float(overall_score),
        'num_samples': len(train_df),
        'num_targets': len(Config.target_cols)
    }
    
    summary_path = os.path.join(Config.output_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}\n")
    
    return oof_preds


# ==========================================
# 7. Post-processing Utilities (Winning Solution Approach)
# ==========================================
def postprocess_single_column(target, ref):
    """
    Match the distribution of predicted column to training distribution.
    
    This technique from the winning solution adjusts predictions to follow
    the same distribution as the training data. Since Spearman correlation
    is rank-based, this can improve rankings by leveraging training set
    distribution knowledge.
    
    Args:
        target: Predicted values for a single column (numpy array)
        ref: Training values for the same column (numpy array)
        
    Returns:
        Postprocessed predictions scaled to [0, 1]
    """
    # Sort indices by predicted values
    ids = np.argsort(target)
    
    # Get value counts from training data, sorted by value
    counts = sorted(Counter(ref).items(), key=lambda s: s[0])
    scores = np.zeros_like(target)
    
    last_pos = 0
    v = 0
    
    # Assign rank values based on training distribution
    for value, count in counts:
        # Calculate position in test set proportional to training distribution
        next_pos = last_pos + int(round(count / len(ref) * len(target)))
        if next_pos == last_pos:
            next_pos += 1
            
        # Assign same score to samples in this range
        cond = ids[last_pos:next_pos]
        scores[cond] = v
        last_pos = next_pos
        v += 1
    
    # Normalize to [0, 1]
    if scores.max() > 0:
        return scores / scores.max()
    return scores


def postprocess_predictions(predictions, train_df, target_cols, use_distribution_matching=True):
    """
    Apply distribution matching and normalization to predictions.
    
    Since Spearman correlation only cares about rankings, not actual values:
    - Snapping to specific values is NOT helpful
    - Distribution matching CAN help by adjusting rankings
    
    Args:
        predictions: numpy array of shape (n_samples, n_targets)
        train_df: Training dataframe with target columns
        target_cols: List of target column names
        use_distribution_matching: If True, apply distribution matching to selected columns
        
    Returns:
        Postprocessed predictions as numpy array
    """
    postprocessed = predictions.copy()
    
    # Columns where distribution matching showed substantial improvement
    distribution_matching_cols = {
        # Original columns from winner's solution
        'question_conversational',
        'question_type_compare',
        'question_type_definition',
        'question_type_entity',
        'question_has_commonly_accepted_answer',
        'question_type_consequence',
        'question_type_spelling',
        
        # Additional challenging targets with sparse/imbalanced distributions
        'question_type_choice',
        'question_not_really_a_question',
        'question_multi_intent',
        'question_type_procedure',
        'question_type_instructions',
        'answer_type_procedure',
        'answer_type_instructions',
        
        # Columns with discrete/categorical distributions
        'question_expect_short_answer',
        'answer_type_reason_explanation',
    }
    
    for i, col in enumerate(target_cols):
        if use_distribution_matching and col in distribution_matching_cols:
            # Apply distribution matching for specific columns
            scores = postprocess_single_column(
                postprocessed[:, i], 
                train_df[col].values
            )
            postprocessed[:, i] = scores
        
        # Scale all columns to [0, 1] interval
        v = postprocessed[:, i]
        v_min, v_max = v.min(), v.max()
        if v_max > v_min:
            postprocessed[:, i] = (v - v_min) / (v_max - v_min)
        else:
            postprocessed[:, i] = 0.5  # If all values are the same
    
    return postprocessed


# ==========================================
# 8. Inference Pipeline (Ensemble with Multi-GPU)
# ==========================================
@torch.no_grad()
def generate_predictions(model, test_loader, device):
    """Generate predictions on test set"""
    model.eval()
    all_preds = []
    
    for batch in tqdm(test_loader, desc="Inference"):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, mask)
        outputs = torch.sigmoid(outputs.float())
        all_preds.append(outputs.cpu().numpy())
            
    return np.concatenate(all_preds)


def inference_pipeline(use_postprocessing=True):
    """
    Complete inference pipeline with fold ensemble and post-processing.
    
    Args:
        use_postprocessing: If True, apply distribution matching post-processing
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}\n")
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(Config.test_csv)
    print(f"Test samples: {len(test_df):,}\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    
    # Prepare test dataloader with larger batch size for inference
    test_dataset = QuestDatasetModernBERT(
        test_df, 
        tokenizer, 
        max_length=Config.max_len,
        target_cols=None,
        mode="test"
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.batch_size * 4,  # Larger batch for inference
        shuffle=False, 
        num_workers=Config.num_workers,
        pin_memory=True
    )
    
    # Ensemble predictions from all folds
    fold_preds = []
    
    # Determine expected number of models
    expected_folds = max(1, Config.n_folds)
    
    for fold in range(1, expected_folds + 1):
        model_path = os.path.join(Config.output_dir, f"best_model_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"⚠ Model for fold {fold} not found (expected path: {model_path})")
            continue
            
        print(f"Loading model fold {fold}...")
        model = QuestModernBertModel()
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        
        preds = generate_predictions(model, test_loader, device)
        fold_preds.append(preds)
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    if not fold_preds:
        raise ValueError("No models found for inference!")
        
    # Average predictions across folds
    avg_preds = np.mean(fold_preds, axis=0)
    print(f"\n✓ Generated predictions from {len(fold_preds)} fold model(s)")
    
    # Post-processing
    if use_postprocessing and os.path.exists(Config.train_csv):
        print("\nApplying distribution matching post-processing...")
        train_df = pd.read_csv(Config.train_csv)
        final_preds = postprocess_predictions(
            avg_preds, 
            train_df, 
            Config.target_cols,
            use_distribution_matching=True
        )
        print("✓ Distribution matching applied to selected columns")
    else:
        final_preds = avg_preds
        if not os.path.exists(Config.train_csv):
            print("⚠ train.csv not found, skipping post-processing")
        else:
            print("Post-processing disabled")

    # Create submission
    print("\nCreating submission file...")
    submission = pd.DataFrame(final_preds, columns=Config.target_cols)
    submission['qa_id'] = test_df['qa_id']
    submission = submission[['qa_id'] + Config.target_cols]
    
    output_path = os.path.join(Config.output_dir, "submission.csv")
    submission.to_csv(output_path, index=False)
    print(f"✓ Submission saved to {output_path}")
    
    return submission


# ==========================================
# 9. Main Entry Point
# ==========================================
def main():
    """Main function to run training and/or inference"""
    parser = argparse.ArgumentParser(description='Google Quest Q&A Labeling Training & Inference with ModernBERT')
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'inference', 'both'],
                        help='Mode to run: train, inference, or both')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Model name to use (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--n_folds', type=int, default=None,
                        help='Number of folds (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--max_len', type=int, default=None,
                        help='Maximum sequence length (overrides config, ModernBERT supports up to 8192)')
    parser.add_argument('--no_postprocessing', action='store_true',
                        help='Disable distribution matching post-processing')
    
    # Loss configuration arguments
    parser.add_argument('--ranking_weight', type=float, default=None,
                        help='Weight for pairwise ranking loss')
    parser.add_argument('--spearman_weight', type=float, default=None,
                        help='Weight for soft spearman loss')
    parser.add_argument('--ranking_margin', type=float, default=None,
                        help='Margin for pairwise ranking loss')
    parser.add_argument('--ranking_threshold', type=float, default=None,
                        help='Threshold for significant pairs in ranking loss')
    parser.add_argument('--spearman_temperature', type=float, default=None,
                        help='Temperature for soft ranking in spearman loss')
    
    args = parser.parse_args()
    
    # Override config if arguments provided
    if args.model_name:
        Config.model_name = args.model_name
    if args.epochs:
        Config.epochs = args.epochs
    if args.n_folds:
        Config.n_folds = args.n_folds
    if args.batch_size:
        Config.batch_size = args.batch_size
    if args.max_len:
        Config.max_len = args.max_len
    
    # Loss configuration overrides
    if args.ranking_weight is not None:
        Config.ranking_loss_weight = args.ranking_weight
    if args.spearman_weight is not None:
        Config.spearman_loss_weight = args.spearman_weight
    if args.ranking_margin is not None:
        Config.ranking_margin = args.ranking_margin
    if args.ranking_threshold is not None:
        Config.ranking_threshold = args.ranking_threshold
    if args.spearman_temperature is not None:
        Config.spearman_temperature = args.spearman_temperature
    
    # Set seed
    seed_everything(Config.seed)
    
    print("="*50)
    print("Google Quest Q&A Labeling with ModernBERT")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Model: {Config.model_name}")
    print(f"Max Length: {Config.max_len} (ModernBERT supports up to 8192)")
    print(f"Epochs: {Config.epochs}")
    print(f"Folds: {Config.n_folds}", end="")
    if Config.n_folds == 1:
        print(f" (Fast iteration - using {Config.validation_split:.1%} validation split)")
    else:
        print(f" (Full cross-validation)")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Post-processing: {'Disabled' if args.no_postprocessing else 'Enabled'}")
    print(f"Grouping: {Config.grouping_column or 'None'}")
    print()
    print("Loss Function: Combined Ranking Loss")
    print(f"  - Pairwise Ranking: {Config.ranking_loss_weight:.1%} (margin={Config.ranking_margin}, threshold={Config.ranking_threshold})")
    print(f"  - Soft Spearman: {Config.spearman_loss_weight:.1%} (temperature={Config.spearman_temperature})")
    print("="*50 + "\n")
    
    # Run training
    if args.mode in ['train', 'both']:
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        oof_preds = train_loop()
    
    # Run inference
    if args.mode in ['inference', 'both']:
        print("\n" + "="*50)
        print("INFERENCE PHASE")
        print("="*50)
        submission = inference_pipeline(use_postprocessing=not args.no_postprocessing)
        print("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()