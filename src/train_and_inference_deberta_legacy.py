"""
Google Quest Q&A Labeling - Training & Inference (Enhanced with Champion Solution Tricks)

This script implements a DeBERTa-based model for multi-label classification of Q&A pairs.

Overview:
- Training: Fine-tune DeBERTa-v3-large on 30 Q&A quality labels
- Inference: Generate predictions on test data with Target Distribution Matching
- Architecture: Dual-head (CLS + SEP pooling) with multi-layer concatenation
- Loss: Hybrid Loss = BCE + Pairwise Ranking + Soft Spearman (optimized for Spearman correlation)
- Post-processing: Target Distribution Matching (from 1st place solution)

Key Improvements (adapted from 1st place solution):
1. Hybrid Loss: Combines BCE (stability) + Ranking (metric optimization)
2. Target Distribution Matching: Forces prediction distribution to match training data

Usage:
    # Training
    python train_and_inference_deberta.py --mode train
    
    # Inference only
    python train_and_inference_deberta.py --mode inference
    
    # Both training and inference
    python train_and_inference_deberta.py --mode both
"""

import os
import gc
import json
import argparse
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Enable anomaly detection for debugging (can be disabled in production)
torch.autograd.set_detect_anomaly(False)


# ==========================================
# 1. Configuration
# ==========================================

class Config:
    """Training and model configuration"""
    # model_name = "microsoft/deberta-v3-base" 
    model_name = "microsoft/deberta-v3-large"
    # model_name = "microsoft/deberta-v2-xxlarge"
    
    max_len = 512
    batch_size = 4          # Reduced for xxlarge model memory constraints
    accum_iter = 4          # Increased to maintain effective batch size of 16
    
    # Lower LR for stability with large batch size
    lr = 1e-5               # Optimal for xxlarge model
    head_lr = 5e-5          # Classification head LR
    
    # Training configuration
    epochs = 4
    n_folds = 1             # 5-fold CV for robust evaluation and ensemble
    validation_split = 0.2  # Only used when n_folds=1 (fast iteration mode)
    seed = 42
    num_workers = 8         # Increased from 2 for better data loading
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    sample_submission_csv = "data/sample_submission.csv"

    output_dir = "models/deberta_v3_large"
    # output_dir = "models/deberta_v2_xxlarge"
    
    # Loss configuration
    # Hybrid Loss: 50% BCE + 50% Ranking (Ranking split into Pairwise + Spearman)
    bce_loss_weight = 0.5        # Reduced from 1.0 to let Ranking lead
    ranking_loss_weight = 1.0    # Increased from 0.65 to restore primary focus
    spearman_loss_weight = 0.5   # Increased from 0.35 for slight boost
    
    ranking_margin = 0.1         # Margin for ranking loss
    ranking_threshold = 0.05     # Only consider pairs with target difference > threshold
    spearman_temperature = 1.0   # Temperature for soft ranking
    
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
# 2. Custom Loss Functions
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
            pred_col = preds[:, label_idx]  # (batch_size,)
            target_col = targets[:, label_idx]  # (batch_size,)
            
            # Create pairwise differences
            # pred_diff[i, j] = pred[i] - pred[j]
            pred_diff = pred_col.unsqueeze(1) - pred_col.unsqueeze(0)  # (batch_size, batch_size)
            target_diff = target_col.unsqueeze(1) - target_col.unsqueeze(0)  # (batch_size, batch_size)
            
            # Only consider pairs where target difference is significant
            # and i < j to avoid counting pairs twice
            mask = (target_diff.abs() > self.threshold)
            
            # Get upper triangular mask to avoid duplicate pairs
            upper_tri = torch.triu(torch.ones(batch_size, batch_size, device=device), diagonal=1).bool()
            mask = mask & upper_tri
            
            if mask.sum() == 0:
                continue
            
            # For pairs where target[i] > target[j], we want pred[i] > pred[j]
            # Sign indicates direction: +1 if target[i] > target[j], -1 otherwise
            sign = torch.sign(target_diff)
            
            # Margin ranking loss: max(0, margin - sign * (pred[i] - pred[j]))
            # When sign = +1 and pred[i] > pred[j] + margin, loss = 0
            # When sign = -1 and pred[j] > pred[i] + margin, loss = 0
            loss_matrix = F.relu(self.margin - sign * pred_diff)
            
            # Apply mask and compute mean
            masked_loss = loss_matrix[mask]
            if masked_loss.numel() > 0:
                total_loss = total_loss + masked_loss.sum()
                num_valid_pairs += masked_loss.numel()
        
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
            
            # Skip if all targets are the same (correlation undefined)
            if target_col.std() < self.eps:
                continue
            
            # Compute soft ranks
            pred_ranks = self.soft_rank(pred_col)
            target_ranks = self.soft_rank(target_col)
            
            # Compute Pearson correlation between ranks (= Spearman correlation)
            corr = self.pearson_correlation(pred_ranks, target_ranks)
            
            # Clamp correlation to valid range
            corr = torch.clamp(corr, -1.0, 1.0)
            
            correlations.append(corr)
        
        if len(correlations) > 0:
            mean_corr = torch.stack(correlations).mean()
            # Loss = 1 - correlation (we want to maximize correlation, so minimize 1 - corr)
            return 1.0 - mean_corr
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


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
        bce_weight=1.0,
        ranking_weight=0.6,
        spearman_weight=0.4,
        ranking_margin=0.1,
        ranking_threshold=0.05,
        spearman_temperature=1.0
    ):
        """
        Args:
            bce_weight: Weight for BCE loss
            ranking_weight: Weight for pairwise ranking loss
            spearman_weight: Weight for soft spearman loss
            ranking_margin: Margin for ranking loss
            ranking_threshold: Threshold for significant pairs in ranking loss
            spearman_temperature: Temperature for soft ranking
        """
        super().__init__()
        
        self.bce_weight = bce_weight
        self.ranking_weight = ranking_weight
        self.spearman_weight = spearman_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
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
        bce_loss = self.bce_loss(preds, targets)
        ranking_loss = self.ranking_loss(preds, targets)
        spearman_loss = self.spearman_loss(preds, targets)
        
        # Combine losses
        # Note: We normalize the ranking component to be roughly equal to BCE
        total_loss = (
            self.bce_weight * bce_loss +
            self.ranking_weight * ranking_loss +
            self.spearman_weight * spearman_loss
        )
        
        return {
            'loss': total_loss,
            'bce_loss': bce_loss.detach(),
            'ranking_loss': ranking_loss.detach(),
            'spearman_loss': spearman_loss.detach()
        }


# ==========================================
# 3. Dataset Class
# ==========================================
class QuestDataset(Dataset):
    """Custom dataset for Q&A labeling task"""
    
    def __init__(self, df, tokenizer, max_len=512, mode="train"):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        
        self.titles = df['question_title'].values
        self.bodies = df['question_body'].values
        self.answers = df['answer'].values
        
        if self.mode != "test":
            self.targets = df[Config.target_cols].values

    def _trim_input_balanced(self, q_tokens, a_tokens, max_len):
        """
        Balanced trimming: preserve head and tail of both Q and A.
        Based on 23rd place solution's insight that both beginning and end contain important info.
        """
        budget = max_len - 3  # [CLS], [SEP], [SEP]
        q_max = budget // 2
        a_max = budget // 2
        
        q_len = len(q_tokens)
        a_len = len(a_tokens)
        
        if q_len + a_len <= budget:
            return q_tokens, a_tokens
        
        # Redistribute budget based on actual lengths
        if a_len <= a_max and q_len > q_max:
            q_max = budget - a_len
        elif q_len <= q_max and a_len > a_max:
            a_max = budget - q_len
        
        # Balanced head-tail trimming for question
        if q_len > q_max:
            head_len = q_max // 2
            tail_len = q_max - head_len
            q_tokens = q_tokens[:head_len] + q_tokens[-tail_len:]
        
        # Balanced head-tail trimming for answer
        if a_len > a_max:
            head_len = a_max // 2
            tail_len = a_max - head_len
            a_tokens = a_tokens[:head_len] + a_tokens[-tail_len:]
        
        return q_tokens, a_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        body = str(self.bodies[idx])
        answer = str(self.answers[idx])
        
        # Combine question parts
        q_text = title + " " + self.tokenizer.sep_token + " " + body
        a_text = answer
        
        # Tokenize
        q_tokens = self.tokenizer.tokenize(q_text)
        a_tokens = self.tokenizer.tokenize(a_text)
        
        # Balanced head-tail trimming (23rd place solution)
        q_tokens, a_tokens = self._trim_input_balanced(q_tokens, a_tokens, self.max_len)
        
        # Build input IDs and track SEP position
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        
        q_ids = self.tokenizer.convert_tokens_to_ids(q_tokens)
        a_ids = self.tokenizer.convert_tokens_to_ids(a_tokens)
        
        # [CLS] + q_tokens + [SEP] + a_tokens + [SEP]
        ids = [cls_id] + q_ids + [sep_id] + a_ids + [sep_id]
        
        # Position of the [SEP] between Q and A (for dual-head)
        # This is the key position that captures Q-A interaction
        sep_idx = len(q_ids) + 1  # 0-indexed: [CLS]=0, q_tokens=1..len(q_ids), [SEP]=len(q_ids)+1
        
        mask = [1] * len(ids)
        padding_len = self.max_len - len(ids)
        
        ids = ids + [pad_id] * padding_len
        mask = mask + [0] * padding_len
        
        output = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'sep_idx': torch.tensor(sep_idx, dtype=torch.long),
        }
        
        if self.mode != "test":
            labels = self.targets[idx].copy()
            labels = np.clip(labels, 0.0, 1.0)
            labels = np.nan_to_num(labels, nan=0.5, posinf=1.0, neginf=0.0)
            output['labels'] = torch.tensor(labels, dtype=torch.float32)        
            
        return output



# ==========================================
# 4. Model Architecture
# ==========================================

class QuestDebertaModel(nn.Module):
    """
    DeBERTa model with Dual-Head Architecture (CLS + SEP pooling).
    
    Key insight from 23rd place solution:
    - [CLS] token captures overall document semantics
    - [SEP] token (between Q and A) captures the Q-A interaction point
    
    Note: DeBERTa-v3 does NOT use token_type_ids, so we rely on
    sep_idx computed in the Dataset to locate the Q-A boundary.
    """
    
    def __init__(self, model_name=Config.model_name, num_labels=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.config.hidden_size
        
        # Dual-head: CLS token captures overall semantics, SEP token captures Q-A boundary
        # Using last 4 layers concatenated (23rd place approach)
        self.cls_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
        )
        
        self.sep_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
        )
        
        # Final classifier: concatenate CLS and SEP representations
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, num_labels),
        )

    def forward(self, input_ids, attention_mask, sep_idx):
        """
        Forward pass with dual-head architecture.
        
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            sep_idx: (batch,) position of [SEP] token between Q and A
            
        Returns:
            logits: (batch, num_labels) classification logits
        """
        # DeBERTa-v3 does NOT use token_type_ids
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
        )
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
        
        # Extract last 4 layers (23rd place approach)
        last_4_layers = hidden_states[-4:]  # 4 tensors of (batch, seq_len, hidden)
        
        # CLS token: position 0
        cls_embeddings = torch.cat([layer[:, 0, :] for layer in last_4_layers], dim=-1)
        cls_out = self.cls_head(cls_embeddings)  # (batch, hidden)
        
        # SEP token: position varies per sample, use sep_idx
        batch_size = input_ids.size(0)
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        
        sep_embeddings_list = []
        for layer in last_4_layers:
            # layer: (batch, seq_len, hidden)
            sep_emb = layer[batch_indices, sep_idx]  # (batch, hidden)
            sep_embeddings_list.append(sep_emb)
        sep_embeddings = torch.cat(sep_embeddings_list, dim=-1)  # (batch, hidden * 4)
        
        sep_out = self.sep_head(sep_embeddings)  # (batch, hidden)
        
        # Concatenate CLS and SEP representations
        combined = torch.cat([cls_out, sep_out], dim=-1)  # (batch, hidden * 2)
        
        logits = self.classifier(combined)
        return logits


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


def create_dataloaders(train_df, val_df, tokenizer):
    """Create train and validation dataloaders"""
    train_dataset = QuestDataset(train_df, tokenizer, mode="train")
    val_dataset = QuestDataset(val_df, tokenizer, mode="train")
    
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


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """Differential learning rate for backbone and head"""
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, accum_iter):
    """Train one epoch with gradient accumulation"""
    model.train()
    train_loss = 0
    bce_loss_sum = 0
    ranking_loss_sum = 0
    spearman_loss_sum = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        sep_idx = batch['sep_idx'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, mask, sep_idx=sep_idx)
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
        bce_loss_sum += loss_dict['bce_loss'].item()
        ranking_loss_sum += loss_dict['ranking_loss'].item()
        spearman_loss_sum += loss_dict['spearman_loss'].item()
        
        progress_bar.set_postfix({
            'loss': train_loss / (step + 1),
            'bce': bce_loss_sum / (step + 1),
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
        sep_idx = batch['sep_idx'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, mask, sep_idx=sep_idx)
        outputs = torch.sigmoid(outputs)
        
        val_preds.append(outputs.cpu().numpy())
        val_trues.append(labels.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_trues = np.concatenate(val_trues)
    
    # Add safety checks for predictions
    val_preds = np.clip(val_preds, 0, 1)
    val_preds = np.nan_to_num(val_preds, nan=0.5, posinf=1.0, neginf=0.0)
    
    score = compute_spearmanr(val_trues, val_preds)
    
    # Post-processing disabled for validation (noisy on small splits)
    # val_preds_post = postprocess_prediction(...)
    
    return score, val_preds, val_trues


# ==========================================
# 5.1 Post-processing Utilities
# ==========================================
def postprocess_single(target, ref):
    """
    Target Distribution Matching:
    Map the rank of the predicted values to the values of the reference (training) distribution.
    """
    ids = np.argsort(target)
    counts = sorted(Counter(ref).items(), key=lambda s: s[0])
    scores = np.zeros_like(target)

    last_pos = 0
    v = 0

    for value, count in counts:
        next_pos = last_pos + int(round(count / len(ref) * len(target)))
        if next_pos == last_pos:
            next_pos += 1

        cond = ids[last_pos:next_pos]
        scores[cond] = v
        last_pos = next_pos
        v += 1

    return scores / scores.max()


def postprocess_prediction(prediction_df, ref_df):
    """Apply post-processing to all columns"""
    postprocessed = prediction_df.copy()

    for col in prediction_df.columns:
        # Use training distribution as reference
        scores = postprocess_single(prediction_df[col].values, ref_df[col].values)
        
        # Scale to 0-1
        v = scores
        postprocessed[col] = (v - v.min()) / (v.max() - v.min())

    return postprocessed


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
    print("Hybrid Loss Configuration:")
    print(f"  BCE Loss weight: {Config.bce_loss_weight}")
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
        model = QuestDebertaModel()
        model.to(device)

        # Differential Learning Rate
        optimizer_parameters = get_optimizer_params(model, encoder_lr=Config.lr, decoder_lr=Config.head_lr)
        
        optimizer = torch.optim.AdamW(optimizer_parameters, weight_decay=0.01)
        
        # Hybrid Loss (BCE + Pairwise Ranking + Soft Spearman)
        # Initialize loss
        loss_fn = CombinedRankingLoss(
            bce_weight=Config.bce_loss_weight,
            ranking_weight=Config.ranking_loss_weight,
            spearman_weight=Config.spearman_loss_weight,
            ranking_margin=Config.ranking_margin,
            ranking_threshold=Config.ranking_threshold,
            spearman_temperature=Config.spearman_temperature
        )
        
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
        
        # Initialize history tracking
        history = {
            'fold': fold + 1,
            'epochs': [],
            'train_loss': [],
            'val_score': [],
            'best_epoch': 0,
            'best_score': -1.0
        }

        for epoch in range(Config.epochs):
            # Training
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, loss_fn, device, 
                Config.accum_iter
            )
            
            # Validation
            val_score, val_preds, _ = validate(model, val_loader, device)
            
            # Record history
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(float(train_loss))
            history['val_score'].append(float(val_score))
            
            print(f"Epoch {epoch+1:2d}/{Config.epochs} | Loss: {train_loss:.4f} | Val Score: {val_score:.5f}", end="")
            
            # Save best model
            if val_score > best_score:
                best_score = val_score
                history['best_epoch'] = epoch + 1
                history['best_score'] = float(best_score)
                
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
        
        # Save training history to JSON
        history_path = os.path.join(Config.output_dir, f"training_history_fold{fold+1}.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  Training history saved to: {history_path}")
        
        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate overall CV score
    if Config.n_folds == 1:
        # For single split mode, only calculate on validation set
        val_idx = fold_splits[0][1]  # Get validation indices from the single split
        overall_score = compute_spearmanr(
            train_df[Config.target_cols].values[val_idx], 
            oof_preds[val_idx]
        )
        score_desc = "Validation Score (Single Split)"
    else:
        # For k-fold CV, all samples have OOF predictions
        overall_score = compute_spearmanr(train_df[Config.target_cols].values, oof_preds)
        score_desc = "Overall CV Spearman Score"
    
    print(f"\n{'='*50}")
    print(f"{score_desc}: {overall_score:.5f}")
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
    # (identified from the winning solution's experiments)
    # Winner reported 0.027-0.030 boost from this technique
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
        # These typically have low frequency in training data
        'question_type_choice',           # Binary choice questions (rare)
        'question_not_really_a_question',  # Edge case detection
        'question_multi_intent',           # Multi-purpose questions
        'question_type_procedure',         # Step-by-step questions
        'question_type_instructions',      # How-to questions
        'answer_type_procedure',           # Step-by-step answers
        'answer_type_instructions',        # How-to answers
        
        # Columns with discrete/categorical distributions that benefit from matching
        'question_expect_short_answer',    # Binary-like distribution
        'answer_type_reason_explanation',  # Explanation-type answers
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
        # This doesn't affect Spearman but ensures valid submission values
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
        sep_idx = batch['sep_idx'].to(device)
        
        outputs = model(input_ids, mask, sep_idx=sep_idx)
        outputs = torch.sigmoid(outputs)
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
    test_dataset = QuestDataset(test_df, tokenizer, mode="test")
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
        model = QuestDebertaModel()
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
    parser = argparse.ArgumentParser(description='Google Quest Q&A Labeling Training & Inference')
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
    print("Google Quest Q&A Labeling")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Model: {Config.model_name}")
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