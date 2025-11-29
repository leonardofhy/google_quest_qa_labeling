import argparse
import os
import json
from datetime import datetime
import random
import html
import time
import warnings
from typing import List, Tuple
from math import floor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
import lightgbm as lgb
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

class Config:
    seed = 42
    model_name = "microsoft/deberta-v3-large"
    max_sequence_length = 1024
    epochs_phase2 = 3
    epochs_phase1 = 3
    batch_size = 8
    gradient_accumulation_steps = 1
    n_splits = 5
    
    # Optimizer
    encoder_lr = 1e-5
    head_lr = 1e-4
    weight_decay = 1e-2
    max_grad_norm = 1.0
    
    # AWP (Disabled for final run)
    awp_lr = 0.0
    awp_eps = 1e-3
    awp_start_epoch = 2
    
    # Loss Weights
    bce_weight = 0.0
    ranking_weight = 0.5
    spearman_weight = 0.5
    
    # Ranking Loss Params
    ranking_margin = 0.1
    
    # Spearman Loss Params
    spearman_temperature = 0.1
    
    # Automatic Weighting
    use_auto_weighting = False
    
    # Paths
    data_dir = "data"
    output_dir = "models"
    train_csv = "train.csv"
    test_csv = "test.csv"
    sub_csv = "sample_submission.csv"
    
    # Local Eval
    local_eval = True
    test_size = 0.1

def seed_everything(seed=Config.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _preprocess_text(s: str) -> str:
    return html.unescape(s)

def _convert_to_transformer_inputs(title: str, question: str, answer: str, tokenizer, max_sequence_length: int, question_only=False):
    title = _preprocess_text(str(title))
    question = _preprocess_text(str(question))
    answer = _preprocess_text(str(answer))
    
    # 1. Tokenize separately
    title_tokens = tokenizer.tokenize(title)
    question_tokens = tokenizer.tokenize(question)
    answer_tokens = tokenizer.tokenize(answer)
    
    # 2. Define special tokens
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    
    # 3. Calculate budget
    # Format: [CLS] Title Question [SEP] Answer [SEP]
    # Special tokens count: 3 ([CLS], [SEP], [SEP])
    special_tokens_count = 3
    available_length = max_sequence_length - special_tokens_count - len(title_tokens)
    
    # If title is too long (unlikely but possible), truncate it
    if available_length < 0:
        title_tokens = title_tokens[:max_sequence_length - special_tokens_count]
        available_length = 0
        question_tokens = []
        answer_tokens = []
    
    if question_only:
        # If question only, give all budget to question
        if len(question_tokens) > available_length:
            # Head+Tail Truncation for Question
            head_len = available_length // 2
            tail_len = available_length - head_len
            question_tokens = question_tokens[:head_len] + question_tokens[-tail_len:]
        answer_tokens = []
    else:
        # Distribute budget between Question and Answer
        # Strategy: Try to keep them balanced, or give more to Answer?
        # Baseline strategy: trim both if needed.
        # Let's assume 50/50 split if both are long
        
        # First, check if total fits
        if len(question_tokens) + len(answer_tokens) > available_length:
            # We need to trim.
            # Simple strategy: Alloc half to Q, half to A.
            # If one is short, give the rest to the other.
            
            q_len = len(question_tokens)
            a_len = len(answer_tokens)
            
            if q_len + a_len > available_length:
                # Calculate target lengths
                target_q_len = available_length // 2
                target_a_len = available_length - target_q_len
                
                # Adjust if one is shorter than target
                if q_len < target_q_len:
                    target_a_len += (target_q_len - q_len)
                    target_q_len = q_len
                elif a_len < target_a_len:
                    target_q_len += (target_a_len - a_len)
                    target_a_len = a_len
                
                # Apply Head+Tail Truncation
                if len(question_tokens) > target_q_len:
                    head = target_q_len // 2
                    tail = target_q_len - head
                    question_tokens = question_tokens[:head] + question_tokens[-tail:]
                    
                if len(answer_tokens) > target_a_len:
                    head = target_a_len // 2
                    tail = target_a_len - head
                    answer_tokens = answer_tokens[:head] + answer_tokens[-tail:]

    # 4. Construct Input IDs
    # [CLS] Title + Question [SEP] Answer [SEP]
    tokens = [cls_token] + title_tokens + question_tokens + [sep_token]
    if not question_only:
        tokens += answer_tokens + [sep_token]
    else:
        # Even for question only, we usually append a SEP at the end
        # But wait, if question_only, we just want [CLS] Title Question [SEP]
        pass

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # 5. Padding
    if len(input_ids) < max_sequence_length:
        padding_length = max_sequence_length - len(input_ids)
        input_ids = input_ids + [pad_token_id] * padding_length
        
    # 6. Attention Mask
    # 1 for tokens, 0 for padding
    attention_mask = [1] * len(tokens) + [0] * (max_sequence_length - len(tokens))
    
    # 7. Token Type IDs
    # DeBERTa v3 doesn't strictly need them, but we can generate them.
    # 0 for [CLS] Title Question [SEP]
    # 1 for Answer [SEP]
    # 0 for Padding
    
    len_q = 1 + len(title_tokens) + len(question_tokens) + 1 # [CLS] ... [SEP]
    len_a = len(answer_tokens) + 1 if not question_only else 0 # ... [SEP]
    
    token_type_ids = [0] * len_q + [1] * len_a + [0] * (max_sequence_length - len_q - len_a)
    
    return input_ids, token_type_ids, attention_mask

def compute_input_arrays(df, tokenizer, max_sequence_length, question_only=False):
    input_ids, input_token_type_ids, input_attention_masks = [], [], []
    for title, body, answer in tqdm(zip(df["question_title"].values, df["question_body"].values, df["answer"].values), total=len(df), desc="Tokenizing"):
        ids, type_ids, mask = _convert_to_transformer_inputs(title, body, answer, tokenizer, max_sequence_length, question_only=question_only)
        input_ids.append(ids)
        input_token_type_ids.append(type_ids)
        input_attention_masks.append(mask)
    return (
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(input_token_type_ids, dtype=np.int32),
        np.asarray(input_attention_masks, dtype=np.int32),
    )

def compute_output_arrays(df, output_categories):
    return np.asarray(df[output_categories])

class Fold(object):
    def __init__(self, n_splits=5, shuffle=True, random_state=71):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_groupkfold(self, train, group_name):
        group = train[group_name]
        unique_group = group.unique()

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        folds_ids = []
        for trn_group_idx, val_group_idx in kf.split(unique_group):
            trn_group = unique_group[trn_group_idx]
            val_group = unique_group[val_group_idx]
            is_trn = group.isin(trn_group)
            is_val = group.isin(val_group)
            trn_idx = train[is_trn].index
            val_idx = train[is_val].index
            folds_ids.append((trn_idx, val_idx))

        return folds_ids

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float))

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average

class AttentionPooling(nn.Module):
    """Attention-based pooling layer"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, hidden_states, attention_mask):
        # hidden_states: (batch, seq_len, hidden_size)
        # attention_mask: (batch, seq_len)
        
        attention_scores = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        attention_scores = attention_scores.masked_fill(
            attention_mask == 0, float('-inf')
        )
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            hidden_states                     # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return pooled

class Model(nn.Module):
    def __init__(self, model_name=Config.model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.config = config
        
        # Weighted Layer Pooling: Use last 4 layers
        self.pooler = WeightedLayerPooling(
            num_hidden_layers=config.num_hidden_layers, 
            layer_start=config.num_hidden_layers - 4,
            layer_weights=None
        )
        
        # Attention Pooling (NEW)
        self.attention_pool = AttentionPooling(config.hidden_size)
        
        # Multi-Sample Dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(5)])
        self.linear = nn.Linear(config.hidden_size, 30)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # DeBERTa v3 handles token_type_ids automatically if passed, but for safety we can pass them.
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        # Weighted Layer Pooling
        all_hidden_states = torch.stack(outputs.hidden_states)
        weighted_pooling_embeddings = self.pooler(all_hidden_states)
        
        # Attention Pooling (instead of Mean Pooling)
        pooled_output = self.attention_pool(weighted_pooling_embeddings, attention_mask)
        
        # Multi-Sample Dropout
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                x = self.linear(dropout(pooled_output))
            else:
                x += self.linear(dropout(pooled_output))
        
        x = x / len(self.dropouts)
        return x

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        adv_lr=1,
        adv_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, inputs, labels, label_weights, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            with torch.cuda.amp.autocast():
                # Re-run forward pass with perturbed weights
                # Note: We need to unpack inputs here just like in the main loop
                input_ids, token_type_ids, attention_mask = inputs
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                # Note: AWP attack step usually uses the same loss function, but for simplicity/speed 
                # we often just use BCE or the main loss. 
                # Here we need to pass config and spearman_loss_fn if we want full consistency.
                # However, `awp.attack_backward` signature doesn't have them.
                # Let's assume we pass them or modify AWP. 
                # For now, let's keep it simple and use a simplified loss or pass None for spearman to save compute?
                # Actually, AWP class is defined in this file, so I can modify it.
                # But to minimize changes, I'll just use BCE + Ranking (if I can access config)
                # Or better, I should update AWP signature.
                pass # Placeholder for the diff block, see below for actual replacement
                
                # Wait, I need to update AWP.attack_backward signature too.
                # Let's do that in a separate chunk or include it here.
                pass
            
            self.optimizer.zero_grad()
            if self.scaler:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

class AWP:
    def __init__(
        self,
        model,
        optimizer,
        adv_param="weight",
        awp_lr=1,
        awp_eps=0.2,
        start_epoch=0,
        adv_step=1,
        scaler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, inputs, labels, label_weights, epoch, config, spearman_loss_fn=None, auto_loss_fn=None, question_only=False):
        if (self.awp_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save() 
        for i in range(self.adv_step):
            self._attack_step() 
            with torch.cuda.amp.autocast():
                # Re-run forward pass with perturbed weights
                # Note: We need to unpack inputs here just like in the main loop
                input_ids, token_type_ids, attention_mask = inputs
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                
                if auto_loss_fn is not None:
                    loss_dict = compute_loss(outputs, labels, label_weights, config, spearman_loss_fn, 
                                           question_only=question_only, return_dict=True)
                    adv_loss = auto_loss_fn(loss_dict['bce'], loss_dict['ranking'], loss_dict['spearman'])
                else:
                    adv_loss = compute_loss(outputs, labels, label_weights, config, spearman_loss_fn,
                                          question_only=question_only)
            
            # DON'T zero grad here! We want to accumulate gradients
            # self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()
            
        self._restore()
        
        return adv_loss.detach()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.awp_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.awp_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

class SoftSpearmanLoss(nn.Module):
    """
    Differentiable approximation of Spearman correlation loss.
    Ported from legacy implementation.
    """
    def __init__(self, temperature=1.0, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
    
    def soft_rank(self, x):
        batch_size = x.shape[0]
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        soft_compare = torch.sigmoid(diff / self.temperature)
        soft_ranks = soft_compare.sum(dim=1)
        return soft_ranks
    
    def pearson_correlation(self, x, y):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum() + self.eps)
        return numerator / denominator
    
    def forward(self, preds, targets):
        # Apply sigmoid to get probabilities
        preds = torch.sigmoid(preds)
        
        correlations = []
        num_labels = preds.shape[1]
        
        for label_idx in range(num_labels):
            pred_col = preds[:, label_idx]
            target_col = targets[:, label_idx]
            
            if target_col.std() < self.eps:
                continue
            
            pred_ranks = self.soft_rank(pred_col)
            target_ranks = self.soft_rank(target_col)
            
            corr = self.pearson_correlation(pred_ranks, target_ranks)
            corr = torch.clamp(corr, -1.0, 1.0)
            correlations.append(corr)
        
        if len(correlations) > 0:
            mean_corr = torch.stack(correlations).mean()
            return 1.0 - mean_corr
        else:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss using uncertainty weighting.
    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics" (CVPR 2018)
    """
    def __init__(self, num=3):
        super().__init__()
        # log_vars = log(sigma^2)
        # Initialize with 0.0 (sigma=1.0)
        self.log_vars = nn.Parameter(torch.zeros(num, requires_grad=True))

    def forward(self, *losses):
        # loss = sum( exp(-log_var) * loss + 0.5 * log_var )
        # This is equivalent to 1/(2*sigma^2) * loss + log(sigma)
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + 0.5 * self.log_vars[i]
        return total_loss

def compute_loss(outputs, targets, label_weights, config, spearman_loss_fn=None, question_only=False, return_dict=False):
    if question_only:
        outputs = outputs[:, :21]
        targets = targets[:, :21]
        current_label_weights = label_weights[:21]
    else:
        current_label_weights = label_weights

    # 1. BCE Loss
    bce = F.binary_cross_entropy_with_logits(outputs, targets, reduction="none")
    bce_val = (bce * current_label_weights).mean()
    
    # 2. Ranking Loss
    batch_size = outputs.size(0)
    if batch_size % 2 == 0:
        outputs1, outputs2 = outputs.sigmoid().contiguous().view(2, batch_size // 2, outputs.size(-1))
        targets1, targets2 = targets.contiguous().view(2, batch_size // 2, outputs.size(-1))
        ordering = (targets1 > targets2).float() - (targets1 < targets2).float()
        margin_rank_loss = (-ordering * (outputs1 - outputs2) + config.ranking_margin).clamp(min=0.0)
        margin_rank_loss_val = (margin_rank_loss * current_label_weights).mean()
    else:
        margin_rank_loss_val = torch.tensor(0.0, device=outputs.device)

    # 3. Spearman Loss
    if spearman_loss_fn is not None:
        spearman_loss_val = spearman_loss_fn(outputs, targets)
    else:
        spearman_loss_val = torch.tensor(0.0, device=outputs.device)

    # Combined Loss
    total_loss = (config.bce_weight * bce_val) + \
                 (config.ranking_weight * margin_rank_loss_val) + \
                 (config.spearman_weight * spearman_loss_val)

    if return_dict:
        return {
            "total": total_loss,
            "bce": bce_val,
            "ranking": margin_rank_loss_val,
            "spearman": spearman_loss_val
        }

    return total_loss

def compute_spearmanr(trues, preds):
    rhos = []
    for col_trues, col_pred in zip(trues.T, preds.T):
        if len(np.unique(col_pred)) == 1:
            col_pred[np.random.randint(0, len(col_pred) - 1)] = col_pred.max() + 1
        rhos.append(spearmanr(col_trues, col_pred).correlation)
    return np.mean(rhos)

def save_config(config, output_dir):
    """Save configuration to JSON file"""
    config_dict = {
        "seed": config.seed,
        "model_name": config.model_name,
        "max_sequence_length": config.max_sequence_length,
        "epochs_phase2": config.epochs_phase2,
        "epochs_phase1": config.epochs_phase1,
        "batch_size": config.batch_size,
        "n_splits": config.n_splits,
        "encoder_lr": config.encoder_lr,
        "head_lr": config.head_lr,
        "weight_decay": config.weight_decay,
        "data_dir": config.data_dir,
        "output_dir": config.output_dir,
        "local_eval": config.local_eval,
        "test_size": config.test_size,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved to {os.path.join(output_dir, 'config.json')}")

def train_and_predict(train_data, valid_data, test_data, q_train_data, q_valid_data, q_test_data, 
                      fold, device, label_weights, output_dir, config):
    
    dataloader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size)
    valid_dataloader = DataLoader(valid_data, shuffle=False, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size)
    q_dataloader = DataLoader(q_train_data, shuffle=True, batch_size=config.batch_size)
    q_valid_dataloader = DataLoader(q_valid_data, shuffle=False, batch_size=config.batch_size)
    q_test_dataloader = DataLoader(q_test_data, shuffle=False, batch_size=config.batch_size)

    model = Model().to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    
    # Layer-wise Learning Rate Decay (LLRD)
    optimizer_grouped_parameters = []
    
    # 1. Head Parameters (Highest LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.named_parameters() if "bert" not in n],
        "weight_decay": config.weight_decay,
        "lr": config.head_lr
    })
    
    # 2. DeBERTa Layers (Decaying LR)
    # DeBERTa v3 base has 12 layers. 
    # Layer 11 (top) gets encoder_lr, Layer 0 (bottom) gets encoder_lr * (decay ** 11)
    # Embeddings get encoder_lr * (decay ** 12)
    
    layer_decay = 0.9
    num_layers = model.config.num_hidden_layers
    
    # Embeddings
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.named_parameters() if "embeddings" in n],
        "weight_decay": config.weight_decay,
        "lr": config.encoder_lr * (layer_decay ** num_layers)
    })
    
    # Encoder Layers
    for layer_i in range(num_layers):
        # Capture parameters for this specific layer
        # Note: DeBERTa layer naming is usually 'encoder.layer.0', 'encoder.layer.1', etc.
        layer_params = [p for n, p in model.named_parameters() if f"encoder.layer.{layer_i}." in n]
        
        optimizer_grouped_parameters.append({
            "params": layer_params,
            "weight_decay": config.weight_decay,
            "lr": config.encoder_lr * (layer_decay ** (num_layers - 1 - layer_i))
        })
        
    # Other parameters (Pooler, etc. if any, though we replaced pooler)
    # Just in case we missed anything, give them base LR
    existing_params = set()
    for group in optimizer_grouped_parameters:
        for p in group["params"]:
            existing_params.add(p)
            
    remaining_params = [p for p in model.parameters() if p not in existing_params]
    if remaining_params:
        optimizer_grouped_parameters.append({
            "params": remaining_params,
            "weight_decay": config.weight_decay,
            "lr": config.encoder_lr
        })
        
    # Automatic Weighting Parameters
    auto_loss_fn = None
    if config.use_auto_weighting:
        print("Enabling Automatic Loss Weighting...")
        auto_loss_fn = AutomaticWeightedLoss(num=3).to(device)
        optimizer_grouped_parameters.append({
            "params": auto_loss_fn.parameters(),
            "weight_decay": 0.0,
            "lr": 1e-3 # Usually needs higher LR than model
        })

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(q_dataloader) * (config.epochs_phase1) * 0.1),
        num_training_steps=len(q_dataloader) * (config.epochs_phase1)
    )
    
    test_predictions = []
    valid_predictions = []

    # Create best_model directory
    best_model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    
    # Initialize AWP
    # Start AWP from epoch 1 (second epoch) to let model stabilize first
    awp = AWP(model, optimizer, awp_lr=config.awp_lr, awp_eps=config.awp_eps, start_epoch=config.awp_start_epoch, scaler=scaler)

    # Initialize SoftSpearmanLoss
    spearman_loss_fn = SoftSpearmanLoss(temperature=config.spearman_temperature).to(device)

    ## Question Only
    print(f"Fold {fold}: Training Question Only")
    best_q_spearman = -1
    
    accumulation_steps = config.gradient_accumulation_steps
    
    for epoch in range(config.epochs_phase1): 
        start = time.time()
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        for step, (input_ids, token_type_ids, attention_mask, targets) in enumerate(tqdm(q_dataloader, total=len(q_dataloader), desc=f"Epoch {epoch+1}/{config.epochs_phase1}")):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            
            # Clear gradients at start of step only if not accumulating
            if step % accumulation_steps == 0:
                optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                if config.use_auto_weighting:
                    loss_dict = compute_loss(outputs, targets, label_weights, config, spearman_loss_fn, question_only=True, return_dict=True)
                    loss = auto_loss_fn(loss_dict['bce'], loss_dict['ranking'], loss_dict['spearman'])
                else:
                    loss = compute_loss(outputs, targets, label_weights, config, spearman_loss_fn, question_only=True)
                
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            train_preds.extend(outputs.detach().sigmoid().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
            train_losses.append(loss.detach().cpu().item() * accumulation_steps)
        
        model.eval()
        valid_losses = []
        valid_preds = []
        valid_targets = []
        with torch.no_grad():
            # Use q_valid_dataloader for question-only validation
            for input_ids, token_type_ids, attention_mask, targets in q_valid_dataloader:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                prob = outputs.sigmoid()
                prob[:, 21:] = 0.0 # Zero out answer predictions
                valid_preds.extend(prob.cpu().numpy())
                valid_targets.extend(targets.cpu().numpy())
                loss = compute_loss(outputs, targets, label_weights, config, spearman_loss_fn, question_only=True)
                valid_losses.append(loss.detach().cpu().item())
            
            valid_predictions.append(np.stack(valid_preds))
            
            test_preds = []
            # Use q_test_dataloader for question-only test predictions
            for input_ids, token_type_ids, attention_mask in q_test_dataloader:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                prob = outputs.sigmoid()
                prob[:, 21:] = 0.0
                test_preds.extend(prob.cpu().numpy())
            test_predictions.append(np.stack(test_preds))

        print("Epoch {}: Train Loss {}, Valid Loss {}".format(epoch + 1, np.mean(train_losses), np.mean(valid_losses)))
        train_spearman = compute_spearmanr(np.stack(train_targets), np.stack(train_preds))
        valid_spearman_avg = compute_spearmanr(np.stack(valid_targets), sum(valid_predictions) / len(valid_predictions))
        valid_spearman_last = compute_spearmanr(np.stack(valid_targets), valid_predictions[-1])
        
        print("\t Train Spearmanr {:.4f}, Valid Spearmanr (avg) {:.4f}, Valid Spearmanr (last) {:.4f}".format(
            train_spearman, valid_spearman_avg, valid_spearman_last
        ))
        print("\t elapsed: {}s".format(time.time() - start))

        # Save best model for Q phase (optional, usually we care about final QA model)
        if valid_spearman_last > best_q_spearman:
            best_q_spearman = valid_spearman_last
            # torch.save(model.state_dict(), os.path.join(output_dir, f"q_model_fold{fold}_best.bin"))
            # print(f"Saved best Q model with Spearman {best_q_spearman}")

        # Log entry
        log_entry = {
            "fold": fold,
            "phase": "question_only",
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "valid_loss": float(np.mean(valid_losses)),
            "train_spearman": float(train_spearman),
            "valid_spearman_avg": float(valid_spearman_avg),
            "valid_spearman_last": float(valid_spearman_last),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add learned loss weights if using auto weighting
        if config.use_auto_weighting and auto_loss_fn is not None:
            learned_weights = torch.exp(-auto_loss_fn.log_vars).detach().cpu().numpy()
            log_entry["learned_weights"] = {
                "bce": float(learned_weights[0]),
                "ranking": float(learned_weights[1]),
                "spearman": float(learned_weights[2])
            }
            print(f"\t Learned Weights: BCE={learned_weights[0]:.4f}, Rank={learned_weights[1]:.4f}, Spear={learned_weights[2]:.4f}")
            
        with open(os.path.join(output_dir, "training_log.jsonl"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    ## Q and A
    print(f"Fold {fold}: Training Q and A")
    model = Model().to(device) # Reset model? The notebook does this.
    
    # Layer-wise Learning Rate Decay (LLRD) for Q&A Phase
    optimizer_grouped_parameters = []
    
    # 1. Head Parameters (Highest LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.named_parameters() if "bert" not in n],
        "weight_decay": config.weight_decay,
        "lr": config.head_lr
    })
    
    # 2. DeBERTa Layers (Decaying LR)
    layer_decay = 0.9
    num_layers = model.config.num_hidden_layers
    
    # Embeddings
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.named_parameters() if "embeddings" in n],
        "weight_decay": config.weight_decay,
        "lr": config.encoder_lr * (layer_decay ** num_layers)
    })
    
    # Encoder Layers
    for layer_i in range(num_layers):
        layer_params = [p for n, p in model.named_parameters() if f"encoder.layer.{layer_i}." in n]
        optimizer_grouped_parameters.append({
            "params": layer_params,
            "weight_decay": config.weight_decay,
            "lr": config.encoder_lr * (layer_decay ** (num_layers - 1 - layer_i))
        })
        
    # Remaining
    existing_params = set()
    for group in optimizer_grouped_parameters:
        for p in group["params"]:
            existing_params.add(p)
    remaining_params = [p for p in model.parameters() if p not in existing_params]
    if remaining_params:
        optimizer_grouped_parameters.append({
            "params": remaining_params,
            "weight_decay": config.weight_decay,
            "lr": config.encoder_lr
        })

    # Automatic Weighting Parameters (Reset for Q&A phase? Or reuse?)
    # Usually we might want to continue training the weights or reset.
    # Since we re-initialize the model, we should probably re-initialize the loss weights too, 
    # or at least re-create the optimizer group.
    # Let's re-initialize to be safe and consistent with model reset.
    if config.use_auto_weighting:
        print("Enabling Automatic Loss Weighting (Q&A Phase)...")
        # If we want to keep learned weights, we should pass them. But here we reset model.
        auto_loss_fn = AutomaticWeightedLoss(num=3).to(device) 
        optimizer_grouped_parameters.append({
            "params": auto_loss_fn.parameters(),
            "weight_decay": 0.0,
            "lr": 1e-3
        })

    optimizer = AdamW(optimizer_grouped_parameters)
    
    scaler = torch.cuda.amp.GradScaler()
    
    # Re-init scheduler for Q&A phase
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(len(dataloader) * (config.epochs_phase2) * 0.1),
        num_training_steps=len(dataloader) * (config.epochs_phase2)
    )

    # Initialize AWP for Q&A phase
    awp = AWP(model, optimizer, awp_lr=config.awp_lr, awp_eps=config.awp_eps, start_epoch=config.awp_start_epoch, scaler=scaler)

    # Initialize SoftSpearmanLoss
    spearman_loss_fn = SoftSpearmanLoss(temperature=config.spearman_temperature).to(device)

    best_qa_spearman = -1

    for epoch in range(config.epochs_phase2): 
        start = time.time()
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        for step, (input_ids, token_type_ids, attention_mask, targets) in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{config.epochs_phase2}")):
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)
            
            if step % accumulation_steps == 0:
                optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                if config.use_auto_weighting:
                    loss_dict = compute_loss(outputs, targets, label_weights, config, spearman_loss_fn, return_dict=True)
                    loss = auto_loss_fn(loss_dict['bce'], loss_dict['ranking'], loss_dict['spearman'])
                else:
                    loss = compute_loss(outputs, targets, label_weights, config, spearman_loss_fn)
                
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # AWP Attack
            # inputs = (input_ids, token_type_ids, attention_mask)
            # awp.attack_backward(inputs, targets, label_weights, epoch, config, spearman_loss_fn, auto_loss_fn)
            
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            train_preds.extend(outputs.detach().sigmoid().cpu().numpy())
            train_targets.extend(targets.detach().cpu().numpy())
            train_losses.append(loss.detach().cpu().item() * accumulation_steps)
        
        model.eval()
        valid_losses = []
        valid_preds = []
        valid_targets = []
        with torch.no_grad():
            for input_ids, token_type_ids, attention_mask, targets in valid_dataloader:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                targets = targets.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                valid_preds.extend(outputs.sigmoid().cpu().numpy())
                valid_targets.extend(targets.cpu().numpy())
                loss = compute_loss(outputs, targets, label_weights, config, spearman_loss_fn)
                valid_losses.append(loss.detach().cpu().item())
            
            valid_predictions.append(np.stack(valid_preds))
            
            test_preds = []
            for input_ids, token_type_ids, attention_mask in test_dataloader:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                test_preds.extend(outputs.sigmoid().cpu().numpy())
            test_predictions.append(np.stack(test_preds))

        print("Epoch {}: Train Loss {}, Valid Loss {}".format(epoch + 1, np.mean(train_losses), np.mean(valid_losses)))
        train_spearman = compute_spearmanr(np.stack(train_targets), np.stack(train_preds))
        valid_spearman_avg = compute_spearmanr(np.stack(valid_targets), sum(valid_predictions) / len(valid_predictions))
        valid_spearman_last = compute_spearmanr(np.stack(valid_targets), valid_predictions[-1])

        print("\t Train Spearmanr {:.4f}, Valid Spearmanr (avg) {:.4f}, Valid Spearmanr (last) {:.4f}".format(
            train_spearman, valid_spearman_avg, valid_spearman_last
        ))
        print("\t elapsed: {}s".format(time.time() - start))

        # Save Best Model
        if valid_spearman_last > best_qa_spearman:
            best_qa_spearman = valid_spearman_last
            torch.save(model.state_dict(), os.path.join(best_model_dir, f"qa_model_fold{fold}_best.bin"))
            print(f"Saved best QA model with Spearman {best_qa_spearman}")

        log_entry = {
            "fold": fold,
            "phase": "qa",
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "valid_loss": float(np.mean(valid_losses)),
            "train_spearman": float(train_spearman),
            "valid_spearman_avg": float(valid_spearman_avg),
            "valid_spearman_last": float(valid_spearman_last),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add learned loss weights if using auto weighting
        if config.use_auto_weighting and auto_loss_fn is not None:
            learned_weights = torch.exp(-auto_loss_fn.log_vars).detach().cpu().numpy()
            log_entry["learned_weights"] = {
                "bce": float(learned_weights[0]),
                "ranking": float(learned_weights[1]),
                "spearman": float(learned_weights[2])
            }
            print(f"\t Learned Weights: BCE={learned_weights[0]:.4f}, Rank={learned_weights[1]:.4f}, Spear={learned_weights[2]:.4f}")
            
        with open(os.path.join(output_dir, "training_log.jsonl"), "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return valid_predictions, test_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=Config.data_dir, help="Directory containing train.csv and test.csv")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir, help="Directory to save submission")
    parser.add_argument("--local_eval", action="store_true", help="Run in local evaluation mode (split train into train/test)")
    parser.add_argument("--epochs_phase2", type=int, default=Config.epochs_phase2, help="Number of epochs for Q&A training")
    parser.add_argument("--epochs_phase1", type=int, default=Config.epochs_phase1, help="Number of epochs for Question-only training")
    parser.add_argument("--folds", type=int, default=Config.n_splits, help="Number of folds to run")
    args = parser.parse_args()

    # Update Config with args
    Config.data_dir = args.data_dir
    Config.output_dir = args.output_dir
    Config.local_eval = args.local_eval
    Config.epochs_phase2 = args.epochs_phase2
    Config.epochs_phase1 = args.epochs_phase1
    
    # We don't update Config.n_splits because that affects KFold splitting logic.
    # We only control how many folds we iterate over.
    run_folds = args.folds
    Config.run_folds = run_folds # Save to Config for logging

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Config.output_dir = os.path.join(Config.output_dir, timestamp)
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Save configuration
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__') and not callable(v)}
    with open(os.path.join(Config.output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"Configuration saved to {os.path.join(Config.output_dir, 'config.json')}")
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Model Name:        {Config.model_name}")
    print(f"Max Seq Length:    {Config.max_sequence_length}")
    print(f"Epochs (Phase 1):  {Config.epochs_phase1}")
    print(f"Epochs (Phase 2):  {Config.epochs_phase2}")
    print(f"Batch Size:        {Config.batch_size}")
    print(f"Grad Accum Steps:  {Config.gradient_accumulation_steps}")
    print(f"N Splits (CV):     {Config.n_splits}")
    print(f"Run Folds:         {run_folds}")
    print(f"Learning Rate:")
    print(f"  - Encoder:       {Config.encoder_lr}")
    print(f"  - Head:          {Config.head_lr}")
    print(f"Weight Decay:      {Config.weight_decay}")
    print(f"AWP:")
    print(f"  - LR:            {Config.awp_lr}")
    print(f"  - EPS:           {Config.awp_eps}")
    print(f"  - Start Epoch:   {Config.awp_start_epoch}")
    print(f"Loss Weights:")
    print(f"  - BCE:           {Config.bce_weight}")
    print(f"  - Ranking:       {Config.ranking_weight}")
    print(f"  - Spearman:      {Config.spearman_weight}")
    print(f"Auto Weighting:    {Config.use_auto_weighting}")
    print(f"Seed:              {Config.seed}")
    print(f"Local Eval:        {Config.local_eval}")
    print(f"Output Directory:  {Config.output_dir}")
    print("="*60 + "\n")

    # Load Data
    df_train = pd.read_csv(os.path.join(Config.data_dir, Config.train_csv))
    
    if Config.local_eval:
        print("LOCAL EVAL MODE: Splitting train.csv into train and local_test")
        # Split train into train and holdout
        df_train, df_test = train_test_split(df_train, test_size=Config.test_size, random_state=42)
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        print(f"Train shape: {df_train.shape}, Local Test shape: {df_test.shape}")
        # Save local test targets for evaluation
        local_test_targets = df_test[list(df_train.columns[11:])].values
    else:
        df_test = pd.read_csv(os.path.join(Config.data_dir, Config.test_csv))
    
    output_categories = list(df_train.columns[11:])
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    
    # Save tokenizer to best_model directory for offline usage
    best_model_dir = os.path.join(Config.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Tokenizer saved to {best_model_dir}")

    # Compute Inputs
    print("Computing input arrays...")
    outputs = torch.tensor(compute_output_arrays(df_train, output_categories), dtype=torch.float)
    inputs = [torch.tensor(x, dtype=torch.long) for x in compute_input_arrays(df_train, tokenizer, Config.max_sequence_length)]
    question_only_inputs = [torch.tensor(x, dtype=torch.long) for x in compute_input_arrays(df_train, tokenizer, Config.max_sequence_length, question_only=True)]
    test_inputs = [torch.tensor(x, dtype=torch.long) for x in compute_input_arrays(df_test, tokenizer, Config.max_sequence_length)]
    test_question_only_inputs = [torch.tensor(x, dtype=torch.long) for x in compute_input_arrays(df_test, tokenizer, Config.max_sequence_length, question_only=True)]

    # Label Weights
    LABEL_WEIGHTS = torch.tensor(1.0 / df_train[output_categories].std().values, dtype=torch.float32).to(device)
    LABEL_WEIGHTS = LABEL_WEIGHTS / LABEL_WEIGHTS.sum() * 30

    # KFold
    # kf = KFold(n_splits=Config.n_splits, shuffle=True, random_state=71)
    # fold_ids = list(kf.split(df_train)) # Using simple KFold for simplicity, notebook used GroupKFold on URL but KFold is fine for general purpose
    
    # GroupKFold (Matching Notebook)
    gkf = Fold(n_splits=Config.n_splits, shuffle=True, random_state=71)
    fold_ids = gkf.get_groupkfold(df_train, group_name="url")

    histories = []
    test_dataset = TensorDataset(*test_inputs)
    q_test_dataset = TensorDataset(*test_question_only_inputs)

    for fold, (train_idx, valid_idx) in enumerate(fold_ids):
        if fold >= run_folds:
            break
        print(f"Starting Fold {fold+1}/{Config.n_splits}")
        
        train_inputs_fold = [inputs[i][train_idx] for i in range(3)]
        q_train_inputs_fold = [question_only_inputs[i][train_idx] for i in range(3)]
        train_outputs_fold = outputs[train_idx]
        train_dataset = TensorDataset(*train_inputs_fold, train_outputs_fold)
        q_train_dataset = TensorDataset(*q_train_inputs_fold, train_outputs_fold)

        valid_inputs_fold = [inputs[i][valid_idx] for i in range(3)]
        q_valid_inputs_fold = [question_only_inputs[i][valid_idx] for i in range(3)]
        valid_outputs_fold = outputs[valid_idx]
        valid_dataset = TensorDataset(*valid_inputs_fold, valid_outputs_fold)
        q_valid_dataset = TensorDataset(*q_valid_inputs_fold, valid_outputs_fold)

        history = train_and_predict(
            train_data=train_dataset, 
            valid_data=valid_dataset,
            test_data=test_dataset, 
            q_train_data=q_train_dataset, 
            q_valid_data=q_valid_dataset,
            q_test_data=q_test_dataset, 
            fold=fold,
            device=device, label_weights=LABEL_WEIGHTS, output_dir=Config.output_dir, config=Config
        )
        histories.append(history)

    # Post-processing (Averaging)
    # The notebook does LightGBM stacking, but for the script, let's first implement simple averaging as a baseline
    # and then if needed we can add LightGBM. The prompt asked to implement based on the notebook, so I should probably include LightGBM.
    # However, LightGBM part is quite complex to set up in a single script without saving intermediate OOFs.
    # Let's stick to the notebook logic:
    
    # 1. Get val preds per each epoch
    if len(histories) > 0:
        n_epochs_total = len(histories[0][0]) # q_epochs + epochs (actually the notebook returns list of lists)
    else:
        n_epochs_total = 0
        
    # Wait, the notebook returns `valid_predictions` which is a list of preds for each epoch.
    # In `train_and_predict` I append to `valid_predictions` in both loops.
    # So `histories[fold][0]` is a list of arrays, one for each epoch.
    
    val_preds_list = []
    for epoch in range(n_epochs_total):
        val_preds_one_epoch = np.zeros([len(df_train), 30])
        for fold in range(len(histories)): # Use len(histories) instead of enumerate(fold_ids)
            # We need to know valid_idx for this fold. 
            # fold_ids is a list of (train_idx, valid_idx)
            _, valid_idx = fold_ids[fold]
            
            val_pred = histories[fold][0][epoch]
            val_preds_one_epoch[valid_idx, :] += val_pred
        val_preds_list.append(val_preds_one_epoch)

    oof_predictions = np.zeros((n_epochs_total, len(df_train), len(output_categories)), dtype=np.float32)
    for j, name in enumerate(output_categories):
        for epoch in range(n_epochs_total):
            oof_predictions[epoch, :, j] = val_preds_list[epoch][:, j]

    # 2. Get test preds per each epoch
    test_preds_list = []
    for epoch in range(n_epochs_total):
        test_preds_one_epoch = 0
        for fold in range(len(histories)):
            test_preds = histories[fold][1][epoch]
            test_preds_one_epoch += test_preds
        test_preds_one_epoch = test_preds_one_epoch / len(histories)
        test_preds_list.append(test_preds_one_epoch)

    test_predictions = np.zeros((n_epochs_total, len(df_test), len(output_categories)), dtype=np.float32)
    for j, name in enumerate(output_categories):
        for epoch in range(n_epochs_total):
            test_predictions[epoch, :, j] = test_preds_list[epoch][:, j]

    # LightGBM Stacking
    if len(histories) == Config.n_splits:
        print("Training LightGBM Stacking...")
        sub = pd.read_csv(os.path.join(Config.data_dir, Config.sub_csv))
            
        final_test_preds = []
        
        # Simple LightGBM wrapper
        lgb_params = {
            "boosting_type": "gbdt",
            "objective": "rmse",
            "learning_rate": 0.1,
            "max_depth": 1,
            "seed": 71,
            "verbose": -1
        }
        
        for i_col, col_name in enumerate(output_categories):
            y_train = compute_output_arrays(df_train, output_categories)[:, i_col]
            
            # Features: predictions from all epochs for this column
            # shape: (n_samples, n_epochs)
            x_train = oof_predictions[:, :, i_col].T 
            x_test = test_predictions[:, :, i_col].T
            
            # Notebook actually concatenates ALL columns from ALL epochs?
            # Notebook: x_train = pd.DataFrame(np.concatenate([oof_predictions[:, :, i].T for i in range(30)], axis=1))
            # This means for each target column, it uses predictions of ALL target columns from ALL epochs as features.
            # That's huge. Let's double check the notebook.
            # "x_train = pd.DataFrame(np.concatenate([oof_predictions[:, :, i].T for i in range(30)], axis=1))"
            # Yes, it concatenates everything.
            
            x_train_all = np.concatenate([oof_predictions[:, :, k].T for k in range(30)], axis=1)
            x_test_all = np.concatenate([test_predictions[:, :, k].T for k in range(30)], axis=1)
            
            # Train LightGBM
            # We need another CV here? Notebook uses `model.cv` which does CV internally.
            # To keep it simple and fast for the script, let's just train on full data and predict on test
            # OR replicate the CV logic if we want to be exact.
            # Notebook uses `fold_ids` for CV in LightGBM.
            
            test_preds_col = np.zeros(len(df_test))
            
            # CV for LightGBM
            for trn_idx, val_idx in fold_ids:
                x_trn_fold = x_train_all[trn_idx]
                y_trn_fold = y_train[trn_idx]
                x_val_fold = x_train_all[val_idx]
                y_val_fold = y_train[val_idx]
                
                d_train = lgb.Dataset(x_trn_fold, label=y_trn_fold)
                d_valid = lgb.Dataset(x_val_fold, label=y_val_fold)
                
                model = lgb.train(lgb_params, d_train, num_boost_round=5000, 
                                  valid_sets=[d_valid], 
                                  callbacks=[lgb.early_stopping(stopping_rounds=20)])
                
                test_preds_col += model.predict(x_test_all) / len(fold_ids)
                
            final_test_preds.append(test_preds_col)
            print(f"Finished LightGBM for {col_name}")
    else:
        print("Skipping LightGBM Stacking because not all folds were run.")
        # Use simple averaging of the last epoch predictions if LightGBM is skipped
        # Or better, use the average of all test predictions we computed above
        # The test_predictions array is (n_epochs, n_samples, n_targets)
        # Let's use the last epoch's prediction as a simple fallback
        final_test_preds = test_predictions[-1].T # (n_targets, n_samples)

    if Config.local_eval:
        sub = pd.DataFrame(columns=['qa_id'] + output_categories)
        sub['qa_id'] = df_test['qa_id']
        sub.iloc[:, 1:] = np.array(final_test_preds).T
    else:
        sub = pd.read_csv(os.path.join(Config.data_dir, Config.sub_csv))
        sub.iloc[:, 1:] = np.array(final_test_preds).T
    
    if Config.local_eval:
        print("\n" + "="*30)
        print("LOCAL EVALUATION RESULTS")
        print("="*30)
        local_score = compute_spearmanr(local_test_targets, sub.iloc[:, 1:].values)
        print(f"Simulated Public LB Score (Spearman): {local_score:.5f}")
        print("="*30 + "\n")
        sub.to_csv(os.path.join(Config.output_dir, "local_submission.csv"), index=False)
        print("Local submission saved to local_submission.csv")
        
        # Save final summary
        summary = {
            "local_spearman_score": float(local_score),
            "n_folds": Config.n_splits,
            "total_epochs_per_fold": Config.epochs_phase1 + Config.epochs_phase2,
            "training_complete": True,
            "completed_at": datetime.now().isoformat()
        }
    else:
        sub.to_csv(os.path.join(Config.output_dir, "submission.csv"), index=False)
        print("Submission saved to submission.csv")
        
        # Save final summary
        summary = {
            "n_folds": Config.n_splits,
            "total_epochs_per_fold": Config.epochs_phase1 + Config.epochs_phase2,
            "training_complete": True,
            "completed_at": datetime.now().isoformat()
        }
    
    with open(os.path.join(Config.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {os.path.join(Config.output_dir, 'summary.json')}")

if __name__ == "__main__":
    main()
