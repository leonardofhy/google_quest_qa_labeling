"""
Google Quest Q&A Labeling - Training & Inference

This script implements a DeBERTa-based model for multi-label classification of Q&A pairs.

Overview:
- Training: Fine-tune DeBERTa-v3-base on 30 Q&A quality labels
- Inference: Generate predictions on test data with post-processing
- Architecture: Weighted layer pooling + multi-sample dropout

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
import argparse
from collections import Counter
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

# Enable anomaly detection for debugging (can be disabled in production)
torch.autograd.set_detect_anomaly(False)


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    """Training and model configuration"""
    model_name = "microsoft/deberta-v3-base"
    
    max_len = 512
    batch_size = 16         
    accum_iter = 1          
    
    # Lower LR for stability with large batch size
    lr = 1e-5               # Reduced from 2e-5 to prevent gradient explosion
    head_lr = 5e-5          # Reduced from 1e-4
    
    # Training configuration
    epochs = 3              # Reduced for fast iteration
    n_folds = 1             # Single fold for fast testing (set to 5 for final training)
    validation_split = 0.1  # 10% validation split when n_folds = 1
    seed = 42
    num_workers = 8         # Increased from 2 for better data loading
    train_csv = "data/train.csv"
    test_csv = "data/test.csv"
    sample_submission_csv = "data/sample_submission.csv"
    output_dir = "models"
    
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
# 2. Dataset Class
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
        
        # Dynamic truncation with budget awareness
        budget = self.max_len - 3  # [CLS], [SEP], [SEP]
        if len(q_tokens) + len(a_tokens) > budget:
            half = budget // 2
            if len(a_tokens) > half and len(q_tokens) > half:
                a_tokens = a_tokens[:half]
                q_tokens = q_tokens[:budget - len(a_tokens)]
            elif len(a_tokens) <= half:
                q_tokens = q_tokens[:budget - len(a_tokens)]
            else:
                a_tokens = a_tokens[:budget - len(q_tokens)]
                
        # Build input IDs
        ids = [self.tokenizer.cls_token_id] + \
              self.tokenizer.convert_tokens_to_ids(q_tokens) + \
              [self.tokenizer.sep_token_id] + \
              self.tokenizer.convert_tokens_to_ids(a_tokens) + \
              [self.tokenizer.sep_token_id]
              
        mask = [1] * len(ids)
        padding_len = self.max_len - len(ids)
        ids = ids + [self.tokenizer.pad_token_id] * padding_len
        mask = mask + [0] * padding_len
        
        output = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }
        
        if self.mode != "test":
            # Add safety check and clipping with stricter bounds
            labels = self.targets[idx].copy()
            labels = np.clip(labels, 0.0, 1.0)  # Ensure labels are in [0, 1]
            labels = np.nan_to_num(labels, nan=0.5, posinf=1.0, neginf=0.0)  # Replace NaN/Inf
            # Additional check for any remaining invalid values
            if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
                labels = np.where(np.isnan(labels) | np.isinf(labels), 0.5, labels)

            output['labels'] = torch.tensor(labels, dtype=torch.float32)        
            
        return output


# ==========================================
# 3. Model Class
# ==========================================
class QuestDebertaModel(nn.Module):
    """DeBERTa model with weighted layer pooling and multi-sample dropout"""
    
    def __init__(self, model_name=Config.model_name, num_labels=30):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        
        # Weighted layer pooling with stable initialization
        n_weights = self.config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        weights_init.data[-1] = 0  # Give more weight to last layer initially
        self.layer_weights = nn.Parameter(weights_init)
        
        # Multi-sample dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.fc = nn.Linear(self.config.hidden_size, num_labels)
        # Removed Sigmoid for BCEWithLogitsLoss stability

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states 
        
        # Stack [CLS] tokens
        cls_outputs = torch.stack([layer[:, 0, :] for layer in hidden_states], dim=1)
        
        # Weighted sum
        weights = torch.softmax(self.layer_weights, dim=0).view(1, -1, 1)
        weighted_cls = (weights * cls_outputs).sum(dim=1)
        
        # Multi-sample dropout
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(weighted_cls)))
        avg_logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        
        return avg_logits


# ==========================================
# 4. Utilities
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
    optimizer.zero_grad()
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, mask)
        loss = loss_fn(outputs, labels)
        
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
        progress_bar.set_postfix({'loss': train_loss / (step + 1)})
    
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
        outputs = torch.sigmoid(outputs)
        
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
# 5. Training Pipeline (K-Fold with Multi-GPU)
# ==========================================
def train_loop():
    """Main training pipeline with GroupKFold and multi-GPU support"""
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(Config.train_csv)
    print(f"Total samples: {len(train_df)}\n")
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print()
    
    # Choose between K-Fold and simple split based on n_folds
    # Winner's strategy: Use question_title for grouping to better represent question topics
    if Config.n_folds == 1:
        # Fast single split for iteration testing
        print("Using single train/val split for fast iteration")
        print("Grouping by: question_title (winner's strategy)\n")
        gss = GroupShuffleSplit(n_splits=1, test_size=Config.validation_split, random_state=Config.seed)
        fold_splits = list(gss.split(train_df, train_df[Config.target_cols], groups=train_df['question_title']))
    else:
        # Full K-Fold for final training
        print(f"Using {Config.n_folds}-Fold Cross Validation")
        print("Grouping by: question_title (winner's strategy)\n")
        gkf = GroupKFold(n_splits=Config.n_folds)
        fold_splits = list(gkf.split(train_df, train_df[Config.target_cols], groups=train_df['question_title']))
    
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
        loss_fn = nn.BCEWithLogitsLoss()
        
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
        patience = 3  # Early stopping patience (increased for stability)

        for epoch in range(Config.epochs):
            # Training
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, Config.accum_iter)
            
            # Validation
            val_score, val_preds, _ = validate(model, val_loader, device)
            
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
        
        # Clean up
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    # Calculate overall CV score
    overall_score = compute_spearmanr(train_df[Config.target_cols].values, oof_preds)
    print(f"\n{'='*50}")
    print(f"Overall CV Spearman Score: {overall_score:.5f}")
    print(f"{'='*50}\n")
    
    # Save OOF predictions
    np.save(os.path.join(Config.output_dir, "oof_preds.npy"), oof_preds)
    
    return oof_preds


# ==========================================
# 6. Post-processing Utilities (Winning Solution Approach)
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
# 7. Inference Pipeline (Ensemble with Multi-GPU)
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
    
    for fold in range(1, Config.n_folds + 1):
        model_path = os.path.join(Config.output_dir, f"best_model_fold{fold}.pth")
        if not os.path.exists(model_path):
            print(f"⚠ Model for fold {fold} not found, skipping...")
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
# 8. Main Entry Point
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
    
    # Set seed
    seed_everything(Config.seed)
    
    print("="*50)
    print("Google Quest Q&A Labeling")
    print("="*50)
    print(f"Mode: {args.mode}")
    print(f"Model: {Config.model_name}")
    print(f"Epochs: {Config.epochs}")
    print(f"Folds: {Config.n_folds}")
    print(f"Batch Size: {Config.batch_size}")
    print(f"Post-processing: {'Disabled' if args.no_postprocessing else 'Enabled'}")
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