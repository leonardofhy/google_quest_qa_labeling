import json
import os
import glob
import pandas as pd
from datetime import datetime

MODELS_DIR = "/home/leonardo298/Workspace/google_quest_qa_labeling/models"

def load_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def load_logs(path):
    if not os.path.exists(path):
        return []
    logs = []
    with open(path, 'r') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                pass
    return logs

def analyze_model(model_dir):
    config_path = os.path.join(model_dir, "config.json")
    log_path = os.path.join(model_dir, "training_log.jsonl")
    
    if not os.path.exists(config_path) or not os.path.exists(log_path):
        return None
        
    config = load_json(config_path)
    if not config:
        return None
        
    logs = load_logs(log_path)
    if not logs:
        return None
        
    # Extract key config parameters
    model_name = config.get("model_name", "unknown")
    
    # Handle epoch naming compatibility
    ep1 = config.get("epochs_phase1", config.get("q_epochs", 0))
    ep2 = config.get("epochs_phase2", config.get("epochs", 0))
    epochs = ep1 + ep2
    
    awp = config.get("awp_eps", 0.0)
    auto_weight = config.get("use_auto_weighting", False)
    loss_weights = f"BCE:{config.get('bce_weight',0)}/Rnk:{config.get('ranking_weight',0)}/Spr:{config.get('spearman_weight',0)}"
    
    # Metrics per fold
    fold_metrics = {}
    
    for log in logs:
        if "valid_spearman_avg" in log:
            fold = log.get("fold", -1)
            epoch = log.get("epoch", -1)
            phase = log.get("phase", "")
            
            score_avg = log.get("valid_spearman_avg", 0)
            score_last = log.get("valid_spearman_last", 0)
            
            if fold not in fold_metrics:
                fold_metrics[fold] = {
                    "best_avg": -1.0, 
                    "best_last": -1.0, 
                    "final_avg": -1.0,
                    "best_avg_epoch": -1,
                    "best_last_epoch": -1
                }
            
            # Track Best Avg (Ensemble)
            if score_avg > fold_metrics[fold]["best_avg"]:
                fold_metrics[fold]["best_avg"] = score_avg
                fold_metrics[fold]["best_avg_epoch"] = epoch
                
            # Track Best Last (Single Model)
            if score_last > fold_metrics[fold]["best_last"]:
                fold_metrics[fold]["best_last"] = score_last
                fold_metrics[fold]["best_last_epoch"] = epoch
                
            # Track Final (Latest entry)
            fold_metrics[fold]["final_avg"] = score_avg

    if not fold_metrics:
        return None

    n_folds = len(fold_metrics)
    
    # Calculate Cross-Validation Averages
    cv_best_avg = sum(m["best_avg"] for m in fold_metrics.values()) / n_folds
    cv_best_last = sum(m["best_last"] for m in fold_metrics.values()) / n_folds
    cv_final_avg = sum(m["final_avg"] for m in fold_metrics.values()) / n_folds
        
    return {
        "timestamp": os.path.basename(model_dir),
        "model": model_name,
        "epochs": epochs,
        "awp": awp,
        "auto_w": auto_weight,
        "loss": loss_weights,
        "cv_best_avg": cv_best_avg,
        "cv_best_last": cv_best_last,
        "cv_final_avg": cv_final_avg,
        "n_folds": n_folds
    }

def main():
    results = []
    model_dirs = sorted(glob.glob(os.path.join(MODELS_DIR, "202*")))
    
    print(f"Found {len(model_dirs)} model directories. Analyzing...")
    
    for d in model_dirs:
        res = analyze_model(d)
        if res:
            results.append(res)
            
    # Sort by Best Avg
    results.sort(key=lambda x: x["cv_best_avg"], reverse=True)
    
    # Create DataFrame for display
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No valid models found.")
        return

    # Reorder columns
    cols = ["timestamp", "cv_best_avg", "cv_best_last", "cv_final_avg", "n_folds", "model", "epochs", "awp", "auto_w", "loss"]
    df = df[cols]
    
    print("\n" + "="*140)
    print(f"{'MODEL PERFORMANCE RANKING (Sorted by Best Ensemble CV)':^140}")
    print("="*140)
    print(df.to_string(index=False, float_format=lambda x: "{:.5f}".format(x)))
    print("="*140)
    
    # Highlight the winner
    winner = df.iloc[0]
    print(f"\n🏆 BEST MODEL (Ensemble): {winner['timestamp']}")
    print(f"   CV Score (Avg):  {winner['cv_best_avg']:.5f}")
    print(f"   CV Score (Last): {winner['cv_best_last']:.5f}")
    print(f"   CV Score (Final):{winner['cv_final_avg']:.5f}")
    print(f"   Config: {winner['model']}, Epochs={winner['epochs']}, AWP={winner['awp']}, AutoWeight={winner['auto_w']}")
    print(f"   Loss Weights: {winner['loss']}")

if __name__ == "__main__":
    main()
