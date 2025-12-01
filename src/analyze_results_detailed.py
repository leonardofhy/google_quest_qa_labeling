import json
import os
import math

experiments = {
    "Exp 0 (Old Long)": "/home/leonardo298/Workspace/google_quest_qa_labeling/models/20251130_043303/training_log.jsonl",
    "Exp 1 (Baseline)": "/home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_005927/training_log.jsonl",
    "Exp 2 (Long Train)": "/home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_010011/training_log.jsonl",
    "Exp 3 (Hybrid Loss)": "/home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_010150/training_log.jsonl",
    "Exp 5 (Auto Weight)": "/home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_011054/training_log.jsonl",
    "Exp 6 (All-In)": "/home/leonardo298/Workspace/google_quest_qa_labeling/models/20251201_011650/training_log.jsonl",
}

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

print("="*100)
print(f"{'Experiment Analysis':^100}")
print("="*100)

# 1. Epoch-wise Improvement Analysis
print(f"\n{'1. Epoch-wise Improvement (Valid Spearman)':<50}")
print("-" * 100)

for name, path in experiments.items():
    logs = load_logs(path)
    if not logs:
        continue
        
    print(f"\n>> {name}")
    
    fold_data = {}
    for log in logs:
        if 'valid_spearman_avg' not in log: continue
        
        fold = log.get('fold', 0)
        phase = log.get('phase', 'unknown')
        epoch = log.get('epoch', 0)
        score = log.get('valid_spearman_avg', 0)
        
        if fold not in fold_data: fold_data[fold] = {}
        if phase not in fold_data[fold]: fold_data[fold][phase] = []
        
        fold_data[fold][phase].append((epoch, score))

    for fold in sorted(fold_data.keys()):
        print(f"  Fold {fold}:")
        for phase in ['question_only', 'qa']:
            if phase in fold_data[fold]:
                scores = sorted(fold_data[fold][phase], key=lambda x: x[0])
                score_strs = [f"Ep{e}:{s:.4f}" for e, s in scores]
                print(f"    {phase:<15}: " + " -> ".join(score_strs))

# 2. Best Performance per Fold Comparison
print(f"\n\n{'2. Best Performance per Fold Comparison':<50}")
print("-" * 120)
header = f"{'Fold':<5} | " + " | ".join([f"{name.split('(')[0].strip():<12}" for name in experiments])
print(header)
print("-" * 120)

folds = range(5)
best_scores = {f: {} for f in folds}
exp_averages = {name: [] for name in experiments}

for name, path in experiments.items():
    logs = load_logs(path)
    if not logs: 
        for f in folds: best_scores[f][name] = -1
        continue
    
    fold_max = {}
    for log in logs:
        if 'valid_spearman_avg' not in log: continue
        fold = log.get('fold', 0)
        score = log.get('valid_spearman_avg', 0)
        fold_max[fold] = max(fold_max.get(fold, -1), score)
        
    for f in folds:
        score = fold_max.get(f, -1)
        best_scores[f][name] = score
        if score != -1:
            exp_averages[name].append(score)

# Print Rows
for f in folds:
    row = f"{f:<5} | "
    for name in experiments:
        score = best_scores[f][name]
        if score != -1:
            row += f"{score:<12.4f} | "
        else:
            row += f"{'N/A':<12} | "
    print(row)

print("-" * 120)

# Calculate Average & Std Dev
avg_row = f"{'AVG':<5} | "
std_row = f"{'STD':<5} | "

for name in experiments:
    scores = exp_averages[name]
    if len(scores) > 0:
        avg = sum(scores) / len(scores)
        variance = sum([((x - avg) ** 2) for x in scores]) / len(scores)
        std = math.sqrt(variance)
        avg_row += f"{avg:.4f} ({len(scores)}) | "
        std_row += f"{std:<12.4f} | "
    else:
        avg_row += f"{'N/A':<12} | "
        std_row += f"{'N/A':<12} | "

print(avg_row)
print(std_row)
