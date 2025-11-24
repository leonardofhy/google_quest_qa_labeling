import json
import pandas as pd
import os
import numpy as np

def analyze_label_distribution(file_path):
    """
    Analyzes the distribution of label scores in a Google QUEST QA Labeling JSONL file,
    focusing on the discrete nature of the data.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    data = []
    
    # Read the JSONL file
    print(f"Reading data from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not data:
        print("File is empty.")
        return

    # Convert to DataFrame for easier analysis
    df = pd.read_json(json.dumps(data))

    # List of known target labels based on the competition schema
    target_columns = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'
    ]

    # Filter columns that actually exist in the dataframe
    existing_targets = [col for col in target_columns if col in df.columns]

    if not existing_targets:
        print("No label columns found in the data.")
        return

    print(f"\nAnalyzed {len(df)} rows.")
    print("=" * 100)
    print(f"{'Label Name':<40} | {'Unique Vals':<11} | {'Top 5 Frequent Values (Value: Count)'}")
    print("=" * 100)

    for col in existing_targets:
        # Get value counts
        val_counts = df[col].value_counts().sort_index()
        unique_count = len(val_counts)
        
        # Get top 5 most frequent values
        top_5 = df[col].value_counts().nlargest(5)
        top_5_str = ", ".join([f"{val:.3f}: {count}" for val, count in top_5.items()])
        
        print(f"{col:<40} | {unique_count:<11} | {top_5_str}")

    print("=" * 100)
    
    # Optional: Attempt to guess the denominator (number of raters)
    # If values are like 0.333, 0.666, denominator is likely 3.
    # If values are like 0.2, 0.4, 0.6, denominator is likely 5.
    print("\nPotential Rater Count Analysis (Heuristic):")
    print("-" * 60)
    for col in existing_targets:
        unique_vals = df[col].unique()
        # Remove 0 and 1 to check intermediate steps
        intermediate_vals = [v for v in unique_vals if 0.0 < v < 1.0]
        
        if not intermediate_vals:
            print(f"{col:<40}: Binary (0 or 1 only)")
            continue
            
        # Check for common denominators
        min_diff = np.min(np.diff(sorted(unique_vals)))
        if min_diff > 0:
            estimated_denominator = round(1 / min_diff)
            print(f"{col:<40}: Step size ~{min_diff:.4f} (Likely {estimated_denominator} steps/raters)")
        else:
             print(f"{col:<40}: Single value or constant")

if __name__ == "__main__":
    # File path based on your request
    FILE_PATH = 'data/train.jsonl'
    
    # You might need to install pandas if you haven't: pip install pandas
    analyze_label_distribution(FILE_PATH)